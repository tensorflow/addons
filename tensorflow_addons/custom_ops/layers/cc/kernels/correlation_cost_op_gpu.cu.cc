/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "gpu/cub/device/device_reduce.cuh"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow_addons/custom_ops/layers/cc/kernels/correlation_cost_op.h"

namespace tensorflow {
namespace addons {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

namespace {

/* There are two ways to implement the correlation layer:
- pad first and then compute cross-correlation costs (faster)
- have if-else in the computation of cross-correlation costs

This implementation is inspired from
https://github.com/NVIDIA/flownet2-pytorch
*/

template <unsigned int THREADS_PER_BLOCK>
__global__ void pad_and_transpose(const float *__restrict__ input,
                                  float *__restrict__ output, int C, int H,
                                  int W, int P) {
  // NCHW -> pad(NHWC)
  const int n = blockIdx.x;
  const int h = blockIdx.y;
  const int w = blockIdx.z;
  const int c0 = threadIdx.x;
  const int pW = (W + 2 * P);
  const int pH = (H + 2 * P);

  for (int c = c0; c < C; c += THREADS_PER_BLOCK) {
    output[n * (C * pH * pW) + (h + P) * (pW * C) + (w + P) * C + c] =
        ldg(input + n * (C * H * W) + c * (H * W) + h * W + w);
  }
}

template <unsigned int THREADS_PER_BLOCK>
__global__ void pad_and_no_transpose(const float *__restrict__ input,
                                     float *__restrict__ output, int C, int H,
                                     int W, int P) {
  // NHWC -> pad(NHWC)
  const int n = blockIdx.x;
  const int h = blockIdx.y;
  const int w = blockIdx.z;
  const int c0 = threadIdx.x;
  const int pW = (W + 2 * P);
  const int pH = (H + 2 * P);

  for (int c = c0; c < C; c += THREADS_PER_BLOCK) {
    output[n * (C * pH * pW) + (h + P) * (pW * C) + (w + P) * C + c] =
        ldg(input + n * (C * H * W) + h * (W * C) + w * C + c);
  }
}

template <unsigned int THREADS_PER_BLOCK>
__global__ void Correlation_forward(float *__restrict__ output, int Cout,
                                    int Hout, int Wout,
                                    const float *__restrict__ pInput1, int Cin,
                                    int Hin, int Win,
                                    const float *__restrict__ pInput2, int pad,
                                    int kernel_size, int max_displacement,
                                    int stride1, int stride2) {
  const int pWin = Win + 2 * pad;
  const int pHin = Hin + 2 * pad;

  const int kernel_rad = (kernel_size - 1) / 2;
  const int displacement_rad = max_displacement / stride2;
  const int displacement_size = 2 * displacement_rad + 1;

  const int n = blockIdx.x;
  const int h1 = blockIdx.y * stride1 + max_displacement + kernel_rad;
  const int w1 = blockIdx.z * stride1 + max_displacement + kernel_rad;
  const int c = threadIdx.x;

  const int K = kernel_size * kernel_size * Cin;

  typedef cub::WarpReduce<float> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_sum_storage;
  float thread_accumulation = 0;

  for (int tj = -displacement_rad; tj <= displacement_rad; ++tj) {
    for (int ti = -displacement_rad; ti <= displacement_rad; ++ti) {
      thread_accumulation = 0;
      int w2 = w1 + ti * stride2;
      int h2 = h1 + tj * stride2;

      for (int j = -kernel_rad; j <= kernel_rad; ++j) {
        for (int i = -kernel_rad; i <= kernel_rad; ++i) {
          for (int ch = c; ch < Cin; ch += THREADS_PER_BLOCK) {
            const int indx1 = n * (pHin * pWin * Cin) +
                              (h1 + j) * (pWin * Cin) + (w1 + i) * Cin + ch;
            const int indx2 = n * (pHin * pWin * Cin) +
                              (h2 + j) * (pWin * Cin) + (w2 + i) * Cin + ch;
            thread_accumulation += ldg(pInput1 + indx1) * ldg(pInput2 + indx2);
          }
        }
      }
      __syncthreads();

      // THREADS_PER_BLOCK==32, hence there is only one warp per block
      const float reduce_sum =
          WarpReduce(temp_sum_storage).Sum(thread_accumulation);
      if (c == 0) {
        const int tc = (tj + displacement_rad) * displacement_size +
                       (ti + displacement_rad);
        const int tindx = n * (Cout * Hout * Wout) + tc * (Hout * Wout) +
                          blockIdx.y * Wout + blockIdx.z;
        output[tindx] = reduce_sum / K;
      }
    }
  }
}

template <unsigned int THREADS_PER_BLOCK>
__global__ void Correlation_backward_input1(
    int item, float *__restrict__ gradInput1, int Cin, int Hin, int Win,
    const float *__restrict__ gradOutput, int Cout, int Hout, int Wout,
    const float *__restrict__ rInput2, int pad_size, int kernel_size,
    int max_displacement, int stride1, int stride2, bool is_NCHW) {
  const int n = item;
  const int h = blockIdx.x * stride1 + pad_size;
  const int w = blockIdx.y * stride1 + pad_size;
  const int c = blockIdx.z;
  const int t0 = threadIdx.x;

  const int kernel_rad = (kernel_size - 1) / 2;
  const int displacement_rad = max_displacement / stride2;
  const int displacement_size = 2 * displacement_rad + 1;

  int Wmin = (w - kernel_rad - max_displacement) / stride1;
  int Hmin = (h - kernel_rad - max_displacement) / stride1;

  int Wmax = (w + kernel_rad - max_displacement) / stride1;
  int Hmax = (h + kernel_rad - max_displacement) / stride1;

  if (Wmax < 0 || Hmax < 0 || Wmin >= Wout || Hmin >= Hout) {
    // assumes gradInput1 is pre-allocated and zero filled
    return;
  }

  if (Wmin > Wmax || Hmin > Hmax) {
    // assumes gradInput1 is pre-allocated and zero filled
    return;
  }

  Wmin = max(0, Wmin);
  Wmax = min(Wout - 1, Wmax);

  Hmin = max(0, Hmin);
  Hmax = min(Hout - 1, Hmax);

  const int pWin = Win + 2 * pad_size;
  const int pHin = Hin + 2 * pad_size;
  const float nelems = kernel_size * kernel_size * Cin;

  typedef cub::WarpReduce<float> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_sum_storage;
  float thread_accumulation = 0;

  for (int tc = t0; tc < Cout; tc += THREADS_PER_BLOCK) {
    int i2 = (tc % displacement_size - displacement_rad) * stride2;
    int j2 = (tc / displacement_size - displacement_rad) * stride2;

    const int indx2 =
        n * (pHin * pWin * Cin) + (h + j2) * (pWin * Cin) + (w + i2) * Cin + c;

    const float val2 = ldg(rInput2 + indx2);

    for (int j = Hmin; j <= Hmax; ++j) {
      for (int i = Wmin; i <= Wmax; ++i) {
        const int tindx =
            n * (Cout * Hout * Wout) + tc * (Hout * Wout) + j * Wout + i;
        thread_accumulation += ldg(gradOutput + tindx) * val2;
      }
    }
  }
  __syncthreads();

  // THREADS_PER_BLOCK==32, hence there is only one warp per block
  const float reduce_sum =
      WarpReduce(temp_sum_storage).Sum(thread_accumulation);
  if (t0 == 0) {
    if (is_NCHW) {
      const int indx1 = n * (Cin * Hin * Win) + c * (Hin * Win) +
                        (h - pad_size) * Win + (w - pad_size);
      gradInput1[indx1] = reduce_sum / nelems;
    } else {
      const int indx1 = n * (Cin * Hin * Win) + (h - pad_size) * (Win * Cin) +
                        (w - pad_size) * Cin + c;
      gradInput1[indx1] = reduce_sum / nelems;
    }
  }
}

template <unsigned int THREADS_PER_BLOCK>
__global__ void Correlation_backward_input2(
    int item, float *__restrict__ gradInput2, int Cin, int Hin, int Win,
    const float *__restrict__ gradOutput, int Cout, int Hout, int Wout,
    const float *rInput1, int pad_size, int kernel_size, int max_displacement,
    int stride1, int stride2, bool is_NCHW) {
  const int n = item;
  const int h = blockIdx.x * stride1 + pad_size;
  const int w = blockIdx.y * stride1 + pad_size;
  const int c = blockIdx.z;
  const int t0 = threadIdx.x;

  const int kernel_rad = (kernel_size - 1) / 2;
  const int displacement_rad = max_displacement / stride2;
  const int displacement_size = 2 * displacement_rad + 1;

  const int pWin = Win + 2 * pad_size;
  const int pHin = Hin + 2 * pad_size;
  const float nelems = kernel_size * kernel_size * Cin;

  typedef cub::WarpReduce<float> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_sum_storage;
  float thread_accumulation = 0;

  for (int tc = t0; tc < Cout; tc += THREADS_PER_BLOCK) {
    const int i2 = (tc % displacement_size - displacement_rad) * stride2;
    const int j2 = (tc / displacement_size - displacement_rad) * stride2;

    int Wmin = (w - kernel_rad - max_displacement - i2) / stride1;
    int Hmin = (h - kernel_rad - max_displacement - j2) / stride1;

    int Wmax = (w + kernel_rad - max_displacement - i2) / stride1;
    int Hmax = (h + kernel_rad - max_displacement - j2) / stride1;

    if (Wmax < 0 || Hmax < 0 || Wmin >= Wout || Hmin >= Hout) {
      // assumes gradInput2 is pre-allocated and zero filled
      continue;
    }

    if (Wmin > Wmax || Hmin > Hmax) {
      // assumes gradInput2 is pre-allocated and zero filled
      continue;
    }

    Wmin = max(0, Wmin);
    Wmax = min(Wout - 1, Wmax);

    Hmin = max(0, Hmin);
    Hmax = min(Hout - 1, Hmax);

    const int indx1 =
        n * (pHin * pWin * Cin) + (h - j2) * (pWin * Cin) + (w - i2) * Cin + c;
    const float val1 = ldg(rInput1 + indx1);

    for (int j = Hmin; j <= Hmax; ++j) {
      for (int i = Wmin; i <= Wmax; ++i) {
        const int tindx =
            n * (Cout * Hout * Wout) + tc * (Hout * Wout) + j * Wout + i;
        thread_accumulation += ldg(gradOutput + tindx) * val1;
      }
    }
  }
  __syncthreads();

  const float reduce_sum =
      WarpReduce(temp_sum_storage).Sum(thread_accumulation);
  if (t0 == 0) {
    if (is_NCHW) {
      const int indx2 = n * (Cin * Hin * Win) + c * (Hin * Win) +
                        (h - pad_size) * (Win) + (w - pad_size);
      gradInput2[indx2] = reduce_sum / nelems;
    } else {
      const int indx2 = n * (Cin * Hin * Win) + (h - pad_size) * (Win * Cin) +
                        (w - pad_size) * Cin + c;
      gradInput2[indx2] = reduce_sum / nelems;
    }
  }
}

};  // namespace

template <typename Dtype>
struct CorrelationCostFunctor<GPUDevice, Dtype> {
  Status operator()(OpKernelContext *context, const Tensor &input_a_t,
                    const Tensor &input_b_t, Tensor *output_t,
                    /* params */
                    int kernel_size, int max_displacement, int stride_1,
                    int stride_2, int pad, TensorFormat data_format) {
    // do not change: the CUDA kernels expects THREADS_PER_BLOCK==32
    const int THREADS_PER_BLOCK = 32;

    const int32 N = GetTensorDim(input_a_t, data_format, 'N');
    const int32 iC = GetTensorDim(input_a_t, data_format, 'C');
    const int32 iH = GetTensorDim(input_a_t, data_format, 'H');
    const int32 iW = GetTensorDim(input_a_t, data_format, 'W');

    Tensor padded_a_t;
    Tensor padded_b_t;
    TensorShape padded_shape({N, iH + 2 * pad, iW + 2 * pad, iC});
    TF_RETURN_IF_ERROR(context->allocate_temp(DataTypeToEnum<Dtype>::value,
                                              padded_shape, &padded_a_t));
    TF_RETURN_IF_ERROR(context->allocate_temp(DataTypeToEnum<Dtype>::value,
                                              padded_shape, &padded_b_t));

    dim3 blocks_grid(N, iH, iW);
    dim3 threads_block(THREADS_PER_BLOCK);

    // the output is always NCHW (python transposes it to NHWC)
    const int32 oC = GetTensorDim(*output_t, FORMAT_NCHW, 'C');
    const int32 oH = GetTensorDim(*output_t, FORMAT_NCHW, 'H');
    const int32 oW = GetTensorDim(*output_t, FORMAT_NCHW, 'W');

    // set everything to zero (we zero-pad)
    cudaMemset(padded_a_t.flat<Dtype>().data(), 0,
               padded_a_t.NumElements() * sizeof(Dtype));
    cudaMemset(padded_b_t.flat<Dtype>().data(), 0,
               padded_b_t.NumElements() * sizeof(Dtype));
    cudaMemset(output_t->flat<Dtype>().data(), 0,
               output_t->NumElements() * sizeof(Dtype));

    const bool is_NCHW = (data_format == FORMAT_NCHW);
    if (is_NCHW) {
      pad_and_transpose<THREADS_PER_BLOCK><<<blocks_grid, threads_block>>>(
          input_a_t.flat<Dtype>().data(), padded_a_t.flat<Dtype>().data(), iC,
          iH, iW, pad);
      pad_and_transpose<THREADS_PER_BLOCK><<<blocks_grid, threads_block>>>(
          input_b_t.flat<Dtype>().data(), padded_b_t.flat<Dtype>().data(), iC,
          iH, iW, pad);
    } else {
      pad_and_no_transpose<THREADS_PER_BLOCK><<<blocks_grid, threads_block>>>(
          input_a_t.flat<Dtype>().data(), padded_a_t.flat<Dtype>().data(), iC,
          iH, iW, pad);
      pad_and_no_transpose<THREADS_PER_BLOCK><<<blocks_grid, threads_block>>>(
          input_b_t.flat<Dtype>().data(), padded_b_t.flat<Dtype>().data(), iC,
          iH, iW, pad);
    }

    const GPUDevice &d = context->eigen_gpu_device();

    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    dim3 totalBlocksCorr(N, oH, oW);

    Correlation_forward<THREADS_PER_BLOCK>
        <<<totalBlocksCorr, threadsPerBlock, 0, d.stream()>>>(
            output_t->flat<Dtype>().data(), oC, oH, oW,
            padded_a_t.flat<Dtype>().data(), iC, iH, iW,
            padded_b_t.flat<Dtype>().data(), pad, kernel_size, max_displacement,
            stride_1, stride_2);

    return Status();
  }
};

template <typename Dtype>
struct CorrelationCostGradFunctor<GPUDevice, Dtype> {
  Status operator()(OpKernelContext *context, const Tensor &input_a_t,
                    const Tensor &input_b_t, const Tensor &topdiff_t,
                    Tensor *output_a_gradient_t, Tensor *output_b_gradient_t,
                    /* params */
                    int kernel_size, int max_displacement, int stride_1,
                    int stride_2, int pad, TensorFormat data_format) {
    // do not change: the CUDA kernels expects THREADS_PER_BLOCK==32
    const int THREADS_PER_BLOCK = 32;

    const int32 N = GetTensorDim(input_a_t, data_format, 'N');
    const int32 iC = GetTensorDim(input_a_t, data_format, 'C');
    const int32 iH = GetTensorDim(input_a_t, data_format, 'H');
    const int32 iW = GetTensorDim(input_a_t, data_format, 'W');

    Tensor padded_a_t;
    Tensor padded_b_t;
    TensorShape padded_shape({N, iH + 2 * pad, iW + 2 * pad, iC});
    TF_RETURN_IF_ERROR(context->allocate_temp(DataTypeToEnum<Dtype>::value,
                                              padded_shape, &padded_a_t));
    TF_RETURN_IF_ERROR(context->allocate_temp(DataTypeToEnum<Dtype>::value,
                                              padded_shape, &padded_b_t));

    dim3 blocks_grid(N, iH, iW);
    dim3 threads_block(THREADS_PER_BLOCK);

    // topdiff is NCHW
    const int32 oC = GetTensorDim(topdiff_t, FORMAT_NCHW, 'C');
    const int32 oH = GetTensorDim(topdiff_t, FORMAT_NCHW, 'H');
    const int32 oW = GetTensorDim(topdiff_t, FORMAT_NCHW, 'W');

    // set everything to zero (we zero-pad)
    cudaMemset(padded_a_t.flat<Dtype>().data(), 0,
               padded_a_t.NumElements() * sizeof(Dtype));
    cudaMemset(padded_b_t.flat<Dtype>().data(), 0,
               padded_b_t.NumElements() * sizeof(Dtype));
    cudaMemset(output_a_gradient_t->flat<Dtype>().data(), 0,
               output_a_gradient_t->NumElements() * sizeof(Dtype));
    cudaMemset(output_b_gradient_t->flat<Dtype>().data(), 0,
               output_b_gradient_t->NumElements() * sizeof(Dtype));

    const bool is_NCHW = (data_format == FORMAT_NCHW);
    if (is_NCHW) {
      pad_and_transpose<THREADS_PER_BLOCK><<<blocks_grid, threads_block>>>(
          input_a_t.flat<Dtype>().data(), padded_a_t.flat<Dtype>().data(), iC,
          iH, iW, pad);
      pad_and_transpose<THREADS_PER_BLOCK><<<blocks_grid, threads_block>>>(
          input_b_t.flat<Dtype>().data(), padded_b_t.flat<Dtype>().data(), iC,
          iH, iW, pad);
    } else {
      pad_and_no_transpose<THREADS_PER_BLOCK><<<blocks_grid, threads_block>>>(
          input_a_t.flat<Dtype>().data(), padded_a_t.flat<Dtype>().data(), iC,
          iH, iW, pad);
      pad_and_no_transpose<THREADS_PER_BLOCK><<<blocks_grid, threads_block>>>(
          input_b_t.flat<Dtype>().data(), padded_b_t.flat<Dtype>().data(), iC,
          iH, iW, pad);
    }

    const GPUDevice &d = context->eigen_gpu_device();

    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    dim3 totalBlocksCorr(iH, iW, iC);

    for (int n = 0; n < N; ++n) {
      Correlation_backward_input1<THREADS_PER_BLOCK>
          <<<totalBlocksCorr, threadsPerBlock>>>(
              n, output_a_gradient_t->flat<Dtype>().data(), iC, iH, iW,
              topdiff_t.flat<Dtype>().data(), oC, oH, oW,
              padded_b_t.flat<Dtype>().data(), pad, kernel_size,
              max_displacement, stride_1, stride_2, is_NCHW);
    }

    for (int n = 0; n < N; n++) {
      Correlation_backward_input2<THREADS_PER_BLOCK>
          <<<totalBlocksCorr, threadsPerBlock>>>(
              n, output_b_gradient_t->flat<Dtype>().data(), iC, iH, iW,
              topdiff_t.flat<Dtype>().data(), oC, oH, oW,
              padded_a_t.flat<Dtype>().data(), pad, kernel_size,
              max_displacement, stride_1, stride_2, is_NCHW);
    }

    return Status();
  }
};

template struct CorrelationCostFunctor<GPUDevice, float>;
template struct CorrelationCostGradFunctor<GPUDevice, float>;

}  // namespace functor
}  // namespace addons
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
