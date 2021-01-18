// kernel_example.h
#ifndef KERNEL_EMBEDDINGBAG_H_
#define KERNEL_EMBEDDINGBAG_H_


namespace tensorflow {
namespace functor {

template <typename Device, typename T_indices>
struct EmbeddingBagFunctor {
  void operator()(const Device& d, const int value_dim, const int bag_dim, const int indices_size, const T_indices* __restrict__ indices, const float* __restrict__ values, const float* __restrict__ weights, float* __restrict__ out);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_EMBEDDINGBAG_H_
