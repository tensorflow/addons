// kernel_example.h
#ifndef KERNEL_EMBEDDINGBAGBACKWARD_H_
#define KERNEL_EMBEDDINGBAGBACKWARD_H_


namespace tensorflow {
namespace functor {

// template <typename Device, typename T_indices>
template <typename Device, typename T_indices>
struct EmbeddingBagBackwardFunctor {
  void operator()(const Device& d, const int value_dim, const int bag_dim, const int indices_size, const int values_size, const T_indices* indices, const float* values, const float* weights, const float* dloss, float* values_grad, float* weights_grad, T_indices* dummy1, T_indices* dummy2);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_EMBEDDINGBAGBACKWARD_H_
