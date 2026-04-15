#include "dev_jacobi_oneapi.h"

std::vector<float> JacobiDevONEAPI(
  const std::vector<float>& a, const std::vector<float>& b,
  float accuracy, sycl::device device) {
  size_t n = b.size();

  sycl::queue q(device);

  float* dev_a = sycl::malloc_device<float>(a.size(), q);
  float* dev_b = sycl::malloc_device<float>(n, q);
  float* dev_prev_x = sycl::malloc_device<float>(n, q);
  float* dev_cur_x = sycl::malloc_device<float>(n, q);
  float* dev_diff = sycl::malloc_device<float>(1, q);

  q.memcpy(dev_a, a.data(), a.size() * sizeof(float));
  q.memcpy(dev_b, b.data(), n * sizeof(float));
  q.memset(dev_prev_x, 0, n * sizeof(float));
  q.wait();

  float diff = 0.0f;

  for (size_t iteration = 0; iteration < ITERATIONS; iteration++) {
    diff = 0.0f;
    q.memcpy(dev_diff, &diff, sizeof(float)).wait();

    q.submit([&](sycl::handler& h) {
      auto reduction = sycl::reduction(dev_diff, sycl::plus<float>());

      h.parallel_for(sycl::range<1>(n), reduction, [=](sycl::id<1> idx, auto& sum_diff) {
        size_t i = idx[0];

        float res = 0.0f;
        for (size_t j = 0; j < n; j++) {
          if (i != j) {
            res += dev_a[i * n + j] * dev_prev_x[j];
          }
        }

        float new_x = (dev_b[i] - res) / dev_a[i * n + i];
        dev_cur_x[i] = new_x;
        sum_diff += (new_x - dev_prev_x[i]) * (new_x - dev_prev_x[i]);
        });
      });
    q.wait();

    q.memcpy(&diff, dev_diff, sizeof(float)).wait();

    if (diff < accuracy * accuracy)
      break;

    std::swap(dev_cur_x, dev_prev_x);
  }

  std::vector<float> result(n);
  q.memcpy(result.data(), dev_cur_x, n * sizeof(float)).wait();

  sycl::free(dev_a, q);
  sycl::free(dev_b, q);
  sycl::free(dev_prev_x, q);
  sycl::free(dev_cur_x, q);
  sycl::free(dev_diff, q);

  return result;
}