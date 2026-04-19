#include "shared_jacobi_oneapi.h"

std::vector<float> JacobiSharedONEAPI(const std::vector<float>& a,
                                      const std::vector<float>& b,
                                      float accuracy, 
									  sycl::device device) {
  const size_t n = b.size();

  if (n == 0 || a.size() != n * n) {
    return {};
  }

  sycl::queue queue(device);

  float* a_shared = sycl::malloc_shared<float>(a.size(), queue);
  float* b_shared = sycl::malloc_shared<float>(b.size(), queue);
  float* x_shared = sycl::malloc_shared<float>(n, queue);
  float* x_new_shared = sycl::malloc_shared<float>(n, queue);

  for (size_t i = 0; i < a.size(); ++i) {
    a_shared[i] = a[i];
  }

  for (size_t i = 0; i < n; ++i) {
    b_shared[i] = b[i];
    x_shared[i] = 0.0f;
  }

  bool converged = false;
  int iter = 0;

  while (!converged && iter < ITERATIONS) {
    queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> index) {
      const size_t row = index[0];
      float sum = 0.0f;
      const float diag = a_shared[row * n + row];

      for (size_t j = 0; j < n; ++j) {
        if (j != row) {
          sum += a_shared[row * n + j] * x_shared[j];
        }
      }

      x_new_shared[row] = (b_shared[row] - sum) / diag;
    });

    queue.wait();

    converged = true;

    for (size_t i = 0; i < n; ++i) {
      const float diff = std::fabs(x_new_shared[i] - x_shared[i]);

      if (diff >= accuracy) {
        converged = false;
      }

      x_shared[i] = x_new_shared[i];
    }

    ++iter;
  }

  std::vector<float> result(n);
  for (size_t i = 0; i < n; ++i) {
    result[i] = x_shared[i];
  }

  sycl::free(a_shared, queue);
  sycl::free(b_shared, queue);
  sycl::free(x_shared, queue);
  sycl::free(x_new_shared, queue);

  return result;
}