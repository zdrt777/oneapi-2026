#include "dev_jacobi_oneapi.h"

std::vector<float> JacobiDevONEAPI(const std::vector<float>& a,
                                   const std::vector<float>& b, 
								   float accuracy,
                                   sycl::device device) {
  const size_t n = b.size();

  if (n == 0 || a.size() != n * n) {
    return {};
  }

  sycl::queue queue(device);

  float* a_dev = sycl::malloc_device<float>(a.size(), queue);
  float* b_dev = sycl::malloc_device<float>(b.size(), queue);
  float* x_dev = sycl::malloc_device<float>(n, queue);
  float* x_new_dev = sycl::malloc_device<float>(n, queue);

  queue.memcpy(a_dev, a.data(), sizeof(float) * a.size());
  queue.memcpy(b_dev, b.data(), sizeof(float) * b.size());
  queue.memset(x_dev, 0, sizeof(float) * n);
  queue.wait();

  std::vector<float> x(n);
  std::vector<float> x_new(n);

  bool converged = false;
  int iter = 0;

  while (!converged && iter < ITERATIONS) {
    queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> index) {
      const size_t row = index[0];
      float sum = 0.0f;
      const float diag = a_dev[row * n + row];

      for (size_t j = 0; j < n; ++j) {
        if (j != row) {
          sum += a_dev[row * n + j] * x_dev[j];
        }
      }

      x_new_dev[row] = (b_dev[row] - sum) / diag;
    });

    queue.wait();

    queue.memcpy(x_new.data(), x_new_dev, sizeof(float) * n).wait();
    queue.memcpy(x.data(), x_dev, sizeof(float) * n).wait();

    converged = true;

    for (size_t i = 0; i < n; ++i) {
      const float diff = std::fabs(x_new[i] - x[i]);

      if (diff >= accuracy) {
        converged = false;
      }

      x[i] = x_new[i];
    }

    queue.memcpy(x_dev, x.data(), sizeof(float) * n).wait();
    ++iter;
  }

  queue.memcpy(x.data(), x_dev, sizeof(float) * n).wait();

  sycl::free(a_dev, queue);
  sycl::free(b_dev, queue);
  sycl::free(x_dev, queue);
  sycl::free(x_new_dev, queue);

  return x;
}