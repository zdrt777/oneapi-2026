#include "acc_jacobi_oneapi.h"

std::vector<float> JacobiAccONEAPI(const std::vector<float>& a,
                                   const std::vector<float>& b, 
								   float accuracy,
                                   sycl::device device) {
  sycl::queue queue(device);

  const size_t n = b.size();

  std::vector<float> x_old(n, 0.0f);
  std::vector<float> x_new(n, 0.0f);
  std::vector<float> inv_diag(n);

  for (size_t i = 0; i < n; ++i) {
    inv_diag[i] = 1.0f / a[i * n + i];
  }

  sycl::buffer<float> a_buf(a.data(), n * n);
  sycl::buffer<float> b_buf(b.data(), n);
  sycl::buffer<float> x_old_buf(x_old.data(), n);
  sycl::buffer<float> x_new_buf(x_new.data(), n);
  sycl::buffer<float> inv_diag_buf(inv_diag.data(), n);

  float error = 0.0f;
  sycl::buffer<float> error_buf(&error, 1);

  const size_t local_size = 128;
  const size_t global_size = ((n + local_size - 1) / local_size) * local_size;

  for (int iter = 0; iter < ITERATIONS; ++iter) {
    {
      auto error_acc = error_buf.get_host_access();
      error_acc[0] = 0.0f;
    }

    queue.submit([&](sycl::handler& handler) {
      auto a_acc = a_buf.get_access<sycl::access::mode::read>(handler);
      auto b_acc = b_buf.get_access<sycl::access::mode::read>(handler);
      auto x_old_acc =
          x_old_buf.get_access<sycl::access::mode::read>(handler);
      auto x_new_acc =
          x_new_buf.get_access<sycl::access::mode::write>(handler);
      auto inv_diag_acc =
          inv_diag_buf.get_access<sycl::access::mode::read>(handler);

      auto error_reduction =
          sycl::reduction(error_buf, handler, sycl::plus<float>());

      sycl::local_accessor<float, 1> x_tile(local_size, handler);

      handler.parallel_for(
          sycl::nd_range<1>(global_size, local_size), error_reduction,
          [=](sycl::nd_item<1> item, auto& error_sum) {
            const size_t row = item.get_global_id(0);
            const size_t local_id = item.get_local_id(0);
            const bool is_active = row < n;

            float sigma = 0.0f;

            for (size_t tile_begin = 0; tile_begin < n;
                 tile_begin += local_size) {
              const size_t col = tile_begin + local_id;

              x_tile[local_id] = (col < n) ? x_old_acc[col] : 0.0f;

              item.barrier(sycl::access::fence_space::local_space);

              const size_t tile_end = sycl::min(tile_begin + local_size, n);

              for (size_t j = tile_begin; j < tile_end; ++j) {
                if (is_active && j != row) {
                  sigma += a_acc[row * n + j] * x_tile[j - tile_begin];
                }
              }

              item.barrier(sycl::access::fence_space::local_space);
            }

            if (is_active) {
              const float new_value = (b_acc[row] - sigma) * inv_diag_acc[row];
              x_new_acc[row] = new_value;

              const float diff = sycl::fabs(new_value - x_old_acc[row]);
              error_sum += diff;
            }
          });
    });

    queue.wait();

    {
      auto error_acc = error_buf.get_host_access();
      error = error_acc[0];
    }

    if (error < accuracy) {
      break;
    }

    std::swap(x_old_buf, x_new_buf);
  }

  {
    auto result_acc = x_old_buf.get_host_access();
    for (size_t i = 0; i < n; ++i) {
      x_old[i] = result_acc[i];
    }
  }

  return x_old;
}