#include "block_gemm_oneapi.h"

std::vector<float> GemmBlockONEAPI(
  const std::vector<float>& a, const std::vector<float>& b,
  size_t size, sycl::device device) {
  const size_t block_size = 16;

  sycl::queue q(device);

  std::vector<float> c(size * size, 0.0f);

  sycl::buffer<float, 1> bufA(a.data(), size * size);
  sycl::buffer<float, 1> bufB(b.data(), size * size);
  sycl::buffer<float, 1> bufC(c.data(), size * size);

  size_t blocks = size / block_size;

  size_t padded = ((size + block_size - 1) / block_size) * block_size;
  sycl::range<2> global(padded, padded);
  sycl::range<2> local(block_size, block_size);

  q.submit([&](sycl::handler& h) {
    auto A = bufA.get_access<sycl::access::mode::read>(h);
    auto B = bufB.get_access<sycl::access::mode::read>(h);
    auto C = bufC.get_access<sycl::access::mode::write>(h);
    sycl::local_accessor<float, 2> blockA(sycl::range<2>(block_size, block_size), h);
    sycl::local_accessor<float, 2> blockB(sycl::range<2>(block_size, block_size), h);

    h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
      size_t global_row = item.get_global_id(0);
      size_t global_col = item.get_global_id(1);

      size_t local_row = item.get_local_id(0);
      size_t local_col = item.get_local_id(1);

      float sum = 0.0f;

      for (size_t k = 0; k < size; k += block_size) {
        if(global_row < size && (k + local_col) < size) {
          blockA[local_row][local_col] = A[global_row * size + (k + local_col)];
        } else {
          blockA[local_row][local_col] = 0.0f;
        }

        if (global_col < size && (k + local_row) < size) {
          blockB[local_row][local_col] = B[(k + local_row) * size + global_col];
        } else {
          blockB[local_row][local_col] = 0.0f;
        }

        item.barrier(sycl::access::fence_space::local_space);

        for (size_t K = 0; K < block_size; K++) {
          sum += blockA[local_row][K] * blockB[K][local_col];
        }

        item.barrier(sycl::access::fence_space::local_space);
      }

      if (global_row < size && global_col < size) {
        C[global_row * size + global_col] = sum;
      }
      });
    });

  q.wait();
  return c;
}