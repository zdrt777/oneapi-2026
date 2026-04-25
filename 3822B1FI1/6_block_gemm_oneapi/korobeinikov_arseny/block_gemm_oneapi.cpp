#include "block_gemm_oneapi.h"

#include <vector>

std::vector<float> GemmBlockONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {
    const std::size_t total = size * size;
    const std::size_t block_size = 16;

    std::vector<float> result(total, 0.0f);

    sycl::queue queue(device);

    float* a_dev = sycl::malloc_device<float>(total, queue);
    float* b_dev = sycl::malloc_device<float>(total, queue);
    float* c_dev = sycl::malloc_device<float>(total, queue);

    queue.memcpy(a_dev, a.data(), sizeof(float) * total);
    queue.memcpy(b_dev, b.data(), sizeof(float) * total);
    queue.fill(c_dev, 0.0f, total).wait();

    const std::size_t rounded =
        ((size + block_size - 1) / block_size) * block_size;

    sycl::range<2> local_range(block_size, block_size);
    sycl::range<2> global_range(rounded, rounded);

    queue.submit([&](sycl::handler& handler) {
        sycl::local_accessor<float, 2> a_tile(sycl::range<2>(block_size, block_size), handler);
        sycl::local_accessor<float, 2> b_tile(sycl::range<2>(block_size, block_size), handler);

        handler.parallel_for(sycl::nd_range<2>(global_range, local_range),
                             [=](sycl::nd_item<2> item) {
                                 const std::size_t row = item.get_global_id(0);
                                 const std::size_t col = item.get_global_id(1);
                                 const std::size_t local_row = item.get_local_id(0);
                                 const std::size_t local_col = item.get_local_id(1);

                                 float value = 0.0f;
                                 const std::size_t tiles = (size + block_size - 1) / block_size;

                                 for (std::size_t t = 0; t < tiles; ++t) {
                                     const std::size_t a_col = t * block_size + local_col;
                                     const std::size_t b_row = t * block_size + local_row;

                                     if (row < size && a_col < size) {
                                         a_tile[local_row][local_col] = a_dev[row * size + a_col];
                                     } else {
                                         a_tile[local_row][local_col] = 0.0f;
                                     }

                                     if (b_row < size && col < size) {
                                         b_tile[local_row][local_col] = b_dev[b_row * size + col];
                                     } else {
                                         b_tile[local_row][local_col] = 0.0f;
                                     }

                                     item.barrier(sycl::access::fence_space::local_space);

                                     for (std::size_t k = 0; k < block_size; ++k) {
                                         value += a_tile[local_row][k] * b_tile[k][local_col];
                                     }

                                     item.barrier(sycl::access::fence_space::local_space);
                                 }

                                 if (row < size && col < size) {
                                     c_dev[row * size + col] = value;
                                 }
                             });
    }).wait();

    queue.memcpy(result.data(), c_dev, sizeof(float) * total).wait();

    sycl::free(a_dev, queue);
    sycl::free(b_dev, queue);
    sycl::free(c_dev, queue);

    return result;
}
