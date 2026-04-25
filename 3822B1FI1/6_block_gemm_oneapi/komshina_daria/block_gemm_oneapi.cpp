#include "block_gemm_oneapi.h"

std::vector<float> GemmBlockONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        size_t size,
        sycl::device device) {
    if (size == 0 || a.size() != size * size || b.size() != size * size) {
        return {};
    }

    constexpr size_t BLOCK = 16;

    sycl::queue q(device, sycl::property::queue::in_order{});

    std::vector<float> c(size * size, 0.0f);

    float* d_a = sycl::malloc_device<float>(a.size(), q);
    float* d_b = sycl::malloc_device<float>(b.size(), q);
    float* d_c = sycl::malloc_device<float>(c.size(), q);

    if (!d_a || !d_b || !d_c) {
        sycl::free(d_a, q);
        sycl::free(d_b, q);
        sycl::free(d_c, q);
        return {};
    }

    q.memcpy(d_a, a.data(), a.size() * sizeof(float));
    q.memcpy(d_b, b.data(), b.size() * sizeof(float));
    q.fill(d_c, 0.0f, c.size());
    q.wait_and_throw();

    size_t global_size = ((size + BLOCK - 1) / BLOCK) * BLOCK;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 2> tile_a(
            sycl::range<2>(BLOCK, BLOCK),
            h
        );

        sycl::local_accessor<float, 2> tile_b(
            sycl::range<2>(BLOCK, BLOCK),
            h
        );

        h.parallel_for(
            sycl::nd_range<2>(
                sycl::range<2>(global_size, global_size),
                sycl::range<2>(BLOCK, BLOCK)
            ),
            [=](sycl::nd_item<2> item) {
                const size_t row = item.get_global_id(0);
                const size_t col = item.get_global_id(1);

                const size_t local_row = item.get_local_id(0);
                const size_t local_col = item.get_local_id(1);

                float sum = 0.0f;

                for (size_t block = 0; block < size; block += BLOCK) {
                    const size_t a_col = block + local_col;
                    const size_t b_row = block + local_row;

                    tile_a[local_row][local_col] =
                        (row < size && a_col < size)
                            ? d_a[row * size + a_col]
                            : 0.0f;

                    tile_b[local_row][local_col] =
                        (b_row < size && col < size)
                            ? d_b[b_row * size + col]
                            : 0.0f;

                    item.barrier(sycl::access::fence_space::local_space);

                    for (size_t k = 0; k < BLOCK; ++k) {
                        sum += tile_a[local_row][k] * tile_b[k][local_col];
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (row < size && col < size) {
                    d_c[row * size + col] = sum;
                }
            }
        );
    }).wait_and_throw();

    q.memcpy(c.data(), d_c, c.size() * sizeof(float)).wait_and_throw();

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);

    return c;
}
