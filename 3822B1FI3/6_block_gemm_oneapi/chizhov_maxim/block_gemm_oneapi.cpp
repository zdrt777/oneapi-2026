#include "block_gemm_oneapi.h"
#include <vector>
#include <iostream>
#include <cassert>

std::vector<float> GemmBlockONEAPI(
    const std::vector<float>& a,
    const std::vector<float>& b,
    size_t size,
    sycl::device device)
{
    constexpr size_t BLOCK_SIZE = 16;

    assert(size % BLOCK_SIZE == 0);

    std::vector<float> c(size * size, 0.0f);

    sycl::queue q(device);

    {
        sycl::buffer<float, 1> a_buf(a.data(), sycl::range<1>(a.size()));
        sycl::buffer<float, 1> b_buf(b.data(), sycl::range<1>(b.size()));
        sycl::buffer<float, 1> c_buf(c.data(), sycl::range<1>(c.size()));

        size_t num_blocks = size / BLOCK_SIZE;

        q.submit([&](sycl::handler& h) {
            auto a_acc = a_buf.get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.get_access<sycl::access::mode::read>(h);
            auto c_acc = c_buf.get_access<sycl::access::mode::write>(h);

            sycl::local_accessor<float, 2> a_block(
                sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), h);
            sycl::local_accessor<float, 2> b_block(
                sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), h);

            h.parallel_for(
                sycl::nd_range<2>(
                    sycl::range<2>(size, size),
                    sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE)),
                [=](sycl::nd_item<2> item) {
                    int i = item.get_global_id(0);
                    int j = item.get_global_id(1);

                    int li = item.get_local_id(0);
                    int lj = item.get_local_id(1);

                    int block_i = i / BLOCK_SIZE;
                    int block_j = j / BLOCK_SIZE;

                    float sum = 0.0f;

                    for (size_t block_k = 0; block_k < num_blocks; block_k++) {
                        int a_row = block_i * BLOCK_SIZE + li;
                        int a_col = block_k * BLOCK_SIZE + lj;
                        a_block[li][lj] = a_acc[a_row * size + a_col];

                        int b_row = block_k * BLOCK_SIZE + li;
                        int b_col = block_j * BLOCK_SIZE + lj;
                        b_block[li][lj] = b_acc[b_row * size + b_col];

                        item.barrier(sycl::access::fence_space::local_space);

                        for (int k = 0; k < BLOCK_SIZE; k++) {
                            sum += a_block[li][k] * b_block[k][lj];
                        }

                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    c_acc[i * size + j] = sum;
                }
            );
        });

        q.wait();
    }

    return c;
}

