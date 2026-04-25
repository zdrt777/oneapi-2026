#include "block_gemm_oneapi.h"

std::vector<float> GemmBlockONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {

    const size_t BLOCK_SIZE = 16;
    sycl::queue q(device);
    sycl::buffer<float, 2> buf_a(a.data(), sycl::range<2>(size, size));
    sycl::buffer<float, 2> buf_b(b.data(), sycl::range<2>(size, size));
    sycl::buffer<float, 2> buf_c(sycl::range<2>(size, size));

    size_t n_blocks = size / BLOCK_SIZE;

    q.submit([&](sycl::handler& h) {
        auto A = buf_a.get_access<sycl::access::mode::read>(h);
        auto B = buf_b.get_access<sycl::access::mode::read>(h);
        auto C = buf_c.get_access<sycl::access::mode::write>(h);

        sycl::accessor<float, 2, sycl::access::mode::read_write, sycl::access::target::local>
            local_A(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), h);
        sycl::accessor<float, 2, sycl::access::mode::read_write, sycl::access::target::local>
            local_B(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), h);

        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(size, size), sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE)),
            [=](sycl::nd_item<2> item) {
                int row = item.get_global_id(0);
                int col = item.get_global_id(1);
                int local_row = item.get_local_id(0);
                int local_col = item.get_local_id(1);
                int block_row = item.get_group(0);
                int block_col = item.get_group(1);

                float sum = 0.0f;
                for (int k = 0; k < n_blocks; ++k) {
                    int a_global_row = block_row * BLOCK_SIZE + local_row;
                    int a_global_col = k * BLOCK_SIZE + local_col;
                    local_A[local_row][local_col] = A[a_global_row][a_global_col];

                    int b_global_row = k * BLOCK_SIZE + local_row;
                    int b_global_col = block_col * BLOCK_SIZE + local_col;
                    local_B[local_row][local_col] = B[b_global_row][b_global_col];

                    item.barrier(sycl::access::fence_space::local_space);

                    for (int i = 0; i < BLOCK_SIZE; ++i) {
                        sum += local_A[local_row][i] * local_B[i][local_col];
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }
                C[row][col] = sum;
            }
        );
    });

    auto host_C = buf_c.get_host_access();
    std::vector<float> result(size * size);
    for (size_t i = 0; i < size; ++i)
        for (size_t j = 0; j < size; ++j)
            result[i * size + j] = host_C[i][j];

    return result;
}