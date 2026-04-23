#include "mkl_gemm_oneapi.h"

#include <oneapi/mkl.hpp>
#include <vector>

std::vector<float> GemmMklONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {
    std::vector<float> c(size * size, 0.0f);

    sycl::queue queue(device);

    {
        sycl::buffer<float, 1> a_buffer(a.data(), sycl::range<1>(a.size()));
        sycl::buffer<float, 1> b_buffer(b.data(), sycl::range<1>(b.size()));
        sycl::buffer<float, 1> c_buffer(c.data(), sycl::range<1>(c.size()));

        const float alpha = 1.0f;
        const float beta = 0.0f;

        oneapi::mkl::blas::row_major::gemm(
            queue,
            oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans,
            size,
            size,
            size,
            alpha,
            a_buffer,
            size,
            b_buffer,
            size,
            beta,
            c_buffer,
            size
        );

        queue.wait();
    }

    return c;
}
