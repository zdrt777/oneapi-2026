#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
    const std::vector<float>& a, const std::vector<float>& b, 
    size_t size, sycl::device device) {
    
    std::vector<float> c(size * size, 0.0f);
    sycl::queue q(device);

    sycl::buffer<float, 1> buf_a{a.data(), sycl::range<1>(a.size())};
    sycl::buffer<float, 1> buf_b{b.data(), sycl::range<1>(b.size())};
    sycl::buffer<float, 1> buf_c{c.data(), sycl::range<1>(c.size())};

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    const std::int64_t n_mkl = static_cast<std::int64_t>(size);

    oneapi::mkl::blas::row_major::gemm(
        q, 
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        n_mkl, n_mkl, n_mkl, 
        alpha, 
        buf_a, n_mkl, 
        buf_b, n_mkl, 
        beta, 
        buf_c, n_mkl
    );

    q.wait(); 

    return c;
}