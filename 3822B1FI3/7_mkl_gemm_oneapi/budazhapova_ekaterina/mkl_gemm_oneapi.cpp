#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl.hpp>
#include <vector>

std::vector<float> GemmMklONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {

    sycl::queue q(device);
    size_t n = size;
    size_t total = n * n;

    std::vector<float> a_col(total), b_col(total);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            a_col[j * n + i] = a[i * n + j];
            b_col[j * n + i] = b[i * n + j];
        }
    }

    sycl::buffer<float, 1> buf_a(a_col.data(), sycl::range<1>(total));
    sycl::buffer<float, 1> buf_b(b_col.data(), sycl::range<1>(total));
    sycl::buffer<float, 1> buf_c(sycl::range<1>(total));

    float alpha = 1.0f, beta = 0.0f;
    oneapi::mkl::transpose trans = oneapi::mkl::transpose::nontrans;
    oneapi::mkl::blas::gemm(q, trans, trans, n, n, n, alpha, buf_a, n, buf_b, n, beta, buf_c, n);
    q.wait();

    std::vector<float> result(total);
    auto host_c = buf_c.get_host_access();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[i * n + j] = host_c[j * n + i];
        }
    }

    return result;
}