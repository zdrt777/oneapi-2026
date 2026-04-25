#include "mkl_gemm_oneapi.h"

#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        size_t size,
        sycl::device device) {
    if (size == 0 || a.size() != size * size || b.size() != size * size) {
        return {};
    }

    sycl::queue q(device, sycl::property::queue::in_order{});

    const size_t elements = size * size;
    std::vector<float> result(elements, 0.0f);

    float* A = sycl::malloc_shared<float>(elements, q);
    float* B = sycl::malloc_shared<float>(elements, q);
    float* C = sycl::malloc_shared<float>(elements, q);

    if (!A || !B || !C) {
        sycl::free(A, q);
        sycl::free(B, q);
        sycl::free(C, q);
        return {};
    }

    for (size_t i = 0; i < elements; ++i) {
        A[i] = a[i];
        B[i] = b[i];
        C[i] = 0.0f;
    }

    oneapi::mkl::blas::row_major::gemm(
        q,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        size,
        size,
        size,
        1.0f,
        A,
        size,
        B,
        size,
        0.0f,
        C,
        size
    );

    q.wait_and_throw();

    for (size_t i = 0; i < elements; ++i) {
        result[i] = C[i];
    }

    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(C, q);

    return result;
}
