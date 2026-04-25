#include "shared_jacobi_oneapi.h"
#include <cmath>
#include <cstring>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {

    size_t N = static_cast<size_t>(std::sqrt(a.size()));
    sycl::queue q(device);
    float* A = sycl::malloc_shared<float>(a.size(), q);
    float* B = sycl::malloc_shared<float>(N, q);
    float* x_old = sycl::malloc_shared<float>(N, q);
    float* x_new = sycl::malloc_shared<float>(N, q);

    std::memcpy(A, a.data(), a.size() * sizeof(float));
    std::memcpy(B, b.data(), N * sizeof(float));
    std::memset(x_old, 0, N * sizeof(float));

    int iter = 0;
    float max_diff = 0.0f;

    for (iter = 0; iter < ITERATIONS; ++iter) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                size_t row = i[0];
                float sum = 0.0f;
                for (size_t j = 0; j < N; ++j) {
                    if (j != row)
                        sum += A[row * N + j] * x_old[j];
                }
                x_new[row] = (B[row] - sum) / A[row * N + row];
            });
        });
        q.wait();
        max_diff = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            float diff = std::fabs(x_new[i] - x_old[i]);
            if (diff > max_diff) max_diff = diff;
        }

        if (max_diff < accuracy) break;
        std::memcpy(x_old, x_new, N * sizeof(float));
    }

    std::vector<float> result(N);
    std::memcpy(result.data(), x_new, N * sizeof(float));
    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(x_old, q);
    sycl::free(x_new, q);

    return result;
}