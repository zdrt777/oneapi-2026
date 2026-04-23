#include "dev_jacobi_oneapi.h"
#include <cmath>
#include <vector>

std::vector<float> JacobiDevONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
            
    size_t N = static_cast<size_t>(std::sqrt(a.size()));
    sycl::queue q(device);

    float* A_dev = sycl::malloc_device<float>(a.size(), q);
    float* b_dev = sycl::malloc_device<float>(N, q);
    float* x_old_dev = sycl::malloc_device<float>(N, q);
    float* x_new_dev = sycl::malloc_device<float>(N, q);

    q.memcpy(A_dev, a.data(), a.size() * sizeof(float));
    q.memcpy(b_dev, b.data(), N * sizeof(float));
    q.memset(x_old_dev, 0, N * sizeof(float));
    q.wait();

    int iter;
    float max_diff = 0.0f;

    for (iter = 0; iter < ITERATIONS; ++iter) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                size_t row = i[0];
                float sum = 0.0f;
                for (size_t j = 0; j < N; ++j) {
                    if (j != row)
                        sum += A_dev[row * N + j] * x_old_dev[j];
                }
                x_new_dev[row] = (b_dev[row] - sum) / A_dev[row * N + row];
            });
        });
        q.wait();

        std::vector<float> x_new_host(N);
        q.memcpy(x_new_host.data(), x_new_dev, N * sizeof(float)).wait();

        std::vector<float> x_old_host(N);
        q.memcpy(x_old_host.data(), x_old_dev, N * sizeof(float)).wait();
        max_diff = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            float diff = std::fabs(x_new_host[i] - x_old_host[i]);
            if (diff > max_diff) max_diff = diff;
        }

        if (max_diff < accuracy) break;
        q.memcpy(x_old_dev, x_new_dev, N * sizeof(float)).wait();
    }
    std::vector<float> result(N);
    q.memcpy(result.data(), x_new_dev, N * sizeof(float)).wait();

    sycl::free(A_dev, q);
    sycl::free(b_dev, q);
    sycl::free(x_old_dev, q);
    sycl::free(x_new_dev, q);

    return result;
}