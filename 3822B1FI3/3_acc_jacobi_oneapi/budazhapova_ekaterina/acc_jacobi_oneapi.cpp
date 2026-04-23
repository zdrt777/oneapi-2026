#include "acc_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {

    size_t N = static_cast<size_t>(std::sqrt(a.size()));
    sycl::queue q(device);
    sycl::buffer<float> buf_a(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<float> buf_b(b.data(), sycl::range<1>(N));
    sycl::buffer<float> buf_x_old(sycl::range<1>(N));
    sycl::buffer<float> buf_x_new(sycl::range<1>(N));

    q.submit([&](sycl::handler& h) {
        auto x_old = buf_x_old.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) { x_old[i] = 0.0f; });
    });

    int iter = 0;
    float max_diff = 0.0f;

    for (iter = 0; iter < ITERATIONS; ++iter) {
        q.submit([&](sycl::handler& h) {
            auto A = buf_a.get_access<sycl::access::mode::read>(h);
            auto B = buf_b.get_access<sycl::access::mode::read>(h);
            auto x_old = buf_x_old.get_access<sycl::access::mode::read>(h);
            auto x_new = buf_x_new.get_access<sycl::access::mode::write>(h);
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

        auto x_old_host = buf_x_old.get_host_access();
        auto x_new_host = buf_x_new.get_host_access();
        max_diff = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            float diff = std::fabs(x_new_host[i] - x_old_host[i]);
            if (diff > max_diff) max_diff = diff;
        }

        if (max_diff < accuracy) break;
        q.submit([&](sycl::handler& h) {
            auto x_old = buf_x_old.get_access<sycl::access::mode::write>(h);
            auto x_new = buf_x_new.get_access<sycl::access::mode::read>(h);
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                x_old[i] = x_new[i];
            });
        });
        q.wait();
    }
    std::vector<float> result(N);
    auto res_acc = buf_x_new.get_host_access();
    for (size_t i = 0; i < N; ++i) result[i] = res_acc[i];
    return result;
}