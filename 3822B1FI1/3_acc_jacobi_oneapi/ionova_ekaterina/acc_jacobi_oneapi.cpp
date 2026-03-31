#include "acc_jacobi_oneapi.h"
#include <cmath>
#include <utility>

std::vector<float> JacobiAccONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    float accuracy, sycl::device device) {

    const size_t n = b.size();
    const float eps_sq = accuracy * accuracy;
    
    sycl::queue q{device, sycl::property::queue::in_order{}};

    std::vector<float> x_host(n, 0.0f);
    
    sycl::buffer<float, 1> buf_a{a.data(), sycl::range<1>(a.size())};
    sycl::buffer<float, 1> buf_b{b.data(), sycl::range<1>(n)};
    sycl::buffer<float, 1> buf_curr{x_host.data(), sycl::range<1>(n)};
    sycl::buffer<float, 1> buf_next{sycl::range<1>(n)};

    for (int k = 0; k < ITERATIONS; ++k) {
        float iter_diff_sq = 0.0f;
        sycl::buffer<float, 1> buf_diff{&iter_diff_sq, 1};

        q.submit([&](sycl::handler& h) {
            auto A = buf_a.get_access<sycl::access::mode::read>(h);
            auto B = buf_b.get_access<sycl::access::mode::read>(h);
            auto X = buf_curr.get_access<sycl::access::mode::read>(h);
            auto Xn = buf_next.get_access<sycl::access::mode::write>(h);

            auto red = sycl::reduction(buf_diff, h, sycl::plus<float>());

            h.parallel_for(sycl::range<1>(n), red, [=](sycl::id<1> id, auto& error_sum) {
                size_t i = id[0];
                float sigma = 0.0f;
                size_t row_start = i * n;

                for (size_t j = 0; j < n; ++j) {
                    if (i != j) {
                        sigma += A[row_start + j] * X[j];
                    }
                }
                
                float res = (B[i] - sigma) / A[row_start + i];
                Xn[i] = res;

                float delta = res - X[i];
                error_sum += delta * delta;
            });
        });

        {
            auto diff_acc = buf_diff.get_host_access();
            if (diff_acc[0] < eps_sq) break;
        }

        std::swap(buf_curr, buf_next);
    }

    std::vector<float> final_res(n);
    auto last_acc = buf_curr.get_host_access();
    for (size_t i = 0; i < n; ++i) final_res[i] = last_acc[i];

    return final_res;
}