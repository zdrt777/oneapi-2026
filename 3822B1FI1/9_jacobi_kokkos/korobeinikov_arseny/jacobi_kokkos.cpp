#include "jacobi_kokkos.h"

#include <Kokkos_MathematicalFunctions.hpp>
#include <vector>

std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy) {
    using execution_space = Kokkos::SYCL;
    using memory_space = Kokkos::SYCLDeviceUSMSpace;
    using layout_type = Kokkos::LayoutRight;

    const int n = static_cast<int>(b.size());

    Kokkos::View<float**, layout_type, memory_space> matrix("matrix", n, n);
    Kokkos::View<float*, memory_space> rhs("rhs", n);
    Kokkos::View<float*, memory_space> current("current", n);
    Kokkos::View<float*, memory_space> next("next", n);

    auto matrix_host = Kokkos::create_mirror_view(matrix);
    auto rhs_host = Kokkos::create_mirror_view(rhs);

    for (int i = 0; i < n; ++i) {
        rhs_host(i) = b[i];
        for (int j = 0; j < n; ++j) {
            matrix_host(i, j) = a[i * n + j];
        }
    }

    Kokkos::deep_copy(matrix, matrix_host);
    Kokkos::deep_copy(rhs, rhs_host);
    Kokkos::deep_copy(current, 0.0f);

    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
        Kokkos::parallel_for(
            "jacobi_step",
            Kokkos::RangePolicy<execution_space>(0, n),
            KOKKOS_LAMBDA(int i) {
                float sum = 0.0f;
                for (int j = 0; j < n; ++j) {
                    if (j != i) {
                        sum += matrix(i, j) * current(j);
                    }
                }

                next(i) = (rhs(i) - sum) / matrix(i, i);
            });

        float max_diff = 0.0f;

        Kokkos::parallel_reduce(
            "jacobi_diff",
            Kokkos::RangePolicy<execution_space>(0, n),
            KOKKOS_LAMBDA(int i, float& local_max) {
                float diff = Kokkos::fabs(next(i) - current(i));
                if (diff > local_max) {
                    local_max = diff;
                }
            },
            Kokkos::Max<float>(max_diff));

        Kokkos::kokkos_swap(current, next);

        if (max_diff < accuracy) {
            break;
        }
    }

    auto result_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), current);

    std::vector<float> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = result_host(i);
    }

    return result;
}
