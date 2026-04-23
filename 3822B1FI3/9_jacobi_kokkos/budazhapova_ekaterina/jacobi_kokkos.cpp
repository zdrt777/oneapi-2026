#include "jacobi_kokkos.h"
#include <cmath>
#include <vector>

std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy) {

    size_t N = static_cast<size_t>(std::sqrt(a.size()));
    size_t total = N * N;
    Kokkos::View<float*, Kokkos::SYCLDeviceUSMSpace> A("A", total);
    Kokkos::View<float*, Kokkos::SYCLDeviceUSMSpace> B("B", N);
    Kokkos::View<float*, Kokkos::SYCLDeviceUSMSpace> x_old("x_old", N);
    Kokkos::View<float*, Kokkos::SYCLDeviceUSMSpace> x_new("x_new", N);

    Kokkos::deep_copy(A, Kokkos::View<const float*, Kokkos::HostSpace>(a.data(), total));
    Kokkos::deep_copy(B, Kokkos::View<const float*, Kokkos::HostSpace>(b.data(), N));
    Kokkos::deep_copy(x_old, 0.0f);
    Kokkos::deep_copy(x_new, 0.0f);

    float max_diff = 0.0f;
    int iter;

    for (iter = 0; iter < ITERATIONS; ++iter) {
        Kokkos::parallel_for("Jacobi_Iteration",
            Kokkos::RangePolicy<Kokkos::SYCL>(0, N),
            KOKKOS_LAMBDA(int i) {
                float sum = 0.0f;
                for (size_t j = 0; j < N; ++j) {
                    if (j != i) {
                        sum += A[i * N + j] * x_old[j];
                    }
                }
                x_new[i] = (B[i] - sum) / A[i * N + i];
            });
        Kokkos::fence();
        Kokkos::parallel_reduce("MaxDiff",
            Kokkos::RangePolicy<Kokkos::SYCL>(0, N),
            KOKKOS_LAMBDA(int i, float& lmax) {
                float diff = Kokkos::fabs(x_new[i] - x_old[i]);
                if (diff > lmax) lmax = diff;
            },
            Kokkos::Max<float>(max_diff));
        Kokkos::fence();

        if (max_diff < accuracy) break;
        Kokkos::parallel_for("Copy",
            Kokkos::RangePolicy<Kokkos::SYCL>(0, N),
            KOKKOS_LAMBDA(int i) {
                x_old[i] = x_new[i];
            });
        Kokkos::fence();
    }
    std::vector<float> result(N);
    Kokkos::deep_copy(Kokkos::View<float*, Kokkos::HostSpace>(result.data(), N), x_new);

    return result;
}