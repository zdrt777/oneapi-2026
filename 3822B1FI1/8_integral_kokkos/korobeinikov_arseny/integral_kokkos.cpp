#include "integral_kokkos.h"

#include <Kokkos_MathematicalFunctions.hpp>

float IntegralKokkos(float start, float end, int count) {
    using execution_space = Kokkos::SYCL;
    using memory_space = Kokkos::SYCLDeviceUSMSpace;

    const float step = (end - start) / static_cast<float>(count);
    const float cell_area = step * step;

    Kokkos::View<float, memory_space> device_sum("device_sum");
    Kokkos::deep_copy(device_sum, 0.0f);

    Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<2>> policy(
        {0, 0}, {count, count});

    Kokkos::parallel_reduce(
        "integral_reduce",
        policy,
        KOKKOS_LAMBDA(int i, int j, float& local_value) {
            const float x = start + (static_cast<float>(i) + 0.5f) * step;
            const float y = start + (static_cast<float>(j) + 0.5f) * step;
            local_value += Kokkos::sin(x) * Kokkos::cos(y);
        },
        device_sum
    );

    execution_space().fence();

    float host_sum = 0.0f;
    Kokkos::deep_copy(host_sum, device_sum);

    return host_sum * cell_area;
}
