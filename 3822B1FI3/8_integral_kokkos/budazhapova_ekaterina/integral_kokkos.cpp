#include "integral_kokkos.h"

float IntegralKokkos(float start, float end, int count) {
    float h = (end - start) / static_cast<float>(count);
    float h2 = h * h;
    float sum = 0.0f;

    Kokkos::parallel_reduce(
        "Integral",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {count, count}),
        KOKKOS_LAMBDA(int i, int j, float& lsum) {
            float x = start + (i + 0.5f) * h;
            float y = start + (j + 0.5f) * h;
            lsum += Kokkos::sin(x) * Kokkos::cos(y) * h2;
        },
        sum
    );

    return sum;
}