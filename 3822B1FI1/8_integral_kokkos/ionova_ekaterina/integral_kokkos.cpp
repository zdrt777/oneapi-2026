#include "integral_kokkos.h"

float IntegralKokkos(float start, float end, int count) {
    float h = (end - start) / count;
    float total_sum = 0.0f;
    
    Kokkos::parallel_reduce("DoubleIntegral", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {count, count}), 
        KOKKOS_LAMBDA(const int i, const int j, float& lsum) {
            float x = start + (static_cast<float>(i) + 0.5f) * h;
            float y = start + (static_cast<float>(j) + 0.5f) * h;

            lsum += std::sin(x) * std::cos(y);
        }, total_sum);

    return total_sum * (h * h);
}