#include "integral_oneapi.h"

#include <vector>
#include <cstddef>
#include <cmath>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    if (count <= 0) {
        return 0.0f;
    }

    const std::size_t n = static_cast<std::size_t>(count);
    const float step = (end - start) / static_cast<float>(count);
    const float cell_area = step * step;

    std::vector<float> partial_sums(n, 0.0f);

    sycl::queue q(device);

    {
        sycl::buffer<float> partial_buf(partial_sums.data(), sycl::range<1>(n));

        q.submit([&](sycl::handler& h) {
            auto partial_acc =
                partial_buf.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                const std::size_t i = id[0];
                const float x =
                    start + (static_cast<float>(i) + 0.5f) * step;

                float row_sum = 0.0f;

                for (std::size_t j = 0; j < n; ++j) {
                    const float y =
                        start + (static_cast<float>(j) + 0.5f) * step;
                    row_sum += sycl::sin(x) * sycl::cos(y);
                }

                partial_acc[i] = row_sum * cell_area;
                });
            });

        q.wait();
    }

    float result = 0.0f;
    for (float value : partial_sums) {
        result += value;
    }

    return result;
}