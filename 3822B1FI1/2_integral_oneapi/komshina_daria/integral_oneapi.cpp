#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    if (count <= 0) {
        return 0.0f;
    }

    sycl::queue queue(device);

    const float step = (end - start) / static_cast<float>(count);
    const size_t total_rectangles =
        static_cast<size_t>(count) * static_cast<size_t>(count);

    float sum = 0.0f;

    {
        sycl::buffer<float, 1> sum_buffer(&sum, sycl::range<1>(1));

        queue.submit([&](sycl::handler& handler) {
            auto reduction_sum = sycl::reduction(
                sum_buffer,
                handler,
                sycl::plus<float>()
            );

            handler.parallel_for(
                sycl::range<1>(total_rectangles),
                reduction_sum,
                [=](sycl::id<1> idx, auto& partial_sum) {
                    const size_t index = idx[0];

                    const int i = static_cast<int>(index % count);
                    const int j = static_cast<int>(index / count);

                    const float x = start + (static_cast<float>(i) + 0.5f) * step;
                    const float y = start + (static_cast<float>(j) + 0.5f) * step;

                    partial_sum += sycl::sin(x) * sycl::cos(y);
                }
            );
        });

        queue.wait_and_throw();
    }

    return sum * step * step;
}
