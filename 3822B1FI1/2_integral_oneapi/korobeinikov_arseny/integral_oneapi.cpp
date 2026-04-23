#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    const float h = (end - start) / static_cast<float>(count);
    const float cell = h * h;
    float sum = 0.0f;

    sycl::queue queue(device);

    {
        sycl::buffer<float, 1> sum_buffer(&sum, sycl::range<1>(1));

        queue.submit([&](sycl::handler& handler) {
            auto red = sycl::reduction(sum_buffer, handler, sycl::plus<float>());

            handler.parallel_for(sycl::range<2>(count, count), red,
                                 [=](sycl::item<2> item, auto& acc) {
                                     const int i = item.get_id(0);
                                     const int j = item.get_id(1);

                                     const float x = start + (static_cast<float>(i) + 0.5f) * h;
                                     const float y = start + (static_cast<float>(j) + 0.5f) * h;

                                     acc.combine(sycl::sin(x) * sycl::cos(y));
                                 });
        });

        queue.wait();
    }

    return sum * cell;
}
