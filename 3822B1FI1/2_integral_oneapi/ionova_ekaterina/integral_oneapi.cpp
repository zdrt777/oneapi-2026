#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    sycl::queue q(device);
    
    float h = (end - start) / count;
    float area_element = h * h;
    float final_sum = 0.0f;

    sycl::range<2> num_items{static_cast<size_t>(count), static_cast<size_t>(count)};

    {
        sycl::buffer<float, 1> sum_buf(&final_sum, 1);

        q.submit([&](sycl::handler& h_ndl) {
            auto red = sycl::reduction(sum_buf, h_ndl, sycl::plus<>());

            h_ndl.parallel_for(num_items, red, [=](sycl::id<2> idx, auto& sum) {
                int i = idx[0];
                int j = idx[1];

                float x = start + (static_cast<float>(i) + 0.5f) * h;
                float y = start + (static_cast<float>(j) + 0.5f) * h;

                sum.combine(sycl::sin(x) * sycl::cos(y));
            });
        }).wait();
    }

    return final_sum * area_element;
}
