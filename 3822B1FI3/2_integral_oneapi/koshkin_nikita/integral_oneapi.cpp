#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
  float result = 0.0f;
  const float step = (end - start) / static_cast<float>(count);
  sycl::queue queue(device);

  {
    sycl::buffer<float> result_buf(&result, 1);

    queue
        .submit([&](sycl::handler& handler) {
          auto sum_reduction =
              sycl::reduction(result_buf, handler, sycl::plus<>());

          handler.parallel_for(
              sycl::range<2>(count, count), sum_reduction,
              [=](sycl::id<2> index, auto& sum) {
                const float x = start + step * (index[0] + 0.5f);
                const float y = start + step * (index[1] + 0.5f);

                sum += sycl::sin(x) * sycl::cos(y);
              });
        })
        .wait();
  }

  return result * step * step;
}