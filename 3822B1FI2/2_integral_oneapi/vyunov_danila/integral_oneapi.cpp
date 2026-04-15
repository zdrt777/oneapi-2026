#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
  float result = 0.0f;
  const float step = (end - start) / count;

  {
    sycl::buffer<float> buf(&result, 1);
    sycl::queue q(device);

    q.submit([&](sycl::handler& cgh) {
      auto sum = sycl::reduction(buf, cgh, sycl::plus<float>());

      cgh.parallel_for(
          sycl::range<2>(count, count), sum,
          [=](sycl::id<2> id, auto& acc) {
            float x = start + (id[0] + 0.5f) * step;
            float y = start + (id[1] + 0.5f) * step;
            acc += sycl::sin(x) * sycl::cos(y);
          });
    });

    q.wait();
  }

  return result * step * step;
}
