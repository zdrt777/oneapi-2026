#include "integral_oneapi.h"

#include <cmath>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    sycl::queue q(device);

    const float step = (end - start) / count;
    const float area = step * step;
    const int total = count * count;

    float* partial = sycl::malloc_shared<float>(total, q);

    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        int idx = id[0];

        int i = idx / count;
        int j = idx % count;

        float x = start + (i + 0.5f) * step;
        float y = start + (j + 0.5f) * step;

        partial[idx] = sycl::sin(x) * sycl::cos(y) * area;
        });

    q.wait();

    float result = 0.0f;
    for (int i = 0; i < total; i++) {
        result += partial[i];
    }

    sycl::free(partial, q);

    return result;
}