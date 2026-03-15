#include "shared_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    int n = b.size();
    
    sycl::queue queue(device, sycl::property::queue::in_order());
    
    float* shared_a = sycl::malloc_shared<float>(n * n, queue);
    float* shared_b = sycl::malloc_shared<float>(n, queue);
    float* shared_x = sycl::malloc_shared<float>(n, queue);
    float* shared_x_new = sycl::malloc_shared<float>(n, queue);
    float* shared_diff = sycl::malloc_shared<float>(1, queue);
    
    for (int i = 0; i < n * n; i++) {
        shared_a[i] = a[i];
    }
    
    for (int i = 0; i < n; i++) {
        shared_b[i] = b[i];
        shared_x[i] = 0.0f;
    }
    
    *shared_diff = 0.0f;
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            int row = i[0];
            float diagonal = shared_a[row * n + row];
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                if (j != row) {
                    sum += shared_a[row * n + j] * shared_x[j];
                }
            }
            shared_x_new[row] = (shared_b[row] - sum) / diagonal;
        }).wait();
        
        *shared_diff = 0.0f;
        for (int i = 0; i < n; i++) {
            float diff = std::abs(shared_x_new[i] - shared_x[i]);
            *shared_diff = std::max(*shared_diff, diff);
        }
        
        for (int i = 0; i < n; i++) {
            shared_x[i] = shared_x_new[i];
        }
        
        if (*shared_diff < accuracy) {
            break;
        }
    }

    std::vector<float> x(n);
    for (int i = 0; i < n; i++) {
        x[i] = shared_x[i];
    }

    sycl::free(shared_a, queue);
    sycl::free(shared_b, queue);
    sycl::free(shared_x, queue);
    sycl::free(shared_x_new, queue);
    sycl::free(shared_diff, queue);
    
    return x;
}
