#include "jacobi_kokkos.h"

std::vector<float> JacobiKokkos(const std::vector<float>& a,
                                const std::vector<float>& b,
                                float accuracy) {
  using exec_space = Kokkos::SYCL;
  using mem_space = Kokkos::SYCLDeviceUSMSpace;

  const int n = static_cast<int>(b.size());

  Kokkos::View<float**, Kokkos::LayoutLeft, mem_space> a_view("a_view", n, n);
  Kokkos::View<float*, mem_space> b_view("b_view", n);
  Kokkos::View<float*, mem_space> inv_diag_view("inv_diag_view", n);
  Kokkos::View<float*, mem_space> x_old_view("x_old_view", n);
  Kokkos::View<float*, mem_space> x_new_view("x_new_view", n);

  auto a_host = Kokkos::create_mirror_view(a_view);
  auto b_host = Kokkos::create_mirror_view(b_view);

  for (int i = 0; i < n; ++i) {
    b_host(i) = b[i];
    for (int j = 0; j < n; ++j) {
      a_host(i, j) = a[i * n + j];
    }
  }

  Kokkos::deep_copy(a_view, a_host);
  Kokkos::deep_copy(b_view, b_host);

  Kokkos::parallel_for(
      "init_jacobi_data", Kokkos::RangePolicy<exec_space>(0, n),
      KOKKOS_LAMBDA(int i) {
        inv_diag_view(i) = 1.0f / a_view(i, i);
        x_old_view(i) = 0.0f;
      });

  bool converged = false;
  const int check_interval = 8;

  for (int iter = 0; iter < ITERATIONS && !converged; ++iter) {
    Kokkos::parallel_for(
        "jacobi_iteration", Kokkos::RangePolicy<exec_space>(0, n),
        KOKKOS_LAMBDA(int i) {
          float sigma = 0.0f;

          for (int j = 0; j < n; ++j) {
            if (j != i) {
              sigma += a_view(i, j) * x_old_view(j);
            }
          }

          x_new_view(i) = inv_diag_view(i) * (b_view(i) - sigma);
        });

    if ((iter + 1) % check_interval == 0) {
      float max_diff = 0.0f;

      Kokkos::parallel_reduce(
          "jacobi_accuracy_check", Kokkos::RangePolicy<exec_space>(0, n),
          KOKKOS_LAMBDA(int i, float& local_max) {
            const float diff = Kokkos::fabs(x_new_view(i) - x_old_view(i));
            if (diff > local_max) {
              local_max = diff;
            }
          },
          Kokkos::Max<float>(max_diff));

      if (max_diff < accuracy) {
        converged = true;
        break;
      }
    }

    Kokkos::kokkos_swap(x_old_view, x_new_view);
  }

  std::vector<float> result(n);
  auto x_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x_old_view);

  for (int i = 0; i < n; ++i) {
    result[i] = x_host(i);
  }

  return result;
}