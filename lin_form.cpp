#include "simd.h"
#include <immintrin.h>
#include <iostream>
#include <vector>

void lin_form_nosimd(double *out_ptr, const double *form_ptr, size_t form_size,
                     const double *data_ptr, size_t data_size) {
  for (size_t k = 0; k <= data_size - 4; k += 4) {
    out_ptr[k + 0] = 0;
    out_ptr[k + 1] = 0;
    out_ptr[k + 2] = 0;
    out_ptr[k + 3] = 0;
    for (size_t i = 0; i < form_size; i++) {
      out_ptr[k + 0] += form_ptr[i] * data_ptr[form_size * (k + 0) + i];
      out_ptr[k + 1] += form_ptr[i] * data_ptr[form_size * (k + 1) + i];
      out_ptr[k + 2] += form_ptr[i] * data_ptr[form_size * (k + 2) + i];
      out_ptr[k + 3] += form_ptr[i] * data_ptr[form_size * (k + 3) + i];
    }
  }
}

void lin_form_simd(double *out_ptr, const double *form_ptr, size_t form_size,
                   const double *data_ptr, size_t data_size) {
  __m256d _acc, _form, _data;

  for (size_t k = 0; k <= data_size - 4; k += 4) {
    _acc = _mm256_setzero_pd();

    for (size_t i = 0; i < form_size; i++) {
      _form = _mm256_set1_pd(form_ptr[i]);
      _data = _mm256_set_pd(
          data_ptr[form_size * (k + 3) + i], data_ptr[form_size * (k + 2) + i],
          data_ptr[form_size * (k + 1) + i], data_ptr[form_size * (k + 0) + i]);
      _acc = _mm256_add_pd(_acc, _mm256_mul_pd(_form, _data));
    }

    _mm256_storeu_pd(out_ptr + k, _acc);
  }
}

#define SIZE (1 << (2 * 12))
#define FORMSIZE (10)
#define TESTS (10)

void lin_form_test() {
  std::vector<double> results_simd(TESTS);
  std::vector<double> results_nosimd(TESTS);

  for (size_t i = 0; i < TESTS; i++) {
    std::vector<double> data(FORMSIZE * SIZE);
    for (size_t k = 0; k < FORMSIZE * SIZE; k++) {
      data[k] = random_double();
    }

    std::vector<double> form(FORMSIZE);
    for (size_t k = 0; k < FORMSIZE; k++) {
      form[k] = random_double();
    }

    {
      std::vector<double> out(SIZE);
      time_point start = now();
      lin_form_nosimd(out.data(), form.data(), FORMSIZE, data.data(), SIZE);
      double time_ms = since_ms(start);
      std::cout << "NOSIMD: " << out[SIZE / 2] << " " << time_ms << "ms\n";
      results_nosimd[i] = time_ms;
    }

    {
      std::vector<double> out(SIZE);
      time_point start = now();
      lin_form_simd(out.data(), form.data(), FORMSIZE, data.data(), SIZE);
      double time_ms = since_ms(start);
      std::cout << "SIMD:   " << out[SIZE / 2] << " " << time_ms << "ms\n\n";
      results_simd[i] = time_ms;
    }
  }

  double avg_nosimd = average(results_nosimd);
  double avg_simd = average(results_simd);

  std::cout << "\e[32mNOSIMD: \e[0m" << avg_nosimd << "ms\n"
            << "\e[32mSIMD:   \e[0m" << avg_simd << "ms ("
            << avg_simd / avg_nosimd << ")\n\n";
}
