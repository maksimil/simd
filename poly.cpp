#include "simd.h"
#include <cstddef>
#include <immintrin.h>
#include <iostream>
#include <vector>

double poly_value_naive(double *data_ptr, size_t data_size, double x) {
  double cx = 1;
  double value = 0;

  for (size_t i = 0; i < data_size; i++) {
    value += cx * data_ptr[i];
    cx *= x;
  }

  return value;
}

double poly_value_nosimd(double *data_ptr, size_t data_size, double x) {
  double cx = 1;
  double sx = x * x;
  double tx = x * x * x;
  double qx = x * x * x * x;

  double value = 0;

  for (size_t i = 0; i <= data_size - 4; i += 4) {
    value += cx * data_ptr[i];
    value += cx * x * data_ptr[i + 1];
    value += cx * sx * data_ptr[i + 2];
    value += cx * tx * data_ptr[i + 3];

    cx *= qx;
  }

  return value;
}

double poly_value_simd(double *data_ptr, size_t data_size, double x) {
  __m256d _value, _current, _cx, _qx;

  _value = _mm256_setzero_pd();
  _cx = _mm256_set_pd(x * x * x, x * x, x, 1);
  _qx = _mm256_set1_pd(x * x * x * x);

  for (size_t i = 0; i <= data_size - 4; i += 4) {
    _current = _mm256_loadu_pd(data_ptr + i);
    _value = _mm256_add_pd(_value, _mm256_mul_pd(_cx, _current));

    _cx = _mm256_mul_pd(_cx, _qx);
  }

  double value = sum256(_value);
  return value;
}

#define SIZE (1 << (2 * 13))
#define TESTS (10)

void poly_test() {
  std::vector<double> results_naive(TESTS);
  std::vector<double> results_simd(TESTS);
  std::vector<double> results_nosimd(TESTS);

  for (size_t i = 0; i < TESTS; i++) {
    std::vector<double> data(SIZE);
    for (size_t k = 0; k < SIZE; k++) {
      data[k] = random_double();
    }
    double x = random_double();

    {
      time_point start = now();
      double value = poly_value_naive(data.data(), data.size(), x);
      double time_ms = since_ms(start);
      std::cout << "NAIVE:  " << value << " " << time_ms << "ms\n";
      results_naive[i] = time_ms;
    }

    {
      time_point start = now();
      double value = poly_value_nosimd(data.data(), data.size(), x);
      double time_ms = since_ms(start);
      std::cout << "NOSIMD: " << value << " " << time_ms << "ms\n";
      results_nosimd[i] = time_ms;
    }

    {
      time_point start = now();
      double value = poly_value_simd(data.data(), data.size(), x);
      double time_ms = since_ms(start);
      std::cout << "SIMD:   " << value << " " << time_ms << "ms\n\n";
      results_simd[i] = time_ms;
    }
  }

  double avg_naive = average(results_naive);
  double avg_nosimd = average(results_nosimd);
  double avg_simd = average(results_simd);

  std::cout << "\e[32mNAIVE:  \e[0m" << avg_naive << "ms\n"
            << "\e[32mNOSIMD: \e[0m" << avg_nosimd << "ms\n"
            << "\e[32mSIMD:   \e[0m" << avg_simd << "ms ("
            << avg_simd / avg_nosimd << ")\n\n";
}
