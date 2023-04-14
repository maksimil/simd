#include "simd.h"
#include <immintrin.h>
#include <iostream>

double dot_naive(const double *data1_ptr, const double *data2_ptr,
                 size_t data_size) {
  double acc = 0;
  for (size_t k = 0; k < data_size; k++) {
    acc += data1_ptr[k] * data2_ptr[k];
  }
  return acc;
}

double dot_nosimd(const double *data1_ptr, const double *data2_ptr,
                  size_t data_size) {
  double acc = 0;
  for (size_t k = 0; k <= data_size - 4; k += 4) {
    acc += data1_ptr[k + 0] * data2_ptr[k + 0] +
           data1_ptr[k + 1] * data2_ptr[k + 1] +
           data1_ptr[k + 2] * data2_ptr[k + 2] +
           data1_ptr[k + 3] * data2_ptr[k + 3];
  }
  return acc;
}

double dot_simd(const double *data1_ptr, const double *data2_ptr,
                size_t data_size) {
  __m256d _acc, _data1, _data2;
  _acc = _mm256_setzero_pd();

  for (size_t k = 0; k <= data_size - 4; k += 4) {
    _data1 = _mm256_loadu_pd(data1_ptr + k);
    _data2 = _mm256_loadu_pd(data2_ptr + k);
    _acc = _mm256_add_pd(_acc, _mm256_mul_pd(_data1, _data2));
  }

  double value = sum256(_acc);
  return value;
}

#define SIZE (1 << (2 * 13))
#define TESTS (10)

void dot_test() {
  std::vector<double> results_naive(TESTS);
  std::vector<double> results_simd(TESTS);
  std::vector<double> results_nosimd(TESTS);

  for (size_t i = 0; i < TESTS; i++) {
    std::vector<double> data1(SIZE);
    std::vector<double> data2(SIZE);
    for (size_t k = 0; k < SIZE; k++) {
      data1[k] = random_double();
      data2[k] = random_double();
    }

    {
      time_point start = now();
      double value = dot_naive(data1.data(), data2.data(), SIZE);
      double time_ms = since_ms(start);
      std::cout << "NAIVE:  " << value << " " << time_ms << "ms\n";
      results_naive[i] = time_ms;
    }

    {
      std::vector<double> out(SIZE);
      time_point start = now();
      double value = dot_nosimd(data1.data(), data2.data(), SIZE);
      double time_ms = since_ms(start);
      std::cout << "NOSIMD: " << value << " " << time_ms << "ms\n";
      results_nosimd[i] = time_ms;
    }

    {
      std::vector<double> out(SIZE);
      time_point start = now();
      double value = dot_simd(data1.data(), data2.data(), SIZE);
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
