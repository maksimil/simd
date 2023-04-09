#include "simd.h"
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <vector>

time_point now() { return std::chrono::high_resolution_clock::now(); }

double since_ms(time_point start) {
  size_t mcs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now() - start)
          .count();
  return ((double)mcs) / 1000000;
}

double average(std::vector<double> times) {
  double sum = 0;
  double maxtime = times[0];
  double mintime = times[1];

  for (double time : times) {
    sum += time;

    if (time > maxtime) {
      maxtime = time;
    }

    if (time < mintime) {
      mintime = time;
    }
  }

  return (sum - maxtime - mintime) / (times.size() - 2);
}

#define RANDOM_SIZE ((1 << 25) + 1)

struct RandomArray {
  double data[RANDOM_SIZE];
  size_t current_idx;

  RandomArray() {
    std::mt19937 eng(time(NULL));
    std::uniform_real_distribution<double> unif(-1, 1);

    for (size_t i = 0; i < RANDOM_SIZE; i++) {
      data[i] = unif(eng);
    }
  }

  double next() {
    this->current_idx = (this->current_idx + 1) % RANDOM_SIZE;
    return this->data[this->current_idx];
  }

  static RandomArray &getInstance() {
    static RandomArray instance;
    return instance;
  }
};

double random_double() { return RandomArray::getInstance().next(); }

double sum256(const __m256d &_data) {
  double data[4];
  _mm256_storeu_pd(data, _data);
  return data[0] + data[1] + data[2] + data[3];
}

int main(int argc, char **argv) {
  std::cout << "\e[33m=== POLY ===\e[0m\n";
  poly_test();
  std::cout << "\e[33m=== FORM ===\e[0m\n";
  lin_form_test();
  std::cout << "\e[33m=== DOT ===\e[0m\n";
  dot_test();
}
