#include <chrono>
#include <cstddef>
#include <immintrin.h>
#include <vector>

// tests
void poly_test();
void lin_form_test();
void dot_test();

// misc
typedef std::chrono::time_point<std::chrono::system_clock,
                                std::chrono::nanoseconds>
    time_point;

time_point now();
double since_ms(time_point start);
double average(std::vector<double> times);
double random_double();
double sum256(const __m256d &_data);
