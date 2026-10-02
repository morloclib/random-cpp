#ifndef __MORLOC_RANDOM_HPP__
#define __MORLOC_RANDOM_HPP__

#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <tuple>
#include <string>
#include <mutex>
#include <stdexcept>
#include "mlccpptypes/prelude.hpp"

namespace morloc_random_internal {

// One engine for the whole process, as Python's and R's random state are: a
// pool serves calls on several threads, and a seed set by one call must
// govern the next whichever thread serves it. Every draw below maps engine
// output to a value by an algorithm fixed here, not by the standard library's
// distributions, whose algorithms differ between libstdc++ and libc++: a
// seeded program prints the same numbers on Linux and macOS.
inline std::mutex& rng_mutex() {
    static std::mutex m;
    return m;
}

inline std::mt19937& rng() {
    static std::mt19937 gen(std::random_device{}());
    return gen;
}

// Uniform in [0, range), range >= 1, by rejection: unbiased.
inline uint64_t below(uint64_t range) {
    std::mt19937& g = rng();
    if (range <= (uint64_t(1) << 32)) {
        const uint64_t span = uint64_t(1) << 32;
        const uint64_t limit = span - span % range;
        uint64_t x;
        do {
            x = g();
        } while (x >= limit);
        return x % range;
    }
    // The largest multiple of `range` minus one that a 64-bit draw reaches.
    const uint64_t limit = UINT64_MAX - (UINT64_MAX % range + 1) % range;
    uint64_t x;
    do {
        x = (uint64_t(g()) << 32) | g();
    } while (x > limit);
    return x % range;
}

// Uniform in [0, 1) with 53 random bits, as CPython's random() makes them.
inline double unit() {
    std::mt19937& g = rng();
    const uint64_t a = g() >> 5, b = g() >> 6;
    return (double(a) * 67108864.0 + double(b)) * (1.0 / 9007199254740992.0);
}

} // namespace morloc_random_internal

void morloc_setSeed(int seed) {
    std::lock_guard<std::mutex> lk(morloc_random_internal::rng_mutex());
    morloc_random_internal::rng().seed(static_cast<unsigned>(seed));
}

double morloc_random_real() {
    std::lock_guard<std::mutex> lk(morloc_random_internal::rng_mutex());
    return morloc_random_internal::unit();
}

int morloc_random_int() {
    std::lock_guard<std::mutex> lk(morloc_random_internal::rng_mutex());
    return static_cast<int>(morloc_random_internal::below(uint64_t(2147483647) + 1));
}

bool morloc_random_bool() {
    std::lock_guard<std::mutex> lk(morloc_random_internal::rng_mutex());
    return morloc_random_internal::below(2) == 1;
}

double morloc_randomRange_real(double lo, double hi) {
    std::lock_guard<std::mutex> lk(morloc_random_internal::rng_mutex());
    return lo + (hi - lo) * morloc_random_internal::unit();
}

int morloc_randomRange_int(int lo, int hi) {
    if (hi < lo) throw std::invalid_argument("randomRange: the upper bound is below the lower");
    std::lock_guard<std::mutex> lk(morloc_random_internal::rng_mutex());
    const uint64_t range = uint64_t(int64_t(hi) - int64_t(lo)) + 1;
    return static_cast<int>(int64_t(lo) + int64_t(morloc_random_internal::below(range)));
}

bool morloc_bernoulli(double p) {
    std::lock_guard<std::mutex> lk(morloc_random_internal::rng_mutex());
    return morloc_random_internal::unit() < p;
}

template <typename T>
T morloc_choice(const std::vector<T>& xs) {
    if (xs.empty()) throw std::invalid_argument("choice: the list is empty");
    std::lock_guard<std::mutex> lk(morloc_random_internal::rng_mutex());
    return xs[morloc_random_internal::below(xs.size())];
}

template <typename T>
T morloc_weightedChoice(const std::vector<std::tuple<double, T>>& pairs) {
    double total = 0;
    for (const auto& p : pairs) {
        if (std::get<0>(p) < 0) throw std::invalid_argument("weightedChoice: a weight is negative");
        total += std::get<0>(p);
    }
    if (!(total > 0)) throw std::invalid_argument("weightedChoice: the weights sum to zero");
    std::lock_guard<std::mutex> lk(morloc_random_internal::rng_mutex());
    const double r = morloc_random_internal::unit() * total;
    double cum = 0;
    for (const auto& p : pairs) {
        cum += std::get<0>(p);
        if (r < cum) return std::get<1>(p);
    }
    // Rounding can leave r at the very top: the last positive weight takes it.
    for (auto it = pairs.rbegin(); it != pairs.rend(); ++it) {
        if (std::get<0>(*it) > 0) return std::get<1>(*it);
    }
    return std::get<1>(pairs.back());
}

template <typename T>
std::vector<T> morloc_sample(int n, const std::vector<T>& xs) {
    if (n < 0 || static_cast<size_t>(n) > xs.size()) {
        throw std::invalid_argument("sample: cannot draw " + std::to_string(n) + " of " +
                                    std::to_string(xs.size()) + " without replacement");
    }
    std::vector<T> pool(xs);
    const size_t k = static_cast<size_t>(n);
    std::lock_guard<std::mutex> lk(morloc_random_internal::rng_mutex());
    for (size_t i = 0; i < k; ++i) {
        std::swap(pool[i], pool[i + morloc_random_internal::below(pool.size() - i)]);
    }
    pool.resize(k);
    return pool;
}

template <typename T>
std::vector<T> morloc_sampleWith(int n, const std::vector<T>& xs) {
    if (n > 0 && xs.empty()) throw std::invalid_argument("sampleWith: the list is empty");
    std::vector<T> result;
    result.reserve(static_cast<size_t>(std::max(n, 0)));
    std::lock_guard<std::mutex> lk(morloc_random_internal::rng_mutex());
    for (int i = 0; i < n; ++i) {
        result.push_back(xs[morloc_random_internal::below(xs.size())]);
    }
    return result;
}

template <typename T>
std::vector<T> morloc_permute(const std::vector<T>& xs) {
    std::vector<T> result(xs);
    std::lock_guard<std::mutex> lk(morloc_random_internal::rng_mutex());
    for (size_t i = result.size(); i > 1; --i) {
        std::swap(result[i - 1], result[morloc_random_internal::below(i)]);
    }
    return result;
}

#endif
