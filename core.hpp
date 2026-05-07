#ifndef __MORLOC_RANDOM_HPP__
#define __MORLOC_RANDOM_HPP__

#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <tuple>
#include <string>
#include "mlccpptypes/prelude.hpp"

namespace {

std::mt19937& morloc_rng() {
    static thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}

} // anonymous namespace

void morloc_setSeed(int seed) {
    morloc_rng().seed(static_cast<unsigned>(seed));
}

double morloc_random_real() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(morloc_rng());
}

int morloc_random_int() {
    std::uniform_int_distribution<int> dist(0, 2147483647);
    return dist(morloc_rng());
}

bool morloc_random_bool() {
    std::uniform_int_distribution<int> dist(0, 1);
    return dist(morloc_rng()) == 1;
}

double morloc_randomRange_real(double lo, double hi) {
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(morloc_rng());
}

int morloc_randomRange_int(int lo, int hi) {
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(morloc_rng());
}

bool morloc_bernoulli(double p) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(morloc_rng()) < p;
}

template <typename T>
T morloc_choice(const std::vector<T>& xs) {
    std::uniform_int_distribution<size_t> dist(0, xs.size() - 1);
    return xs[dist(morloc_rng())];
}

template <typename T>
T morloc_weightedChoice(const std::vector<std::tuple<double, T>>& pairs) {
    std::vector<double> weights;
    weights.reserve(pairs.size());
    for (const auto& p : pairs) {
        weights.push_back(std::get<0>(p));
    }
    std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
    return std::get<1>(pairs[dist(morloc_rng())]);
}

template <typename T>
std::vector<T> morloc_sample(int n, const std::vector<T>& xs) {
    std::vector<T> pool(xs);
    size_t k = static_cast<size_t>(n);
    for (size_t i = 0; i < k; ++i) {
        std::uniform_int_distribution<size_t> dist(i, pool.size() - 1);
        std::swap(pool[i], pool[dist(morloc_rng())]);
    }
    pool.resize(k);
    return pool;
}

template <typename T>
std::vector<T> morloc_sampleWith(int n, const std::vector<T>& xs) {
    std::vector<T> result;
    result.reserve(static_cast<size_t>(n));
    std::uniform_int_distribution<size_t> dist(0, xs.size() - 1);
    for (int i = 0; i < n; ++i) {
        result.push_back(xs[dist(morloc_rng())]);
    }
    return result;
}

template <typename T>
std::vector<T> morloc_permute(const std::vector<T>& xs) {
    std::vector<T> result(xs);
    for (size_t i = result.size(); i > 1; --i) {
        std::uniform_int_distribution<size_t> dist(0, i - 1);
        std::swap(result[i - 1], result[dist(morloc_rng())]);
    }
    return result;
}

#endif
