#include <benchmark/benchmark.h>
#include "nanoflann/KDTreeFlatVectorAdaptor.h"
#include <random>


static void BM_Int16Ptr(benchmark::State& state) {
    const unsigned dim = 32;
    const unsigned n = 2;
    std::vector<int16_t > datasource(dim * n, 0);
    std::default_random_engine eng;
    std::uniform_int_distribution<> dist(-64, 64);
    for(short & i : datasource) i = dist(eng);

    typedef KDTreeFlatVectorAdaptor< std::vector<int16_t >, int16_t >  my_kd_tree_t;
    my_kd_tree_t tree(dim,datasource);

    my_kd_tree_t::metric_t l2(tree);
    for(auto _ : state)
        l2.evalMetric(&datasource[0], 1, dim);

}

static void BM_FloatPtr(benchmark::State& state) {
    const unsigned dim = 32;
    const unsigned n = 2;
    std::vector<float> datasource(dim * n, 0);
    std::default_random_engine eng;
    std::uniform_int_distribution<> dist(-64, 64);
    for(float & i : datasource) i = float(dist(eng));

    typedef KDTreeFlatVectorAdaptor< std::vector<float >, float >  my_kd_tree_t;
    my_kd_tree_t tree(dim,datasource);

    my_kd_tree_t::metric_t l2(tree);
    for(auto _ : state)
        l2.evalMetric(&datasource[0], 1, dim);

}

BENCHMARK(BM_Int16Ptr);
BENCHMARK(BM_FloatPtr);


BENCHMARK_MAIN();

