#include "KDTreeFlatVectorAdaptor.h"

#include <iostream>
#include <random>
#include <chrono>
#include <functional>
#include <numeric>
#include <utility>
#include <boost/timer.hpp>
#include <boost/progress.hpp>

const int SAMPLES_DIM = 8;


template <typename T, typename RNG>
inline void generateRandomPoint(T *point, size_t dim, RNG& g){
//    std::default_random_engine eng{(unsigned)std::chrono::system_clock::now().time_since_epoch().count()};
//    // std::uniform_real_distribution<float> distr(0, max_range);
//    std::uniform_int_distribution<> distr(0, int(max_range));
    for (size_t d = 0; d < dim; d++)
        point[d] = static_cast<T>(g());
}

template <typename T, typename RNG>
inline void generateRandomPoint(std::vector<T> &point, size_t dim, RNG& g){
    if (point.size() != dim) point.resize(dim);
    generateRandomPoint<T, RNG>(&point[0], dim, g);
}

template <typename T, typename RNG>
void generateRandomPointCloud(std::vector<T> &samples,
                              const size_t N,
                              const size_t dim,
                              RNG& g)
{
    std::cout << "Generating "<< N << " random points...";
    samples.resize(N * dim);
    for (size_t i = 0; i < N; i++)  generateRandomPoint<T, RNG>(&samples[i * dim], dim, g);
    std::cout << "done\n";
}


template <typename T>
void kdtree_demo(const size_t nSamples,
        const size_t dim,
        unsigned seed=(unsigned)std::chrono::system_clock::now().time_since_epoch().count())
{
    std::vector<T>  samples;
    const T max_range = 32;

    std::mt19937_64 eng{seed};
    std::uniform_int_distribution<> distr(0, int(max_range));

    auto rng =std::bind(distr, eng);
    // Generate points:
    generateRandomPointCloud<T, decltype(rng)>(samples, nSamples, dim, rng);


    unsigned nq = 10000;
    // Query point:
    std::vector<T> query_pt(dim);
    generateRandomPointCloud(query_pt, nq, dim, rng);

    // construct a kd-tree index:
    // Dimensionality set at run-time (default: L2)
    // ------------------------------------------------------------
    typedef KDTreeFlatVectorAdaptor< std::vector<T>, T >  my_kd_tree_t;

    my_kd_tree_t   mat_index(dim /*dim*/, samples, 10 /* max leaf */ );
    mat_index.index->buildIndex();

    // do a knn search
    const size_t num_results = 1;
    std::vector<size_t>   ret_indexes(num_results);
    std::vector<T> out_dists_sqr(num_results);

    std::vector<std::pair<size_t, T>> resutls(nq, {(size_t)0, T(0)});
    std::cout << "Start Query ...\n";
    boost::timer timer;
    timer.restart();
    {
        boost::progress_display progress(nq);
        for (unsigned qi = 0; qi < nq; ++qi) {
            mat_index.query(&query_pt[qi * dim], num_results, &ret_indexes[0], &out_dists_sqr[0]);
            resutls[qi].first = ret_indexes[0];
            resutls[qi].second = out_dists_sqr[0];
            ++ progress;
        }
    }
//    auto t2 = std::chrono::high_resolution_clock::now();
//    std::cout << "Query: " << std::chrono::duration_cast<std::chrono::milliseconds >(t2 - t1).count() << "\n";
    std::cout << "QUERY TIME: " << timer.elapsed() << "\n";
    std::cout << std::string(80, '-') << '\n';
    std::cout << "knnSearch(nn="<<num_results<<"): \n";
    for(unsigned qi = 0;qi < nq;++ qi)
        std::cout << "ret_index["<<qi<<"]=" << resutls[qi].first << " out_dist_sqr=" << resutls[qi].second << '\n';
    std::cout << std::string(80, '-') << "\n\n";
}

int main()
{
    kdtree_demo<int16_t >((1u << 22u) /* samples */, SAMPLES_DIM /* dim */, 20190429u);
    kdtree_demo<float>((1u << 22u) /* samples */, SAMPLES_DIM /* dim */, 20190429u);
    return 0;
}
