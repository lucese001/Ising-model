#pragma once
#include <vector>
#include <cstddef>
#include <chrono>

using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;
using std::chrono::duration_cast;

// Creazione di un timer per misurare le prestazioni
struct timer {
    static timer timerCost;

    static auto now() { return high_resolution_clock::now(); }

    size_t tot = 0;
    size_t n = 0;
    high_resolution_clock::time_point from;

    void start() { n++; from = now(); }
    void stop() { tot += duration_cast<nanoseconds>(now()-from).count(); }

    double get(bool sub = true) {
        double res = (double) tot;
        if (sub && timerCost.n > 0)
            res -= timerCost.tot * (double)n / (double)timerCost.n;
        return res / 1e9;
    }
};

// Converte un indice lineare in coordinate multi-dimensionali
inline void index_to_coord(size_t index, size_t n_dim, const size_t *arr_ptr, size_t *coord_buf) {
    for (size_t d = 0; d < n_dim; ++d) {
        coord_buf[d] = index % arr_ptr[d];
        index /= arr_ptr[d];
    }
}

// Converte coordinate multi-dimensionali in un indice lineare
inline size_t coord_to_index(size_t n_dim, const size_t *arr_ptr, const size_t *coord_buf) {
    size_t index = 0;
    size_t mult = 1;
    for (size_t d = 0; d < n_dim; ++d) {
        index += coord_buf[d] * mult;
        mult *= arr_ptr[d];
    }
    return index;
}

// Calcola l'indice globale di un sito dato il suo indice locale
inline size_t compute_global_index(size_t iSite_local,
                                   const vector<size_t>& local_L,
                                   const vector<size_t>& global_offset,
                                   const vector<size_t>& arr,
                                   size_t N_dim,
                                   size_t* coord_local,
                                   size_t* coord_global) {
    // Converte l'indice locale in coordinate locali
    index_to_coord(iSite_local, N_dim, local_L.data(), coord_local);

    // Converte coordinate locali in coordinate globali
    for (size_t d = 0; d < N_dim; ++d) {
        coord_global[d] = coord_local[d] + global_offset[d];
    }

    // Converte coordinate globali in indice globale
    return coord_to_index(N_dim, arr.data(), coord_global);
}

// Classifica i siti in Red/Black per l'algoritmo a scacchiera
inline void classify_sites_by_parity(size_t N_local, size_t N_dim,
                                     const vector<size_t>& local_L,
                                     const vector<size_t>& global_offset,
                                     const vector<size_t>& arr,
                                     vector<size_t>& red_sites,
                                     vector<size_t>& red_indices,
                                     vector<size_t>& black_sites,
                                     vector<size_t>& black_indices) {
    vector<size_t> coord_buf(N_dim);
    vector<size_t> coord_global(N_dim);

    for (size_t iSite = 0; iSite < N_local; ++iSite) {
        size_t global_idx = compute_global_index(iSite, local_L, global_offset, arr, N_dim,
                                                  coord_buf.data(), coord_global.data());
        size_t sum_global = 0;
        for (size_t d = 0; d < N_dim; ++d) {
            sum_global += coord_global[d];
        }
        int parity = sum_global % 2;

        if (parity == 0) {
            red_sites.push_back(iSite);
            red_indices.push_back(global_idx);
        } else {
            black_sites.push_back(iSite);
            black_indices.push_back(global_idx);
        }
    }
}
