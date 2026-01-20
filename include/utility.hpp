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

// Overload semplificato per calcolare la parità dalle coordinate
// Ritorna la somma delle coordinate (usata per calcolare parità)
inline size_t compute_global_index(const vector<size_t>& coord_full) {
    size_t sum = 0;
    for (size_t d = 0; d < coord_full.size(); ++d) {
        sum += coord_full[d];
    }
    return sum;
}

// Classifica i siti in bulk (interni) e boundary (al bordo)
// Popola anche i vettori con gli indici globali corrispondenti
// Classifica i siti in bulk/boundary e Red/Black
inline void classify_sites(size_t N_local, size_t N_dim,
                           const vector<size_t>& local_L,
                           const vector<size_t>& global_offset,
                           const vector<size_t>& arr,
                           vector<size_t>& bulk_red_sites,
                           vector<size_t>& bulk_red_indices,
                           vector<size_t>& bulk_black_sites,
                           vector<size_t>& bulk_black_indices,
                           vector<size_t>& boundary_red_sites,
                           vector<size_t>& boundary_red_indices,
                           vector<size_t>& boundary_black_sites,
                           vector<size_t>& boundary_black_indices) {
    
    vector<size_t> coord_buf(N_dim);
    vector<size_t> coord_global(N_dim);  // buffer per compute_global_index
    
    for (size_t iSite = 0; iSite < N_local; ++iSite) {
        index_to_coord(iSite, N_dim, local_L.data(), coord_buf.data());
        
        // Determina se il sito è al bordo
        bool is_boundary = false;
        for (size_t d = 0; d < N_dim; ++d) {
            if (coord_buf[d] == 0 || coord_buf[d] == local_L[d] - 1) {
                is_boundary = true;
                break;
            }
        }
        
        // Calcola l'indice globale e coordinate globali
        size_t global_idx = compute_global_index(iSite, local_L, global_offset, arr, N_dim,
                                                  coord_buf.data(), coord_global.data());
        
        // Calcola parità globale
        size_t sum_global = 0;
        for (size_t d = 0; d < N_dim; ++d) {
            sum_global += coord_global[d];
        }
        int parity = sum_global % 2; // 0 = Rosso, 1 = Nero
        
        // Classifica il sito
        if (!is_boundary) { // Bulk
            if (parity == 0) {
                bulk_red_sites.push_back(iSite);
                bulk_red_indices.push_back(global_idx);
            } else {
                bulk_black_sites.push_back(iSite);
                bulk_black_indices.push_back(global_idx);
            }
        } else { // Boundary
            if (parity == 0) {
                boundary_red_sites.push_back(iSite);
                boundary_red_indices.push_back(global_idx);
            } else {
                boundary_black_sites.push_back(iSite);
                boundary_black_indices.push_back(global_idx);
            }
        }
    }
}
