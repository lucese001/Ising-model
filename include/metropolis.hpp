#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <random>
#include <algorithm>
#include <omp.h>

#ifdef USE_PHILOX
#include "philox_rng.hpp"
#else
#include "prng_engine.hpp"
#endif

#include "utility.hpp"
#include "ising.hpp"

using namespace std;

// Variabili globali esterne (definite in main.cpp)
extern size_t N_dim;
extern size_t N;
extern double Beta;

// metropolis_update: esegue uno sweep Metropolis completo
// con aggiornamento a scacchiera sui siti specificati

#ifdef USE_PHILOX

// Philox RNG: riproducible per update Bulk-Boundary (non dipende dalla
// sequenza estratta).
inline void metropolis_update(vector<int8_t>& conf_local,
                              const vector<size_t>& sites,
                              const vector<size_t>& sites_global_indices,
                              const vector<size_t>& local_L,
                              const vector<size_t>& local_L_halo,
                              PhiloxRNG& gen,
                              int iConf,
                              size_t nThreads,
                              size_t N_local,
                              int target_parity,
                              vector<size_t>& arr )
{
    vector<size_t> coord_buf(N_dim);
    vector<size_t> coord_tmp(N_dim);

    #pragma omp parallel firstprivate(coord_buf, coord_tmp)
    {
        const size_t iThread = omp_get_thread_num();
        const size_t chunkSize = (sites.size() + nThreads - 1) / nThreads;
        const size_t beg = chunkSize * iThread;
        const size_t end = std::min(sites.size(), beg + chunkSize);

        for (size_t idx = beg; idx < end; ++idx) {
            size_t iSite = sites[idx];
            size_t global_idx = sites_global_indices[idx];

            // Converti iSite in coord per trovare halo index vicini
            index_to_coord(iSite, N_dim, local_L.data(), coord_buf.data());

            for (size_t d = 0; d < N_dim; ++d) {
                coord_tmp[d] = coord_buf[d] + 1;
            }
            size_t iSite_halo = coord_to_index(N_dim, local_L_halo.data(),
                                              coord_tmp.data());

            const int8_t oldVal = conf_local[iSite_halo];
            const int enBefore = computeEnSite(conf_local, iSite,
                                              local_L, local_L_halo);

            // Philox é un counter-based RNG. In questo modo l'ordine in cui
            // vengono aggiornati i siti non influenza i numeri estratti
            // (dipendono solo dall'indice globale e dalla configurazione)
            uint32_t rand0 = gen.get1(global_idx, iConf, 0, false);
            uint32_t rand1 = gen.get1(global_idx, iConf, 1, false);

            // Sample 0: Proposta di spin
            int8_t proposed_spin = (rand0 & 1) ? 1 : -1;
            conf_local[iSite_halo] = proposed_spin;

            const int enAfter = computeEnSite(conf_local, iSite,
                                             local_L, local_L_halo);
            const int eDiff = enAfter - enBefore;
            const double pAcc = std::min(1.0, exp(-Beta * (double)eDiff));

            // Sample 1: Probabilitá di accetazione
            const double rand_uniform = (double)rand1 / 4294967296.0;
            const int acc = (rand_uniform < pAcc) ? 1 : 0;

            if (!acc) conf_local[iSite_halo] = oldVal;
        }
    }
}

#else  // Versione originale prng_engine (dipende dalla sequenza estratta)

inline void metropolis_update(vector<int8_t>& conf_local,
                              const vector<size_t>& sites,
                              const vector<size_t>& sites_global_indices,
                              const vector<size_t>& local_L,
                              const vector<size_t>& local_L_halo,
                              prng_engine& gen,
                              int iConf,
                              size_t nThreads,
                              size_t N_local,
                              int target_parity)
{
    vector<size_t> coord_buf(N_dim);
    vector<size_t> coord_tmp(N_dim);

    #pragma omp parallel firstprivate(coord_buf, coord_tmp)
    {
        const size_t iThread = omp_get_thread_num();
        const size_t chunkSize = (sites.size() + nThreads - 1) / nThreads;
        const size_t beg = chunkSize * iThread;
        const size_t end = std::min(sites.size(), beg + chunkSize);

        for (size_t idx = beg; idx < end; ++idx) {
            size_t iSite = sites[idx];
            size_t global_idx = sites_global_indices[idx];

            // discard basato sull'indice globale
            prng_engine genView = gen;
            genView.discard(2 * 2 * (global_idx + N * iConf));

            // Converti iSite in coord per trovare halo index vicini
            index_to_coord(iSite, N_dim, local_L.data(), coord_buf.data());

            for (size_t d = 0; d < N_dim; ++d) {
                coord_tmp[d] = coord_buf[d] + 1;
            }
            size_t iSite_halo = coord_to_index(N_dim, local_L_halo.data(),
                                              coord_tmp.data());

            const int8_t oldVal = conf_local[iSite_halo];
            const int enBefore = computeEnSite(conf_local, iSite,
                                              local_L, local_L_halo);

            conf_local[iSite_halo] = (int8_t)(binomial_distribution<int>(1, 0.5)(genView) * 2 - 1);

            const int enAfter = computeEnSite(conf_local, iSite,
                                             local_L, local_L_halo);
            const int eDiff = enAfter - enBefore;
            const double pAcc = std::min(1.0, exp(-Beta * (double)eDiff));
            const int acc = binomial_distribution<int>(1, pAcc)(genView);

            if (!acc) conf_local[iSite_halo] = oldVal;
        }
    }
}

#endif 
