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

// Esegue un update Metropolis sui siti specificati usando la tabella dei vicini
// - sites: indici locali dei siti da aggiornare
// - global_indices: indici globali corrispondenti (per RNG riproducibile)
// - table: tabella pre-calcolata dei vicini

#ifdef USE_PHILOX

inline void metropolis_update(vector<int8_t>& conf_local,
                              const vector<size_t>& sites,
                              const vector<size_t>& global_indices,
                              const NeighborTable& table,
                              PhiloxRNG& gen,
                              int iConf,
                              uint8_t nThreads)
{
    const uint32_t n_sites = (uint32_t)sites.size();

    #pragma omp parallel
    {
        const uint8_t thread_id = (uint8_t)omp_get_thread_num();
        const uint32_t chunk_size = (n_sites + nThreads - 1) / nThreads;
        const uint32_t begin = chunk_size * thread_id;
        const uint32_t end = std::min(n_sites, begin + chunk_size);

        for (uint32_t idx = begin; idx < end; ++idx) {
            const uint32_t site = (uint32_t)sites[idx];
            const size_t global_idx = global_indices[idx];  // Resta size_t per RNG

            // Indice nel vettore con halo
            const uint32_t halo_idx = table.site_to_halo[site];

            const int8_t old_spin = conf_local[halo_idx];
            const int energy_before = computeEnSite(conf_local, site, table);

            // Philox RNG: numeri riproducibili basati su indice globale e configurazione
            uint32_t rand0 = gen.get1(global_idx, iConf, 0, false);
            uint32_t rand1 = gen.get1(global_idx, iConf, 1, false);

            // Proposta: spin casuale +1 o -1
            int8_t proposed_spin = (rand0 & 1) ? 1 : -1;
            conf_local[halo_idx] = proposed_spin;

            const int energy_after = computeEnSite(conf_local, site, table);
            const int delta_energy = energy_after - energy_before;
            const double accept_prob = std::min(1.0, exp(-Beta * (double)delta_energy));

            // Accetta/rifiuta in base a numero casuale uniforme
            const double rand_uniform = (double)rand1 / 4294967296.0;
            if (rand_uniform >= accept_prob) {
                conf_local[halo_idx] = old_spin;  // Rifiuta: ripristina
            }
        }
    }
}

#else  // prng_engine

inline void metropolis_update(vector<int8_t>& conf_local,
                              const vector<size_t>& sites,
                              const vector<size_t>& global_indices,
                              const NeighborTable& table,
                              prng_engine& gen,
                              int iConf,
                              uint8_t nThreads)
{
    const uint32_t n_sites = (uint32_t)sites.size();

    #pragma omp parallel
    {
        const uint8_t thread_id = (uint8_t)omp_get_thread_num();
        const uint32_t chunk_size = (n_sites + nThreads - 1) / nThreads;
        const uint32_t begin = chunk_size * thread_id;
        const uint32_t end = std::min(n_sites, begin + chunk_size);

        for (uint32_t idx = begin; idx < end; ++idx) {
            const uint32_t site = (uint32_t)sites[idx];
            const size_t global_idx = global_indices[idx];  // Resta size_t per RNG

            // Discard per rendere riproducibile basato sull'indice globale
            prng_engine gen_local = gen;
            gen_local.discard(2 * 2 * (global_idx + N * iConf));

            // Indice nel vettore con halo
            const uint32_t halo_idx = table.site_to_halo[site];

            const int8_t old_spin = conf_local[halo_idx];
            const int energy_before = computeEnSite(conf_local, site, table);

            // Proposta: spin casuale +1 o -1
            conf_local[halo_idx] = (int8_t)(binomial_distribution<int>(1, 0.5)(gen_local) * 2 - 1);

            const int energy_after = computeEnSite(conf_local, site, table);
            const int delta_energy = energy_after - energy_before;
            const double accept_prob = std::min(1.0, exp(-Beta * (double)delta_energy));

            // Accetta/rifiuta
            const int accept = binomial_distribution<int>(1, accept_prob)(gen_local);
            if (!accept) {
                conf_local[halo_idx] = old_spin;  // Rifiuta: ripristina
            }
        }
    }
}

#endif
