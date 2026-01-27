#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <random>
#include "prng_engine.hpp"
#include "utility.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

// Variabili globali esterne (definite in main.cpp)
extern size_t N_dim;

// Tabella pre-calcolata dei vicini per ogni sito locale
// Supporta circa 4 miliardi di siti per rank
struct NeighborTable {
    vector<uint32_t> site_to_halo;  // [N_local] -> indice halo del sito
    vector<uint32_t> neighbors;     // [N_local * 2*N_dim] -> indici halo dei vicini
    uint8_t n_neighbors_per_site;   // = 2 * N_dim (max 12 per 6D)

    inline uint32_t get_halo_index(uint32_t iSite) const {
        return site_to_halo[iSite];
    }

    inline uint32_t get_neighbor(uint32_t iSite, uint8_t n) const {
        return neighbors[iSite * n_neighbors_per_site + n];
    }
};

// Costruisce le tabelle dei vicini (chiamata una volta all'avvio)
inline bool build_neighbor_table(NeighborTable& table,
                                  uint32_t N_local,
                                  uint8_t N_dim,
                                  const vector<size_t>& local_L,
                                  const vector<size_t>& local_L_halo) {
    // Verifica che il dominio locale non sia troppo grande per uint32_t
    size_t N_alloc = 1;
    for (uint8_t d = 0; d < N_dim; ++d) {
        N_alloc *= local_L_halo[d];
    }
    if (N_alloc > UINT32_MAX) {
        fprintf(stderr, "Errore: dominio locale troppo grande per uint32_t (%zu > %u).\n"
                        "Usa piu' rank MPI o un reticolo piu' piccolo.\n",
                N_alloc, UINT32_MAX);
        return false;
    }

    table.n_neighbors_per_site = (uint8_t)(2 * N_dim);
    table.site_to_halo.resize(N_local);
    table.neighbors.resize(N_local * 2 * N_dim);

    vector<size_t> coord_local(N_dim);
    vector<size_t> coord_halo(N_dim);

    for (uint32_t iSite = 0; iSite < N_local; ++iSite) {
        // Converti indice locale -> coordinate locali
        index_to_coord(iSite, N_dim, local_L.data(), coord_local.data());

        // Coordinate halo = coordinate locali + 1 (per saltare il bordo halo)
        for (uint8_t d = 0; d < N_dim; ++d) {
            coord_halo[d] = coord_local[d] + 1;
        }

        // Salva l'indice halo del sito stesso
        table.site_to_halo[iSite] = (uint32_t)coord_to_index(N_dim, local_L_halo.data(), coord_halo.data());

        // Per ogni dimensione, calcola i due vicini (- e +)
        for (uint8_t d = 0; d < N_dim; ++d) {
            // Vicino in direzione -d (coord_halo[d] - 1)
            size_t saved = coord_halo[d];
            coord_halo[d] = saved - 1;
            table.neighbors[iSite * 2 * N_dim + 2*d + 0] =
                (uint32_t)coord_to_index(N_dim, local_L_halo.data(), coord_halo.data());

            // Vicino in direzione +d (coord_halo[d] + 1)
            coord_halo[d] = saved + 1;
            table.neighbors[iSite * 2 * N_dim + 2*d + 1] =
                (uint32_t)coord_to_index(N_dim, local_L_halo.data(), coord_halo.data());

            // Ripristina
            coord_halo[d] = saved;
        }
    }

    return true;
}

// Calcola l'energia locale di un sito (somma delle interazioni con i vicini)
inline int computeEnSite(const vector<int8_t>& conf,
                         uint32_t iSite,
                         const NeighborTable& table) {
    const uint32_t idx_center = table.site_to_halo[iSite];
    const int8_t spin_center = conf[idx_center];

    int en = 0;
    const uint8_t n_neigh = table.n_neighbors_per_site;
    const uint32_t base = iSite * n_neigh;

    for (uint8_t n = 0; n < n_neigh; ++n) {
        en -= conf[table.neighbors[base + n]] * spin_center;
    }

    return en;
}

// Calcola l'energia totale del reticolo locale (riduzione parallela)
inline int computeEnergy(const vector<int8_t>& conf, uint32_t N_local,
                         const NeighborTable& table) {
    long long en = 0;
#pragma omp parallel for reduction(+:en)
    for (uint32_t iSite = 0; iSite < N_local; ++iSite) {
        en += computeEnSite(conf, iSite, table);
    }
    return (int)(en / 2);  // Dividi per 2 perché ogni legame è contato due volte
}

// Calcola la magnetizzazione totale del reticolo locale
inline double computeMagnetization(const vector<int8_t>& conf, uint32_t N_local,
                                   const NeighborTable& table) {
    long long mag = 0;

#pragma omp parallel for reduction(+:mag)
    for (uint32_t iSite = 0; iSite < N_local; ++iSite) {
        mag += conf[table.site_to_halo[iSite]];
    }

    return (double) mag;
}

// Struttura per accumulare le osservabili durante la simulazione
struct Observables {
    double sum_mag;      // Somma delle magnetizzazioni (per <M>)
    double sum_abs_mag;  // Somma dei valori assoluti (per <|M|>)
    double sum_mag2;     // Somma dei quadrati (per chi, ovvero suscettibilitá magnetica)
    double sum_en;       // Somma delle energie (per <E>)
    double sum_en2;      // Somma dei quadrati (per C_v)
    size_t n_samples;    // Numero di campioni

    Observables() : sum_mag(0), sum_abs_mag(0), sum_mag2(0),
                               sum_en(0), sum_en2(0), n_samples(0) {}

    void reset() {
        sum_mag = sum_abs_mag = sum_mag2 = sum_en = sum_en2 = 0;
        n_samples = 0;
    }

    // Aggiunge una misurazione (magnetizzazione ed energia per sito)
    void add_measurement(double mag_per_site, double en_per_site) {
        sum_mag += mag_per_site;
        sum_abs_mag += std::abs(mag_per_site);
        sum_mag2 += mag_per_site * mag_per_site;
        sum_en += en_per_site;
        sum_en2 += en_per_site * en_per_site;
        n_samples++;
    }

    // Calcola la magnetizzazione media <M>
    double avg_magnetization() const {
        return (n_samples > 0) ? sum_mag / n_samples : 0.0;
    }

    // Calcola la magnetizzazione media in valore assoluto <|M|>
    double avg_abs_magnetization() const {
        return (n_samples > 0) ? sum_abs_mag / n_samples : 0.0;
    }

    // Calcola l'energia media <E>
    double avg_energy() const {
        return (n_samples > 0) ? sum_en / n_samples : 0.0;
    }

    // Calcola <M^2>
    double avg_mag_squared() const {
        return (n_samples > 0) ? sum_mag2 / n_samples : 0.0;
    }

    // Calcola <E^2>
    double avg_energy_squared() const {
        return (n_samples > 0) ? sum_en2 / n_samples : 0.0;
    }

    // Suscettività magnetica: chi = N * beta * (<M^2> - <M>^2)
    // Nota: M qui è per sito, quindi chi = N * beta * (<m^2> - <m>^2)
    //       dove m = M/N è la magnetizzazione per sito
    double magnetic_susceptibility(double beta, size_t N) const {
        double avg_m = avg_magnetization();
        double avg_m2 = avg_mag_squared();
        return N * beta * (avg_m2 - avg_m * avg_m);
    }

    // Capacità termica: C_v = N * beta^2 * (<E^2> - <E>^2)
    // Nota: E qui è per sito
    double heat_capacity(double beta, size_t N) const {
        double avg_e = avg_energy();
        double avg_e2 = avg_energy_squared();
        return N * beta * beta * (avg_e2 - avg_e * avg_e);
    }
};

// Crea una configurazione iniziale casuale usando l'indice globale
// (per garantire riproducibilità) indipendente dal numero di rank/thread
inline void initialize_configuration(vector<int8_t>& conf_local,
                                     size_t N_local,
                                     size_t N_dim,
                                     const vector<size_t>& local_L,
                                     const vector<size_t>& local_L_halo,
                                     const vector<size_t>& global_offset,
                                     const vector<size_t>& arr,
                                     uint64_t base_seed) {
    // Inizializza tutto a 0
    std::fill(conf_local.begin(), conf_local.end(), 0);

    #pragma omp parallel
    {
        // Ogni thread ha i suoi buffer per le coordinate
        vector<size_t> coord_local(N_dim);
        vector<size_t> coord_halo(N_dim);
        vector<size_t> coord_global(N_dim);  // buffer per compute_global_index

        #pragma omp for
        for (size_t i = 0; i < N_local; ++i) {

            size_t global_index = compute_global_index(i, local_L, global_offset, arr, N_dim,
                                                       coord_local.data(), coord_global.data());
            uint64_t site_seed = base_seed + global_index;
            prng_engine site_gen(site_seed);
            int8_t spin = (site_gen() & 1) ? 1 : -1;

            // Converti l'indice locale (senza halo) in indice con halo
            index_to_coord(i, N_dim, local_L.data(), coord_local.data());
            for (size_t d = 0; d < N_dim; ++d) {
                coord_halo[d] = coord_local[d] + 1;  // +1 per saltare l'halo
            }
            size_t idx_halo = coord_to_index(N_dim, local_L_halo.data(), coord_halo.data());

            // Memorizza lo spin
            conf_local[idx_halo] = spin;
        }
    }
}

// Calcola il modo di Fourier a k_min = 2π/L (lungo la prima dimensione)
// Usato per calcolare la lunghezza di correlazione ξ
inline void computeFourierMode(const vector<int8_t>& conf,
                               uint32_t N_local,
                               uint8_t N_dim,
                               const vector<size_t>& local_L,
                               const vector<size_t>& local_L_halo,
                               const vector<size_t>& global_offset,
                               size_t L,
                               double& Re, double& Im)
{
    const double k = 2.0 * M_PI / L;
    double local_Re = 0, local_Im = 0;

    #pragma omp parallel reduction(+:local_Re, local_Im)
    {
        vector<size_t> coord(N_dim), coord_halo(N_dim);

        #pragma omp for
        for (uint32_t iSite = 0; iSite < N_local; ++iSite) {
            index_to_coord(iSite, N_dim, local_L.data(), coord.data());
            for (uint8_t d = 0; d < N_dim; ++d) {
                coord_halo[d] = coord[d] + 1;
            }
            size_t halo_idx = coord_to_index(N_dim, local_L_halo.data(), coord_halo.data());

            size_t global_x = coord[0] + global_offset[0];
            double angle = k * global_x;
            local_Re += conf[halo_idx] * cos(angle);
            local_Im += conf[halo_idx] * sin(angle);
        }
    }

    Re = local_Re;
    Im = local_Im;
}
