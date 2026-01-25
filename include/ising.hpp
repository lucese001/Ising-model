#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include "prng_engine.hpp"
#include "utility.hpp"

using std::vector;
using std::binomial_distribution;
using std::mt19937_64;
using namespace std;

// Variabili globali esterne (definite in new_ising.cpp)
extern size_t N_dim;

// computeEnSite: energia locale attorno a iSite
inline int computeEnSite(const vector<int8_t>& conf, 
                         const size_t& iSite_local,
                         const vector<size_t>& local_L,
                         const vector<size_t>& local_L_halo) {
    
    static thread_local vector<size_t> coord_site(N_dim);
    static thread_local vector<size_t> coord_halo(N_dim);
    static thread_local vector<size_t> coord_neigh(N_dim);
    
    if (coord_site.size() != N_dim) {
        coord_site.resize(N_dim);
        coord_halo.resize(N_dim);
        coord_neigh.resize(N_dim);
    }
    
    // Converti iSite_local (senza halo) in coordinate locali
    index_to_coord(iSite_local, N_dim, local_L.data(), coord_site.data());
    
    // Aggiungi offset +1 per l'halo (le celle interne iniziano da 1)
    for (size_t d = 0; d < N_dim; ++d) {
        coord_halo[d] = coord_site[d] + 1;
    }
    
    // Indice nel conf_local (con halo)
    size_t idx_center = coord_to_index(N_dim, local_L_halo.data(), coord_halo.data());
    
    int en = 0;
    for (size_t d = 0; d < N_dim; ++d) {
        // Vicino +1
        memcpy(coord_neigh.data(), coord_halo.data(), N_dim * sizeof(size_t));
        coord_neigh[d] = coord_halo[d] + 1;
        size_t idx_plus = coord_to_index(N_dim, local_L_halo.data(), coord_neigh.data());
        en -= conf[idx_plus] * conf[idx_center];
        
        // Vicino -1
        memcpy(coord_neigh.data(), coord_halo.data(), N_dim * sizeof(size_t));
        coord_neigh[d] = coord_halo[d] - 1;
        size_t idx_minus = coord_to_index(N_dim, local_L_halo.data(), coord_neigh.data());
        en -= conf[idx_minus] * conf[idx_center];
    }
    
    return en;
}
inline int computeEnSiteDebug(const vector<int8_t>& conf, 
                         const size_t& iSite_local,
                         const vector<size_t>& local_L,
                         const vector<size_t>& local_L_halo,bool condPrint) {
    
    static thread_local vector<size_t> coord_site(N_dim);
    static thread_local vector<size_t> coord_halo(N_dim);
    static thread_local vector<size_t> coord_neigh(N_dim);
    
    if (coord_site.size() != N_dim) {
        coord_site.resize(N_dim);
        coord_halo.resize(N_dim);
        coord_neigh.resize(N_dim);
    }
    
    // Converti iSite_local (senza halo) in coordinate locali
    index_to_coord(iSite_local, N_dim, local_L.data(), coord_site.data());
    if (condPrint){
            cout<<"coord[x]"<<static_cast<size_t>(coord_site.data()[0])<<endl;
            cout<<"coord[y]"<<static_cast<size_t>(coord_site.data()[1])<<endl;
   }
    
    // Aggiungi offset +1 per l'halo (le celle interne iniziano da 1)
    for (size_t d = 0; d < N_dim; ++d) {
        coord_halo[d] = coord_site[d] + 1;
        if (condPrint){
            cout<<"coord_halo["<<d<<"]"<<coord_halo[d]<<endl;
        }
    }
    if (condPrint){
        cout << "=== Configuration (with halo) ===" << endl;
        cout << "   ";
        for (size_t x = 0; x < local_L_halo[0]; ++x) {
            printf("%zu ", x);
        }
        printf("\n");
        for (size_t y = 0; y < local_L_halo[1]; ++y) {
            printf("%zu: ", y);
            for (size_t x = 0; x < local_L_halo[0]; ++x) {
                size_t idx_halo = x + y * local_L_halo[0];
                // Mark the current site with brackets
                if (x == coord_halo[0] && y == coord_halo[1]) {
                    printf("[%c]", conf[idx_halo] > 0 ? '+' : '-');
                } else {
                    printf(" %c ", conf[idx_halo] > 0 ? '+' : '-');
                }
            }
            printf("\n");
        }
        cout << "Current site: (" << coord_halo[0] << ", " << coord_halo[1] << ")" << endl;
    }



    // Indice nel conf_local (con halo)
    size_t idx_center = coord_to_index(N_dim, local_L_halo.data(), coord_halo.data());
    
    int en = 0;
    for (size_t d = 0; d < N_dim; ++d) {
        // Vicino +1
        memcpy(coord_neigh.data(), coord_halo.data(), N_dim * sizeof(size_t));
        coord_neigh[d] = coord_halo[d] + 1;
        if (condPrint){
            cout<<"coord_neigh["<<d<<"]"<<coord_neigh[d]<<endl;
        }
        size_t idx_plus = coord_to_index(N_dim, local_L_halo.data(), coord_neigh.data());
        if (condPrint){
            cout<<"idx_plus:"<<  static_cast<size_t>(idx_plus) << endl;
            cout<<"idx_center:"<<  static_cast<size_t>(idx_center) << endl;
            cout<<"conf_idx_plus:"<<  static_cast<int>(conf[idx_plus]) << endl;
            cout<<"conf_idx_center:"<<  static_cast<int>(conf[idx_center]) << endl;
        }
        en -= conf[idx_plus] * conf[idx_center];
        if (condPrint){
            cout<<"en1 "<<en<<endl;
        }
        
        // Vicino -1
        memcpy(coord_neigh.data(), coord_halo.data(), N_dim * sizeof(size_t));
        coord_neigh[d] = coord_halo[d] - 1;
        if (condPrint){
            cout<<"coord_neigh1["<<d<<"]"<<coord_neigh[d]<<endl;
        }
        size_t idx_minus = coord_to_index(N_dim, local_L_halo.data(), coord_neigh.data());
        en -= conf[idx_minus] * conf[idx_center];
        if (condPrint){
            cout<<"idx_minus:"<<  static_cast<size_t>(idx_minus) << endl;
            cout<<"idx_center:"<<  static_cast<size_t>(idx_center) << endl;
            cout<<"conf_idx_minus:"<<  static_cast<int>(conf[idx_minus]) << endl;
            cout<<"conf_idx_center:"<<  static_cast<int>(conf[idx_center]) << endl;
            cout<<"en2 "<<en<<endl;
        }  

    }
    
    return en;
}

// computeEn: energia totale (riduzione parallela)
inline int computeEn(const vector<int8_t>& conf, size_t N_local,
                     const vector<size_t>& local_L,
                     const vector<size_t>& local_L_halo) {
    long long en = 0;
#pragma omp parallel for reduction(+:en)
    for (size_t iSite = 0; iSite < N_local; ++iSite) {
        en += computeEnSite(conf, iSite, local_L, local_L_halo);
    }
    return (int)(en / 2);
}

// computeMagnetization_local: magnetizzazione locale
inline double computeMagnetization_local(const vector<int8_t>& conf, size_t N_local,
                                         const vector<size_t>& local_L,
                                         const vector<size_t>& local_L_halo) {
    long long mag = 0;
    
#pragma omp parallel reduction(+:mag)
    {
        vector<size_t> coord_local(N_dim);
        vector<size_t> coord_halo(N_dim);
        
#pragma omp for
        for (size_t iSite = 0; iSite < N_local; ++iSite) {
            index_to_coord(iSite, N_dim, local_L.data(), coord_local.data());
            for (size_t d = 0; d < N_dim; ++d) {
                coord_halo[d] = coord_local[d] + 1;
            }
            size_t idx_halo = coord_to_index(N_dim, local_L_halo.data(), coord_halo.data());
            mag += conf[idx_halo];
        }
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
