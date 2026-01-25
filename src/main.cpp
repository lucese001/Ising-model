#define PARALLEL_RNG
// #define DEBUG_PRINT
// #define DEBUG_HALO

#ifdef USE_PHILOX
#include "philox_rng.hpp"
#else
#include "prng_engine.hpp"
#endif

#include "utility.hpp"
#include "ising.hpp"
#include "metropolis.hpp"
#include "halo.hpp"
#include "io.hpp"

#include <cstdint>
#include <random>
#include <vector>
#include <cstdio>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mpi.h>

using namespace std;

// Parametri della simulazione
size_t N = 1;              // numero totale di siti
double Beta;               // inverso della temperatura
size_t nThreads;           // numero di thread OpenMP
size_t N_dim;              // numero di dimensioni
vector<size_t> arr;        // lunghezze del reticolo per dimensione
size_t nConfs;             // numero di configurazioni
size_t seed;               // seed per il generatore di numeri casuali

// Parametri temperature sweep
int sweep_mode = 0;        // 0=singola T, 1=sweep
double Beta_start, Beta_end;
int N_temp_steps;

// Definizione della variabile statica timerCost (dichiarata in utility.hpp)
timer timer::timerCost;

int main(int argc, char** argv) {
    timer totalTime, computeTime, mpiTime, ioTime, setupTime;
    totalTime.start();
    setupTime.start();
    int world_size;
    int world_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    printf("world_size  %d rank %d\n", world_size, world_rank);

    // Lettura del file di input
    if (world_rank == 0) {
        if (!read_input_file("input/dimensioni.txt", N_dim, arr, nConfs, nThreads, Beta, seed,
                             sweep_mode, Beta_start, Beta_end, N_temp_steps)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    // Broadcast dati agli altri processi (usando MPI_BYTE per size_t)
    MPI_Bcast(&N_dim, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (world_rank != 0) {
        arr.resize(N_dim);
    }
    MPI_Bcast(arr.data(), N_dim * sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nConfs, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nThreads, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&seed, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sweep_mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Beta_start, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Beta_end, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N_temp_steps, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calcolo del numero totale di siti
    for (size_t i = 0; i < N_dim; ++i) {
        N *= arr[i];
    }

    vector<int> Chunks(N_dim);
    MPI_Dims_create(world_size, N_dim, Chunks.data());

    // Controllo che arr sia divisibile per Chunks
    for (size_t d = 0; d < N_dim; ++d) {
        if (arr[d] % Chunks[d] != 0) {
            if (world_rank == 0)
                cerr << "Errore: arr[" << d << "] non divisibile per Chunks[" << d << "]\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    size_t N_local = 1;
    size_t N_alloc = 1;
    size_t N_global = 1;
    vector<size_t> local_L(N_dim);
    vector<size_t> local_L_halo(N_dim);
    for (size_t d = 0; d < N_dim; ++d) {
        local_L[d] = arr[d] / Chunks[d];
        local_L_halo[d] = local_L[d] + 2;
        N_local *= local_L[d];
        N_alloc *= local_L_halo[d];
        N_global *= arr[d];
    }

    vector<int> rank_coords(N_dim);
    MPI_Comm cart_comm;
    vector<int> periods(N_dim, 1);  // Condizioni periodiche
    MPI_Cart_create(MPI_COMM_WORLD, (int)N_dim, Chunks.data(),
                    periods.data(), 1, &cart_comm);
    MPI_Cart_coords(cart_comm, world_rank, N_dim, rank_coords.data());

    // Calcolo dell'offset globale
    vector<size_t> global_offset(N_dim);
    for (size_t d = 0; d < N_dim; ++d) {
        global_offset[d] = rank_coords[d] * local_L[d];
    }

    print_mpi_topology(world_rank, world_size, N_dim,
                       rank_coords, global_offset, local_L);

    // Vicini MPI
    std::vector<std::vector<int>> neighbors;
    halo_index(cart_comm, (int)N_dim, neighbors);
    omp_set_num_threads((int)nThreads);

    // Apertura file output
    FILE* measFile = nullptr;
    if (world_rank == 0) {
        measFile = fopen("output/meas.txt", "w");
        if (!measFile) {
            perror("Errore apertura output/meas.txt");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Calibrazione timer
    for (size_t i = 0; i < 100000; ++i) {
        timer::timerCost.start();
        timer::timerCost.stop();
    }

#ifdef USE_PHILOX
    PhiloxRNG gen((seed + 104729) * 1664525 & 0xFFFFFFFF);
    print_simulation_info(N_dim, N, nThreads, nConfs, Beta, sizeof(PhiloxRNG), true);
#else
    prng_engine gen((seed + 104729) * 1664525 & 0xFFFFFFFF);
    print_simulation_info(N_dim, N, nThreads, nConfs, Beta, sizeof(prng_engine), true);
#endif

    vector<int8_t> conf_local(N_alloc);

    // Classificazione siti in Rosso/Nero (senza distinzione bulk/boundary)
    vector<size_t> red_sites, black_sites;
    vector<size_t> red_indices, black_indices;
    classify_sites_by_parity(N_local, N_dim, local_L, global_offset, arr,
                             red_sites, red_indices, black_sites, black_indices);

    // Inizializzazione configurazione
    initialize_configuration(conf_local, N_local, N_dim, local_L, local_L_halo,
                             global_offset, arr, seed);

    // Buffer e cache per halo exchange (usando funzioni da halo.hpp)
    HaloBuffers halo_buffers;
    halo_buffers.resize(N_dim);
    vector<MPI_Request> requests;

    // Pre-calcola indici delle facce
    vector<FaceInfo> faces = build_faces(local_L, N_dim);
    vector<FaceCache> face_cache = build_face_cache(faces, local_L, local_L_halo,
                                                     global_offset, arr, N_dim);

    // Parametri per il sampling delle osservabili
    double beta_crit = 0.29;
    int step;

    setupTime.stop();

    // Scrivi header se in sweep mode
    if (sweep_mode == 1 && world_rank == 0) {
        write_sweep_header(measFile);
    }

    // Loop sulle temperature (1 iterazione se sweep_mode=0)
    int n_temps = (sweep_mode == 1) ? N_temp_steps : 1;
    double dBeta = (N_temp_steps > 1) ? (Beta_end - Beta_start) / (N_temp_steps - 1) : 0;

    for (int iTemp = 0; iTemp < n_temps; ++iTemp) {
        // Calcola e aggiorna Beta corrente (globale, usato da metropolis_update)
        if (sweep_mode == 1) {
            Beta = Beta_start + iTemp * dBeta;
        }

        if (world_rank == 0) {
            if (sweep_mode == 1) {
                printf("\n=== Temperatura %d/%d: Beta = %lg ===\n", iTemp + 1, n_temps, Beta);
            }
        }

        // Reset accumulatori per questa temperatura
        double cumul_mag = 0;
        double cumul_mag_abs = 0;
        double cumul_mag_sq = 0;
        double cumul_en = 0;
        double cumul_en_sq = 0;
        int sample = 0;
        int n_meas = 0;

        // Determina step di campionamento in base alla distanza da beta_crit
        // step più grande vicino al punto critico (maggiore autocorrelazione)
        if (std::abs(Beta - beta_crit) > 0.2) {
            step = 1;  // lontano da Tc: poca autocorrelazione
        } else {
            step = 5;  // vicino a Tc: più autocorrelazione
        }

        // Re-inizializza configurazione per ogni temperatura (per indipendenza)
        if (sweep_mode == 1) {
            initialize_configuration(conf_local, N_local, N_dim, local_L, local_L_halo,
                                     global_offset, arr, seed + iTemp);
        }

        // Loop sulle configurazioni
        for (int iConf = 0; iConf < (int)nConfs; ++iConf) {
            if (world_rank == 0 && sweep_mode == 0) {
                print_progress(iConf, nConfs);
            }

#ifdef DEBUG_PRINT
            print_global_configuration_debug(conf_local, local_L, local_L_halo, global_offset, arr,
                                              N_dim, N_local, N_global, world_rank, world_size,
                                              iConf, cart_comm);
#endif

            // Halo exchange completo
            mpiTime.start();
            start_full_halo_exchange(conf_local, local_L, local_L_halo,
                                    neighbors, cart_comm, N_dim,
                                    halo_buffers, requests, face_cache);
            finish_halo_exchange(requests);
            write_full_halo_data(conf_local, halo_buffers, N_dim, face_cache);
            mpiTime.stop();

            // Misure
            computeTime.start();
            double local_mag = computeMagnetization_local(conf_local, N_local,
                                                          local_L, local_L_halo);
            double local_en = computeEn(conf_local, N_local, local_L, local_L_halo);
            computeTime.stop();

            mpiTime.start();
            double global_mag, global_en;
            MPI_Reduce(&local_mag, &global_mag, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
            MPI_Reduce(&local_en, &global_en, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
            mpiTime.stop();

            // Prendi le misure una volta completato il thermal bath (metà configurazioni)
            if (iConf > (int)nConfs / 2 && sample >= step) {
                // Normalizza per sito
                double mag_per_site = global_mag / N;
                double en_per_site = global_en / N;
                cumul_mag += mag_per_site;
                cumul_mag_abs += std::abs(mag_per_site);
                cumul_mag_sq += mag_per_site * mag_per_site;
                cumul_en += en_per_site;
                cumul_en_sq += en_per_site * en_per_site;
                sample = 0;
                n_meas++;
            } else {
                sample++;
            }

            // Scrivi misure per ogni config solo in modo singola temperatura
            if (sweep_mode == 0 && world_rank == 0) {
                ioTime.start();
                write_measurement(measFile, global_mag, global_en, N);
                ioTime.stop();
            }

            // Aggiorna tutti i siti rossi
            computeTime.start();
            metropolis_update(conf_local, red_sites, red_indices,
                              local_L, local_L_halo, gen,
                              iConf, nThreads, N_local, 0);
            computeTime.stop();

            // Halo exchange per update nero successivo
            mpiTime.start();
            start_full_halo_exchange(conf_local, local_L, local_L_halo, neighbors,
                                     cart_comm, N_dim, halo_buffers, requests, face_cache);
            finish_halo_exchange(requests);
            write_full_halo_data(conf_local, halo_buffers, N_dim, face_cache);
            mpiTime.stop();

            // Aggiorna tutti i siti neri
            computeTime.start();
            metropolis_update(conf_local, black_sites, black_indices,
                              local_L, local_L_halo, gen,
                              iConf, nThreads, N_local, 1);
            computeTime.stop();
        }

        // Calcola medie finali per questa temperatura
        double avg_mag = cumul_mag / n_meas;
        double avg_mag_abs = cumul_mag_abs / n_meas;
        double avg_en = cumul_en / n_meas;
        double avg_mag_sq = cumul_mag_sq / n_meas;
        double avg_en_sq = cumul_en_sq / n_meas;
        double chi = N * Beta * (avg_mag_sq - avg_mag * avg_mag);
        double Cv = N * Beta * Beta * (avg_en_sq - avg_en * avg_en);

        // Scrivi risultati
        if (world_rank == 0) {
            if (sweep_mode == 1) {
                // In sweep mode, stampa ogni osservabile
                ioTime.start();
                write_sweep_measurement(measFile, Beta, avg_mag, avg_mag_abs, avg_en, chi, Cv);
                ioTime.stop();
                printf("  n_meas=%d, <m>=%.6f, <|m|>=%.6f, <e>=%.6f, chi=%.6f, Cv=%.6f\n",
                       n_meas, avg_mag, avg_mag_abs, avg_en, chi, Cv);
            } else {
                // Caso per T fissa, stampa le osservabili
                printf("\nRisultati finali (n_meas=%d):\n", n_meas);
                printf("  <m>   = %.6f\n", avg_mag);
                printf("  <|m|> = %.6f\n", avg_mag_abs);
                printf("  <e>   = %.6f\n", avg_en);
                printf("  chi   = %.6f\n", chi);
                printf("  Cv    = %.6f\n", Cv);
            }
        }
    }

    if (world_rank == 0 && measFile) {
        fclose(measFile);
    }
    totalTime.stop();

    if (world_rank == 0) {
        print_performance_summary(totalTime.get(), computeTime.get(),
                                  mpiTime.get(), ioTime.get(), setupTime.get(), nConfs);
    }
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
