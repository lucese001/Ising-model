#define PARALLEL_RNG
// #define DEBUG_PRINT

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
        if (!read_input_file("input/dimensioni.txt", N_dim, arr, nConfs, nThreads, Beta, seed)) {
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

    // Calcolo del numero totale di siti
    for (size_t i = 0; i < N_dim; ++i) {
        N *= arr[i];
    }

    vector<int> Chunks(N_dim);
    vector<size_t> local_L(N_dim);
    MPI_Dims_create(world_size, N_dim, Chunks.data());

    // Controllo che arr sia divisibile per Chunks
    for (size_t d = 0; d < N_dim; ++d) {
        if (arr[d] % Chunks[d] != 0) {
            if (world_rank == 0)
                cerr << "Errore: arr[" << d << "] non divisibile per Chunks[" << d << "]\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        local_L[d] = arr[d] / Chunks[d];
    }

    size_t N_local = 1;
    size_t N_alloc = 1;
    size_t N_global = 1;
    vector<size_t> local_L_halo(N_dim);
    for (size_t d = 0; d < N_dim; ++d) {
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
    PhiloxRNG gen(seed + 104729);
    print_simulation_info(N_dim, N, nThreads, nConfs, Beta, sizeof(PhiloxRNG), true);
#else
    prng_engine gen(seed + world_rank * 104729);
    print_simulation_info(N_dim, N, nThreads, nConfs, Beta, sizeof(prng_engine), true);
#endif

    vector<int8_t> conf_local(N_alloc);

    // Classificazione siti in Rosso/Nero (senza distinzione bulk/boundary)
    vector<size_t> red_sites, black_sites;
    vector<size_t> red_indices, black_indices;

    // Classifica TUTTI i siti per parità
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

    // Inizializzazione configurazione
    initialize_configuration(conf_local, N_local, N_dim, local_L, local_L_halo,
                             global_offset, arr, seed);

    // Buffer per halo exchange (senza parity - scambia TUTTO)
    vector<vector<int8_t>> send_minus(N_dim), send_plus(N_dim);
    vector<vector<int8_t>> recv_minus(N_dim), recv_plus(N_dim);

    // Pre-calcola indici delle facce (TUTTI i siti, non per parità)
    vector<vector<size_t>> face_minus_idx(N_dim), face_plus_idx(N_dim);
    vector<vector<size_t>> halo_minus_idx(N_dim), halo_plus_idx(N_dim);

    for (size_t d = 0; d < N_dim; ++d) {
        // Calcola dimensione faccia
        size_t face_size = 1;
        for (size_t k = 0; k < N_dim; ++k) {
            if (k != d) face_size *= local_L[k];
        }

        send_minus[d].resize(face_size);
        send_plus[d].resize(face_size);
        recv_minus[d].resize(face_size);
        recv_plus[d].resize(face_size);
        face_minus_idx[d].resize(face_size);
        face_plus_idx[d].resize(face_size);
        halo_minus_idx[d].resize(face_size);
        halo_plus_idx[d].resize(face_size);

        // Costruisci indici
        vector<size_t> coord_face(N_dim - 1);
        vector<size_t> coord_full(N_dim);
        vector<size_t> face_dims;
        for (size_t k = 0; k < N_dim; ++k) {
            if (k != d) face_dims.push_back(local_L[k]);
        }

        for (size_t i = 0; i < face_size; ++i) {
            index_to_coord(i, face_dims.size(), face_dims.data(), coord_face.data());

            // Mappa coordinate faccia a coordinate full
            size_t face_idx = 0;
            for (size_t k = 0; k < N_dim; ++k) {
                if (k < d) {
                    coord_full[k] = coord_face[face_idx++] + 1;  // +1 per halo offset
                } else if (k > d) {
                    coord_full[k] = coord_face[face_idx++] + 1;
                }
            }

            // Faccia meno (coord[d] = 1 in halo coords = 0 in local coords)
            coord_full[d] = 1;
            face_minus_idx[d][i] = coord_to_index(N_dim, local_L_halo.data(), coord_full.data());

            // Halo meno (coord[d] = 0)
            coord_full[d] = 0;
            halo_minus_idx[d][i] = coord_to_index(N_dim, local_L_halo.data(), coord_full.data());

            // Faccia più (coord[d] = local_L[d] in halo coords)
            coord_full[d] = local_L[d];
            face_plus_idx[d][i] = coord_to_index(N_dim, local_L_halo.data(), coord_full.data());

            // Halo più (coord[d] = local_L[d] + 1)
            coord_full[d] = local_L[d] + 1;
            halo_plus_idx[d][i] = coord_to_index(N_dim, local_L_halo.data(), coord_full.data());
        }
    }

    setupTime.stop();

    for (int iConf = 0; iConf < (int)nConfs; ++iConf) {
#ifdef DEBUG_PRINT
        print_global_configuration_debug(conf_local, local_L, local_L_halo, global_offset, arr,
                                          N_dim, N_local, N_global, world_rank, world_size,
                                          iConf, cart_comm);
#endif

        // ===== ALGORITMO SEMPLIFICATO =====
        // 1. Scambia TUTTO l'halo (tutti i siti delle facce)
        // 2. Aggiorna tutti i siti ROSSI
        // 3. Scambia TUTTO l'halo
        // 4. Aggiorna tutti i siti NERI

        // --- Passo 1: Halo exchange completo ---
        mpiTime.start();
        vector<MPI_Request> requests;

        for (size_t d = 0; d < N_dim; ++d) {
            size_t face_size = face_minus_idx[d].size();

            // Prepara buffer di invio
            for (size_t i = 0; i < face_size; ++i) {
                send_minus[d][i] = conf_local[face_minus_idx[d][i]];
                send_plus[d][i] = conf_local[face_plus_idx[d][i]];
            }

            int tag_minus = 100 + d;
            int tag_plus = 200 + d;
            MPI_Request req;

            // Ricevi da vicino "dietro" (sarà scritto in halo minus)
            MPI_Irecv(recv_minus[d].data(), face_size, MPI_INT8_T,
                      neighbors[d][0], tag_plus, cart_comm, &req);
            requests.push_back(req);

            // Ricevi da vicino "avanti" (sarà scritto in halo plus)
            MPI_Irecv(recv_plus[d].data(), face_size, MPI_INT8_T,
                      neighbors[d][1], tag_minus, cart_comm, &req);
            requests.push_back(req);

            // Invia a vicino "dietro"
            MPI_Isend(send_minus[d].data(), face_size, MPI_INT8_T,
                      neighbors[d][0], tag_minus, cart_comm, &req);
            requests.push_back(req);

            // Invia a vicino "avanti"
            MPI_Isend(send_plus[d].data(), face_size, MPI_INT8_T,
                      neighbors[d][1], tag_plus, cart_comm, &req);
            requests.push_back(req);
        }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

        // Scrivi dati ricevuti nell'halo
        for (size_t d = 0; d < N_dim; ++d) {
            size_t face_size = halo_minus_idx[d].size();
            for (size_t i = 0; i < face_size; ++i) {
                conf_local[halo_minus_idx[d][i]] = recv_minus[d][i];
                conf_local[halo_plus_idx[d][i]] = recv_plus[d][i];
            }
        }
        mpiTime.stop();

        // --- Passo 2: Aggiorna tutti i siti ROSSI ---
        computeTime.start();
        metropolis_update(conf_local, red_sites, red_indices,
                          local_L, local_L_halo, gen,
                          iConf, nThreads, N_local, 0);
        computeTime.stop();

        // --- Passo 3: Halo exchange completo ---
        mpiTime.start();
        requests.clear();

        for (size_t d = 0; d < N_dim; ++d) {
            size_t face_size = face_minus_idx[d].size();

            for (size_t i = 0; i < face_size; ++i) {
                send_minus[d][i] = conf_local[face_minus_idx[d][i]];
                send_plus[d][i] = conf_local[face_plus_idx[d][i]];
            }

            int tag_minus = 100 + d;
            int tag_plus = 200 + d;
            MPI_Request req;

            MPI_Irecv(recv_minus[d].data(), face_size, MPI_INT8_T,
                      neighbors[d][0], tag_plus, cart_comm, &req);
            requests.push_back(req);

            MPI_Irecv(recv_plus[d].data(), face_size, MPI_INT8_T,
                      neighbors[d][1], tag_minus, cart_comm, &req);
            requests.push_back(req);

            MPI_Isend(send_minus[d].data(), face_size, MPI_INT8_T,
                      neighbors[d][0], tag_minus, cart_comm, &req);
            requests.push_back(req);

            MPI_Isend(send_plus[d].data(), face_size, MPI_INT8_T,
                      neighbors[d][1], tag_plus, cart_comm, &req);
            requests.push_back(req);
        }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

        for (size_t d = 0; d < N_dim; ++d) {
            size_t face_size = halo_minus_idx[d].size();
            for (size_t i = 0; i < face_size; ++i) {
                conf_local[halo_minus_idx[d][i]] = recv_minus[d][i];
                conf_local[halo_plus_idx[d][i]] = recv_plus[d][i];
            }
        }
        mpiTime.stop();

        // --- Passo 4: Aggiorna tutti i siti NERI ---
        computeTime.start();
        metropolis_update(conf_local, black_sites, black_indices,
                          local_L, local_L_halo, gen,
                          iConf, nThreads, N_local, 1);
        computeTime.stop();

        // --- Passo 5: Halo exchange finale prima di misurare energia ---
        mpiTime.start();
        requests.clear();

        for (size_t d = 0; d < N_dim; ++d) {
            size_t face_size = face_minus_idx[d].size();

            for (size_t i = 0; i < face_size; ++i) {
                send_minus[d][i] = conf_local[face_minus_idx[d][i]];
                send_plus[d][i] = conf_local[face_plus_idx[d][i]];
            }

            int tag_minus = 100 + d;
            int tag_plus = 200 + d;
            MPI_Request req;

            MPI_Irecv(recv_minus[d].data(), face_size, MPI_INT8_T,
                      neighbors[d][0], tag_plus, cart_comm, &req);
            requests.push_back(req);

            MPI_Irecv(recv_plus[d].data(), face_size, MPI_INT8_T,
                      neighbors[d][1], tag_minus, cart_comm, &req);
            requests.push_back(req);

            MPI_Isend(send_minus[d].data(), face_size, MPI_INT8_T,
                      neighbors[d][0], tag_minus, cart_comm, &req);
            requests.push_back(req);

            MPI_Isend(send_plus[d].data(), face_size, MPI_INT8_T,
                      neighbors[d][1], tag_plus, cart_comm, &req);
            requests.push_back(req);
        }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

        for (size_t d = 0; d < N_dim; ++d) {
            size_t face_size = halo_minus_idx[d].size();
            for (size_t i = 0; i < face_size; ++i) {
                conf_local[halo_minus_idx[d][i]] = recv_minus[d][i];
                conf_local[halo_plus_idx[d][i]] = recv_plus[d][i];
            }
        }
        mpiTime.stop();

        // --- Misure ---
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

        if (world_rank == 0) {
            ioTime.start();
            write_measurement(measFile, global_mag, global_en, N);
            ioTime.stop();
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
