#pragma once
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <mpi.h>
#include "utility.hpp"

using std::vector;

// Legge i parametri di simulazione dal file dimensioni.txt
// Restituisce true se la lettura è andata a buon fine, false altrimenti
inline bool read_input_file(const char* filename,
                            size_t& N_dim,
                            vector<size_t>& arr,
                            size_t& nConfs,
                            size_t& nThreads,
                            double& Beta,size_t& seed) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Errore apertura %s: ", filename);
        perror("");
        return false;
    }

    if (fscanf(fp, "%zu", &N_dim) != 1) {
        fprintf(stderr, "Errore lettura N_dim\n");
        fclose(fp);
        return false;
    }   
    arr.resize(N_dim);

    for (size_t i = 0; i < N_dim; ++i) {
        if (fscanf(fp, "%zu", &arr[i]) != 1) {
            fprintf(stderr, "Errore lettura arr[%zu]\n", i);
            fclose(fp);
            return false;
        }
    }
    
    if (fscanf(fp, "%d", &nConfs) != 1) {
        fprintf(stderr, "Errore lettura nConfs\n");
        fclose(fp);
        return false;
    }
    
    if (fscanf(fp, "%zu", &nThreads) != 1) {
        fprintf(stderr, "Errore lettura nThreads\n");
        fclose(fp);
        return false;
    }
    
    if (fscanf(fp, "%lf", &Beta) != 1) {
        fprintf(stderr, "Errore lettura Beta\n");
        fclose(fp);
        return false;
    }
    
    if (fscanf(fp, "%zu", &seed) != 1) {
        fprintf(stderr, "Errore lettura seed\n");
        fclose(fp);
        return false;
    }
    
    fclose(fp);
    
    // Stampa ció che hai letto in caso (debug)
    printf("Rank 0 ha letto: N_dim=%zu, nConfs=%d, nThreads=%zu, Beta=%lg, seed=%zu\n", 
           N_dim, nConfs, nThreads, Beta, seed);
    printf("Dimensioni: ");
    for (size_t i = 0; i < N_dim; ++i) printf("%zu ", arr[i]);
    printf("\n");
    
    return true;
}

// Stampa il riepilogo delle prestazioni alla fine della simulazione
inline void print_performance_summary(double total, double compute, 
                                       double mpi, double io, double init,
                                       int nConfs) {
    double overhead = total - compute - mpi - io;
    printf("\n");
    printf("          PERFORMANCE PROFILING        \n");
    printf("Total runtime:             %10.3f s (100.0%%)\n", total);
    printf("Computation time:          %10.3f s (%5.1f%%)\n", compute, 100.0*compute/total);
    printf("MPI Communication:         %10.3f s (%5.1f%%)\n", mpi, 100.0*mpi/total);
    printf("I/O (file write):          %10.3f s (%5.1f%%)\n", io, 100.0*io/total);
    printf("Initialitation time:       %10.3f s (%5.1f%%)\n", init, 100.0*init/total);
    printf("Overhead:                  %10.3f s (%5.1f%%)\n", overhead, 100.0*overhead/total);
    printf("Configurations:               %d\n", nConfs);
    printf("Time per config:           %10.3f s\n", total/nConfs);
}

// Scrive una misura nel file di output
inline void write_measurement(FILE* measFile, double mag, double en, size_t N) {
    fprintf(measFile, "%lg %lg\n", mag/N, en/N);
    fflush(measFile);
}

// Stampa il progresso della simulazione
/*inline void print_progress(int iConf, int nConfs, int world_size) {
    if ((iConf + 1) % 10 == 0 || (iConf + 1) == nConfs) {
        int global_progress = 0;
        for (int r = 0; r < world_size; ++r) {
            int r_done = std::min(iConf + 1, 
                           (int)(nConfs / world_size + (r < (nConfs % world_size) ? 1 : 0)));
            global_progress += r_done;
        }

        printf("Progress: %d/%d (%.1f%%)\n", 
               global_progress, nConfs, 100.0 * global_progress / nConfs);
    }
}*/

// Stampa le informazioni sulla simulazione
inline void print_simulation_info(size_t N_dim, size_t N, size_t nThreads, int nConfs, 
                                   double Beta, size_t rng_memory, bool parallel_rng) {
    printf("N_dim: %zu, Npunti: %zu, NThreads: %zu, nConfs: %d, Beta: %lg\n", 
           N_dim, N, nThreads, nConfs, Beta);
    if (parallel_rng) {
        printf("Memory usage of the rng: %zu Bytes\n", rng_memory);
    } else {
        printf("Memory usage of the rng: %zu MB\n", rng_memory / (1 << 20));
    }
}

// Stampa la topologia MPI per debug
inline void print_mpi_topology(int world_rank, int world_size, size_t N_dim,
                                const vector<int>& rank_coords,
                                const vector<size_t>& global_offset,
                                const vector<size_t>& local_L) {
    for (int r = 0; r < world_size; ++r) {
        if (world_rank == r) {
            printf("RANK %d \n", world_rank);
            printf("  rank_coords: [");
            for (size_t d = 0; d < N_dim; ++d) {
                printf("%d", rank_coords[d]);
                if (d < N_dim-1) printf(", ");
            }
            printf("]\n");
        
            printf("  global_offset: [");
            for (size_t d = 0; d < N_dim; ++d) {
                printf("%zu", global_offset[d]);
                if (d < N_dim-1) printf(", ");
            }
            printf("]\n");

            printf("  local_L: [");
            for (size_t d = 0; d < N_dim; ++d) {
                printf("%zu", local_L[d]);
                if (d < N_dim-1) printf(", ");
            }
            printf("]\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// Stampa la configurazione locale per debug
// Ogni rank stampa in ordine la propria configurazione
inline void print_configuration_debug(const vector<int8_t>& conf_local,
                                       const vector<size_t>& local_L,
                                       const vector<size_t>& local_L_halo,
                                       size_t N_dim, size_t N_local,
                                       int world_rank, int world_size,
                                       int iConf, MPI_Comm comm) {
    for (int r = 0; r < world_size; ++r) {
        if (world_rank == r) {
            printf("RANK %d, Conf %d\n", world_rank, iConf);
            
            if (N_dim == 2) {
                // Stampa come matrice 2D
                for (size_t y = 0; y < local_L[1]; ++y) {
                    printf("  ");
                    for (size_t x = 0; x < local_L[0]; ++x) {
                        // Converti coordinate locali (senza halo) in coordinate con halo
                        size_t coord_halo[2] = {x + 1, y + 1};
                        size_t idx_halo = coord_halo[0] + coord_halo[1] * local_L_halo[0];
                        // Stampa '+' per spin +1, '-' per spin -1
                        printf("%c ", conf_local[idx_halo] > 0 ? '+' : '-');
                    }
                    printf("\n");
                }
            } else {
                // Per altre dimensioni, stampa linearmente
                vector<size_t> coord_local(N_dim);
                vector<size_t> coord_halo(N_dim);
                printf("  Spins: ");
                for (size_t i = 0; i < N_local; ++i) {
                    index_to_coord(i, N_dim, local_L.data(), coord_local.data());
                    for (size_t d = 0; d < N_dim; ++d) {
                        coord_halo[d] = coord_local[d] + 1;
                    }
                    size_t idx_halo = coord_to_index(N_dim, local_L_halo.data(), coord_halo.data());
                    printf("%+d ", (int)conf_local[idx_halo]);
                    if ((i + 1) % 16 == 0 && i + 1 < N_local) printf("\n        ");
                }
                printf("\n");
            }
            fflush(stdout);
        }
        MPI_Barrier(comm);
    }
    if (world_rank == 0) {
        printf("-----------------------------------\n");
        fflush(stdout);
    }
    MPI_Barrier(comm);
}

// Stampa la configurazione GLOBALE per debug (ricostruita da tutti i rank)
// Utile per confrontare configurazioni con diverso numero di rank
inline void print_global_configuration_debug(const vector<int8_t>& conf_local,
                                              const vector<size_t>& local_L,
                                              const vector<size_t>& local_L_halo,
                                              const vector<size_t>& global_offset,
                                              const vector<size_t>& arr,
                                              size_t N_dim, size_t N_local, size_t N_global,
                                              int world_rank, int world_size,
                                              int iConf, MPI_Comm comm) {
    // Alloca buffer per la configurazione globale su rank 0
    vector<int8_t> global_conf;
    if (world_rank == 0) {
        global_conf.resize(N_global);
    }
    
    // Ogni rank prepara i suoi dati: (global_index, spin) pairs
    vector<size_t> my_global_indices(N_local);
    vector<int8_t> my_spins(N_local);
    
    vector<size_t> coord_local(N_dim);
    vector<size_t> coord_halo(N_dim);
    vector<size_t> coord_global(N_dim);
    
    for (size_t i = 0; i < N_local; ++i) {
        // Calcola indice globale
        size_t global_idx = compute_global_index(i, local_L, global_offset, arr, N_dim,
                                                  coord_local.data(), coord_global.data());
        my_global_indices[i] = global_idx;
        
        // Leggi spin dalla posizione con halo
        index_to_coord(i, N_dim, local_L.data(), coord_local.data());
        for (size_t d = 0; d < N_dim; ++d) {
            coord_halo[d] = coord_local[d] + 1;
        }
        size_t idx_halo = coord_to_index(N_dim, local_L_halo.data(), coord_halo.data());
        my_spins[i] = conf_local[idx_halo];
    }
    
    // Raccogli i dati su rank 0
    if (world_rank == 0) {
        // Metti i miei dati
        for (size_t i = 0; i < N_local; ++i) {
            global_conf[my_global_indices[i]] = my_spins[i];
        }
        
        // Ricevi dagli altri rank
        for (int r = 1; r < world_size; ++r) {
            int recv_count;
            MPI_Recv(&recv_count, 1, MPI_INT, r, 0, comm, MPI_STATUS_IGNORE);
            
            vector<size_t> recv_indices(recv_count);
            vector<int8_t> recv_spins(recv_count);
            
            MPI_Recv(recv_indices.data(), recv_count * sizeof(size_t), MPI_BYTE, r, 1, comm, MPI_STATUS_IGNORE);
            MPI_Recv(recv_spins.data(), recv_count, MPI_INT8_T, r, 2, comm, MPI_STATUS_IGNORE);
            
            for (int i = 0; i < recv_count; ++i) {
                global_conf[recv_indices[i]] = recv_spins[i];
            }
        }
        
        // Stampa la configurazione globale
        printf("=== GLOBAL CONFIG, Conf %d ===\n", iConf);
        if (N_dim == 2) {
            for (size_t y = 0; y < arr[1]; ++y) {
                printf("  ");
                for (size_t x = 0; x < arr[0]; ++x) {
                    size_t idx = x + y * arr[0];
                    printf("%c ", global_conf[idx] > 0 ? '+' : '-');
                }
                printf("\n");
            }
        } else {
            printf("  ");
            for (size_t i = 0; i < N_global; ++i) {
                printf("%+d ", (int)global_conf[i]);
                if ((i + 1) % 16 == 0 && i + 1 < N_global) printf("\n  ");
            }
            printf("\n");
        }
        printf("==============================\n");
        fflush(stdout);
    } else {
        // Invia i miei dati a rank 0
        int send_count = (int)N_local;
        MPI_Send(&send_count, 1, MPI_INT, 0, 0, comm);
        MPI_Send(my_global_indices.data(), N_local * sizeof(size_t), MPI_BYTE, 0, 1, comm);
        MPI_Send(my_spins.data(), N_local, MPI_INT8_T, 0, 2, comm);
    }
    
    MPI_Barrier(comm);
}

