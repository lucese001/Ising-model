#pragma once
#include <vector>
#include <array>
#include <cstddef>
#include <cstdint>
#include <mpi.h>
#include "utility.hpp"

using std::vector;

// Tipo per passare i buffer per riferimento
struct HaloBuffers {
    vector<vector<int8_t>> send_minus, send_plus; 
    vector<vector<int8_t>> recv_minus, recv_plus; 
    
    void resize(size_t N_dim) {
        send_minus.resize(N_dim);  
        send_plus.resize(N_dim);
        recv_minus.resize(N_dim);
        recv_plus.resize(N_dim);
    }
};

// Tipo per passare per riferimento le informazioni delle facce
struct FaceInfo {
    vector<size_t> dims;
    vector<size_t> map;
};

struct FaceCache {
    size_t face_size;
    //nota: ogni vettore è composto da due vettori,
    // uno per il caso pari e uno per il caso dispari.
    std::array<vector<size_t>, 2> face_to_full;
    std::array<vector<size_t>, 2> idx_minus;      // Indici del boundary negativo (per fare Send)
    std::array<vector<size_t>, 2> idx_plus;       // Indici del boundary positivo (per fare Send)
    std::array<vector<size_t>, 2> idx_halo_minus; // Indici della regione halo negativa (per fare Recv/Write)
    std::array<vector<size_t>, 2> idx_halo_plus;  // Indici della regione halo positiva (per fare Recv/Write)
};

// Costruisce le informazioni delle facce per lo scambio halo
inline vector<FaceInfo> build_faces(const vector<size_t>& local_L, size_t N_dim) {
    vector<FaceInfo> faces(N_dim);

    for (size_t d = 0; d < N_dim; ++d) {
        for (size_t k = 0; k < N_dim; ++k) {
            if (k == d) continue;
            faces[d].dims.push_back(local_L[k]);
            faces[d].map.push_back(k);
        }
    }
    return faces;
}

inline vector<FaceCache> 
build_face_cache(const vector<FaceInfo>& faces,
                 const vector<size_t>& local_L,
                 const vector<size_t>& local_L_halo,
                 size_t N_dim)
{
    vector<FaceCache> cache(N_dim);

    vector<size_t> coord_face;
    vector<size_t> coord_full(N_dim);

    for (size_t d = 0; d < N_dim; ++d) {

        const vector<size_t>& face_dims    = faces[d].dims;
        const vector<size_t>& face_to_full = faces[d].map;

        size_t face_size = 1;
        for (size_t i = 0; i < face_dims.size(); ++i)
            face_size *= face_dims[i];

        cache[d].face_size = face_size;

        for (int p = 0; p < 2; ++p) {
            cache[d].idx_minus[p].reserve(face_size/2);
            cache[d].idx_plus[p].reserve(face_size/2);
            cache[d].idx_halo_minus[p].reserve(face_size/2);
            cache[d].idx_halo_plus[p].reserve(face_size/2);
        }

        coord_face.resize(face_dims.size());

        for (size_t i = 0; i < face_size; ++i) {

            index_to_coord(i, face_dims.size(),
                           face_dims.data(), coord_face.data());

            // copia coordinate della faccia
            for (size_t j = 0; j < face_to_full.size(); ++j)
                coord_full[face_to_full[j]] = coord_face[j] + 1;

            // faccia meno (boundary interno a coord=1)
            coord_full[d] = 1;
            size_t idx_inner_minus =
                coord_to_index(N_dim, local_L_halo.data(), coord_full.data());

            size_t gidx = compute_global_index(coord_full);
            int parity = gidx & 1;

            cache[d].idx_minus[parity].push_back(idx_inner_minus);

            // halo meno (a coord=0) - calcola parity correttamente per l'halo
            coord_full[d] = 0;
            gidx = compute_global_index(coord_full);
            int parity_halo_minus = gidx & 1;
            cache[d].idx_halo_minus[parity_halo_minus].push_back(
                coord_to_index(N_dim, local_L_halo.data(), coord_full.data())
            );

            // faccia più (boundary interno a coord=local_L[d])
            coord_full[d] = local_L[d];
            size_t idx_inner_plus =
                coord_to_index(N_dim, local_L_halo.data(), coord_full.data());

            gidx = compute_global_index(coord_full);
            parity = gidx & 1;

            cache[d].idx_plus[parity].push_back(idx_inner_plus);

            // halo più (a coord=local_L[d]+1) - calcola parity correttamente per l'halo
            coord_full[d] = local_L[d] + 1;
            gidx = compute_global_index(coord_full);
            int parity_halo_plus = gidx & 1;
            cache[d].idx_halo_plus[parity_halo_plus].push_back(
                coord_to_index(N_dim, local_L_halo.data(), coord_full.data())
            );
        }
    }

    return cache;
}

// Debug print per visualizzare le facce con configurazione, indice globale e coordinate globali
inline void print_face_cache_debug(
    const vector<FaceCache>& cache,
    const vector<int8_t>& conf_local,
    const vector<size_t>& local_L,
    const vector<size_t>& local_L_halo,
    const vector<size_t>& global_offset,
    const vector<size_t>& arr,
    size_t N_dim,
    int rank)
{
    vector<size_t> coord_halo(N_dim);
    vector<size_t> coord_global(N_dim);

    printf("\n[Rank %d] ========== FACE CACHE DEBUG ==========\n", rank);

    for (size_t d = 0; d < N_dim; ++d) {
        const char* dim_name = (d == 0) ? "X" : (d == 1) ? "Y" : "Z";

        printf("\n[Rank %d] --- Dimension %zu (%s) ---\n", rank, d, dim_name);

        for (int parity = 0; parity < 2; ++parity) {
            const char* parity_name = (parity == 0) ? "RED" : "BLACK";

            // FACE MINUS (boundary interno a coord=1)
            printf("\n[Rank %d] FACE MINUS (dim %s, parity %s=%d):\n",
                   rank, dim_name, parity_name, parity);
            printf("[Rank %d]   %5s %6s %8s %15s %10s\n",
                   rank, "i", "idx", "spin", "global_coord", "global_idx");

            for (size_t i = 0; i < cache[d].idx_minus[parity].size(); ++i) {
                size_t idx = cache[d].idx_minus[parity][i];
                int8_t spin = conf_local[idx];

                // Converti idx (in coordinate halo) in coordinate halo
                index_to_coord(idx, N_dim, local_L_halo.data(), coord_halo.data());

                // Converti coordinate halo in coordinate globali
                // coord_halo ha +1 offset, quindi coord_local = coord_halo - 1
                // coord_global = coord_local + global_offset
                for (size_t k = 0; k < N_dim; ++k) {
                    coord_global[k] = (coord_halo[k] - 1) + global_offset[k];
                }
                // Applica PBC
                for (size_t k = 0; k < N_dim; ++k) {
                    coord_global[k] = coord_global[k] % arr[k];
                }

                size_t global_idx = coord_to_index(N_dim, arr.data(), coord_global.data());

                printf("[Rank %d]   %5zu %6zu %8c (", rank, i, idx, spin > 0 ? '+' : '-');
                for (size_t k = 0; k < N_dim; ++k) {
                    printf("%zu", coord_global[k]);
                    if (k < N_dim - 1) printf(",");
                }
                printf(") %10zu\n", global_idx);
            }

            // HALO MINUS (a coord=0, riceve dal vicino)
            printf("\n[Rank %d] HALO MINUS (dim %s, parity %s=%d) [destination for received data]:\n",
                   rank, dim_name, parity_name, parity);
            printf("[Rank %d]   %5s %6s %8s %15s %10s\n",
                   rank, "i", "idx", "spin", "global_coord", "global_idx");

            for (size_t i = 0; i < cache[d].idx_halo_minus[parity].size(); ++i) {
                size_t idx = cache[d].idx_halo_minus[parity][i];
                int8_t spin = conf_local[idx];

                index_to_coord(idx, N_dim, local_L_halo.data(), coord_halo.data());

                // Per l'halo minus, coord_halo[d]=0, che corrisponde a coord_local[d]=-1
                // In coordinate globali con PBC: global[d] = (global_offset[d] - 1 + arr[d]) % arr[d]
                for (size_t k = 0; k < N_dim; ++k) {
                    if (k == d) {
                        // Halo minus: viene dal vicino "dietro"
                        coord_global[k] = (global_offset[k] + arr[k] - 1) % arr[k];
                    } else {
                        coord_global[k] = (coord_halo[k] - 1) + global_offset[k];
                        coord_global[k] = coord_global[k] % arr[k];
                    }
                }

                size_t global_idx = coord_to_index(N_dim, arr.data(), coord_global.data());

                printf("[Rank %d]   %5zu %6zu %8c (", rank, i, idx, spin > 0 ? '+' : '-');
                for (size_t k = 0; k < N_dim; ++k) {
                    printf("%zu", coord_global[k]);
                    if (k < N_dim - 1) printf(",");
                }
                printf(") %10zu\n", global_idx);
            }

            // FACE PLUS (boundary interno a coord=local_L[d])
            printf("\n[Rank %d] FACE PLUS (dim %s, parity %s=%d):\n",
                   rank, dim_name, parity_name, parity);
            printf("[Rank %d]   %5s %6s %8s %15s %10s\n",
                   rank, "i", "idx", "spin", "global_coord", "global_idx");

            for (size_t i = 0; i < cache[d].idx_plus[parity].size(); ++i) {
                size_t idx = cache[d].idx_plus[parity][i];
                int8_t spin = conf_local[idx];

                index_to_coord(idx, N_dim, local_L_halo.data(), coord_halo.data());

                for (size_t k = 0; k < N_dim; ++k) {
                    coord_global[k] = (coord_halo[k] - 1) + global_offset[k];
                    coord_global[k] = coord_global[k] % arr[k];
                }

                size_t global_idx = coord_to_index(N_dim, arr.data(), coord_global.data());

                printf("[Rank %d]   %5zu %6zu %8c (", rank, i, idx, spin > 0 ? '+' : '-');
                for (size_t k = 0; k < N_dim; ++k) {
                    printf("%zu", coord_global[k]);
                    if (k < N_dim - 1) printf(",");
                }
                printf(") %10zu\n", global_idx);
            }

            // HALO PLUS (a coord=local_L[d]+1, riceve dal vicino)
            printf("\n[Rank %d] HALO PLUS (dim %s, parity %s=%d) [destination for received data]:\n",
                   rank, dim_name, parity_name, parity);
            printf("[Rank %d]   %5s %6s %8s %15s %10s\n",
                   rank, "i", "idx", "spin", "global_coord", "global_idx");

            for (size_t i = 0; i < cache[d].idx_halo_plus[parity].size(); ++i) {
                size_t idx = cache[d].idx_halo_plus[parity][i];
                int8_t spin = conf_local[idx];

                index_to_coord(idx, N_dim, local_L_halo.data(), coord_halo.data());

                // Per l'halo plus, coord_halo[d]=local_L[d]+1, che corrisponde a coord_local[d]=local_L[d]
                // In coordinate globali con PBC: global[d] = (global_offset[d] + local_L[d]) % arr[d]
                for (size_t k = 0; k < N_dim; ++k) {
                    if (k == d) {
                        // Halo plus: viene dal vicino "davanti"
                        coord_global[k] = (global_offset[k] + local_L[k]) % arr[k];
                    } else {
                        coord_global[k] = (coord_halo[k] - 1) + global_offset[k];
                        coord_global[k] = coord_global[k] % arr[k];
                    }
                }

                size_t global_idx = coord_to_index(N_dim, arr.data(), coord_global.data());

                printf("[Rank %d]   %5zu %6zu %8c (", rank, i, idx, spin > 0 ? '+' : '-');
                for (size_t k = 0; k < N_dim; ++k) {
                    printf("%zu", coord_global[k]);
                    if (k < N_dim - 1) printf(",");
                }
                printf(") %10zu\n", global_idx);
            }
        }
    }
    printf("\n[Rank %d] ========== END FACE CACHE DEBUG ==========\n\n", rank);
}

// Inizia lo scambio halo non-blocking
inline void start_halo_exchange(
    vector<int8_t>& conf_local,
    const vector<size_t>& local_L,
    const vector<size_t>& local_L_halo,
    const vector<vector<int>>& neighbors,
    MPI_Comm cart_comm,
    size_t N_dim,
    HaloBuffers& buffers,
    const vector<FaceInfo>& faces,
    vector<MPI_Request>& requests,
    const vector<FaceCache>& cache,
    int parity,
    bool debug_print)
{
    requests.clear();

    buffers.send_minus.resize(N_dim);
    buffers.send_plus.resize(N_dim);
    buffers.recv_minus.resize(N_dim);
    buffers.recv_plus.resize(N_dim);

    int rank;
    if (debug_print) {
        MPI_Comm_rank(cart_comm, &rank);
    }

    for (size_t d = 0; d < N_dim; ++d) {

        // Calcola la dimensione della faccia (tiene conto della parità)
        const size_t face_size = cache[d].idx_minus[parity].size();

        // Alloca i buffer
        buffers.send_minus[d].resize(face_size);
        buffers.send_plus[d].resize(face_size);
        buffers.recv_minus[d].resize(face_size);
        buffers.recv_plus[d].resize(face_size);

        // Prepara i buffer con le configurazioni
        // (solo siti della parità richiesta)
        for (size_t i = 0; i < face_size; ++i) {
            buffers.send_minus[d][i] = conf_local[cache[d].idx_minus[parity][i]];
            buffers.send_plus[d][i] = conf_local[cache[d].idx_plus[parity][i]];
        }

        // DEBUG: Print data facce
        if (debug_print) {
            const char* dim_name = (d == 0) ? "X" : (d == 1) ? "Y" : "Z";

            printf("[Rank %d] === SENDING dim=%zu (%s), parity=%d ===\n", rank, d, dim_name, parity);

            // Print faccia meno
            printf("[Rank %d] FACE MINUS (dim %s, coord=%zu=1) -> neighbor %d:\n",
                   rank, dim_name, d, neighbors[d][0]);
            printf("[Rank %d]   indices: ", rank);
            for (size_t i = 0; i < face_size && i < 10; ++i) {
                printf("%zu ", cache[d].idx_minus[parity][i]);
            }
            if (face_size > 10) printf("...");
            printf("\n");
            printf("[Rank %d]   values:  ", rank);
            for (size_t i = 0; i < face_size && i < 10; ++i) {
                printf("%c ", buffers.send_minus[d][i] > 0 ? '+' : '-');
            }
            if (face_size > 10) printf("...");
            printf("\n");

            // Print faccia piú
            printf("[Rank %d] FACE PLUS (dim %s, coord=%zu=%zu) -> neighbor %d:\n",
                   rank, dim_name, d, local_L[d], neighbors[d][1]);
            printf("[Rank %d]   indices: ", rank);
            for (size_t i = 0; i < face_size && i < 10; ++i) {
                printf("%zu ", cache[d].idx_plus[parity][i]);
            }
            if (face_size > 10) printf("...");
            printf("\n");
            printf("[Rank %d]   values:  ", rank);
            for (size_t i = 0; i < face_size && i < 10; ++i) {
                printf("%c ", buffers.send_plus[d][i] > 0 ? '+' : '-');
            }
            if (face_size > 10) printf("...");
            printf("\n");
        }

        int tag_minus = 100 + d;
        int tag_plus  = 200 + d;

        MPI_Request req;

        // Ricevi da vicino "dietro"
        MPI_Irecv(buffers.recv_minus[d].data(),
                  face_size, MPI_INT8_T,
                  neighbors[d][0], tag_plus,
                  cart_comm, &req);
        requests.push_back(req);

        // Ricevi da vicino "davanti"
        MPI_Irecv(buffers.recv_plus[d].data(),
                  face_size, MPI_INT8_T,
                  neighbors[d][1], tag_minus,
                  cart_comm, &req);
        requests.push_back(req);

        // Invia a vicino "dietro"
        MPI_Isend(buffers.send_minus[d].data(),
                  face_size, MPI_INT8_T,
                  neighbors[d][0], tag_minus,
                  cart_comm, &req);
        requests.push_back(req);

        // Invia a vicino "davanti"
        MPI_Isend(buffers.send_plus[d].data(),
                  face_size, MPI_INT8_T,
                  neighbors[d][1], tag_plus,
                  cart_comm, &req);
        requests.push_back(req);
    }
}


// Aspetta il completamento dello scambio halo
inline void finish_halo_exchange(vector<MPI_Request>& reqs) {
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    reqs.clear();
}

// Scrive i dati ricevuti nelle regioni halo
inline void write_halo_data(
    vector<int8_t>& conf_local,
    const HaloBuffers& buffers,
    const vector<FaceInfo>& faces,
    const vector<size_t>& local_L,
    const vector<size_t>& local_L_halo,
    size_t N_dim,
    const vector<FaceCache>& cache,
    int parity,
    MPI_Comm cart_comm = MPI_COMM_WORLD,
    bool debug_print = false)
{
    int rank;
    if (debug_print) {
        MPI_Comm_rank(cart_comm, &rank);
    }

    for (size_t d = 0; d < N_dim; ++d) {

        // Determina la dimensione della faccia (considerando la parità)
        const size_t face_size = cache[d].idx_halo_minus[parity].size();

        // DEBUG: Print received data before writing
        if (debug_print) {
            const char* dim_name = (d == 0) ? "X" : (d == 1) ? "Y" : "Z";

            printf("[Rank %d] === RECEIVED dim=%zu (%s), parity=%d ===\n", rank, d, dim_name, parity);

            // Print MINUS halo (received from neighbor behind)
            printf("[Rank %d] HALO MINUS (dim %s, coord=%zu=0):\n", rank, dim_name, d);
            printf("[Rank %d]   halo indices: ", rank);
            for (size_t i = 0; i < face_size && i < 10; ++i) {
                printf("%zu ", cache[d].idx_halo_minus[parity][i]);
            }
            if (face_size > 10) printf("...");
            printf("\n");
            printf("[Rank %d]   recv values:  ", rank);
            for (size_t i = 0; i < face_size && i < 10; ++i) {
                printf("%c ", buffers.recv_minus[d][i] > 0 ? '+' : '-');
            }
            if (face_size > 10) printf("...");
            printf("\n");

            // Print PLUS halo (received from neighbor ahead)
            printf("[Rank %d] HALO PLUS (dim %s, coord=%zu=%zu):\n", rank, dim_name, d, local_L[d] + 1);
            printf("[Rank %d]   halo indices: ", rank);
            for (size_t i = 0; i < face_size && i < 10; ++i) {
                printf("%zu ", cache[d].idx_halo_plus[parity][i]);
            }
            if (face_size > 10) printf("...");
            printf("\n");
            printf("[Rank %d]   recv values:  ", rank);
            for (size_t i = 0; i < face_size && i < 10; ++i) {
                printf("%c ", buffers.recv_plus[d][i] > 0 ? '+' : '-');
            }
            if (face_size > 10) printf("...");
            printf("\n");
        }

        for (size_t i = 0; i < face_size; ++i) {

            // Scrive i dati ricevuti negli halo meno
            conf_local[cache[d].idx_halo_minus[parity][i]] = buffers.recv_minus[d][i];

            // Scrive i dati ricevuti negli halo più
            conf_local[cache[d].idx_halo_plus[parity][i]] = buffers.recv_plus[d][i];
        }
    }
}

// Calcola gli indici dei vicini usando la topologia cartesiana MPI
inline void halo_index(MPI_Comm cart_comm, int N_dim,
                      vector<vector<int>>& neighbors) {
    neighbors.resize(N_dim);
    for (int d = 0; d < N_dim; ++d) {
        neighbors[d].resize(2);
        MPI_Cart_shift(cart_comm, d, 1, &neighbors[d][0], &neighbors[d][1]);
    }
}

