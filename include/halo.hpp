#pragma once
#include <vector>
#include <array>
#include <cstddef>
#include <cstdint>
#include <mpi.h>
#include "utility.hpp"

using namespace std;


// Cache pre-calcolata degli indici per lo scambio halo
// Per ogni dimensione, memorizza gli indici dei siti al confine e nell'halo
struct FaceCache {
    uint32_t face_size;              // Numero di siti per faccia
    vector<uint32_t> idx_minus;      // Indici del boundary negativo (per Send)
    vector<uint32_t> idx_plus;       // Indici del boundary positivo (per Send)
    vector<uint32_t> idx_halo_minus; // Indici della regione halo negativa (per Recv)
    vector<uint32_t> idx_halo_plus;  // Indici della regione halo positiva (per Recv)
};

// Costruisce la cache degli indici per lo scambio halo
// Per ogni dimensione d, calcola gli indici dei siti sulla faccia perpendicolare a d
inline vector<FaceCache> build_face_cache(const vector<size_t>& local_L,
                                          const vector<size_t>& local_L_halo,
                                          uint8_t N_dim)
{
    vector<FaceCache> cache(N_dim);
    vector<size_t> coord_full(N_dim);

    for (uint8_t d = 0; d < N_dim; ++d) {
        // Calcola dimensioni della faccia (tutte le dimensioni tranne d)
        vector<size_t> face_dims;
        vector<uint8_t> face_to_full;  // Mappa: indice nella faccia -> dimensione completa
        for (uint8_t k = 0; k < N_dim; ++k) {
            if (k != d) {
                face_dims.push_back(local_L[k]);
                face_to_full.push_back(k);
            }
        }

        // Numero totale di siti nella faccia
        uint32_t face_size = 1;
        for (size_t dim : face_dims) {
            face_size *= (uint32_t)dim;
        }
        cache[d].face_size = face_size;

        // Pre-alloca memoria
        cache[d].idx_minus.reserve(face_size);
        cache[d].idx_plus.reserve(face_size);
        cache[d].idx_halo_minus.reserve(face_size);
        cache[d].idx_halo_plus.reserve(face_size);

        // Itera su tutti i siti della faccia
        vector<size_t> coord_face(face_dims.size());
        for (uint32_t i = 0; i < face_size; ++i) {
            index_to_coord(i, face_dims.size(), face_dims.data(), coord_face.data());

            // Converti coordinate faccia -> coordinate complete (con offset +1 per halo)
            for (uint8_t j = 0; j < face_to_full.size(); ++j) {
                coord_full[face_to_full[j]] = coord_face[j] + 1;
            }

            // Boundary negativo (primo strato interno, coord_halo = 1)
            coord_full[d] = 1;
            cache[d].idx_minus.push_back(
                (uint32_t)coord_to_index(N_dim, local_L_halo.data(), coord_full.data()));

            // Halo negativo (ghost cells, coord_halo = 0)
            coord_full[d] = 0;
            cache[d].idx_halo_minus.push_back(
                (uint32_t)coord_to_index(N_dim, local_L_halo.data(), coord_full.data()));

            // Boundary positivo (ultimo strato interno, coord_halo = local_L[d])
            coord_full[d] = local_L[d];
            cache[d].idx_plus.push_back(
                (uint32_t)coord_to_index(N_dim, local_L_halo.data(), coord_full.data()));

            // Halo positivo (ghost cells, coord_halo = local_L[d] + 1)
            coord_full[d] = local_L[d] + 1;
            cache[d].idx_halo_plus.push_back(
                (uint32_t)coord_to_index(N_dim, local_L_halo.data(), coord_full.data()));
        }
    }

    return cache;
}

// Debug print per visualizzare le facce
inline void print_face_cache_debug(const vector<FaceCache>& cache, uint8_t N_dim, int rank) {
    printf("\n[Rank %d] ========== FACE CACHE DEBUG ==========\n", rank);

    for (uint8_t d = 0; d < N_dim; ++d) {
        const char* dim_name = (d == 0) ? "X" : (d == 1) ? "Y" : "Z";
        printf("\n[Rank %d] Dimension %u (%s): face_size=%u\n",
               rank, d, dim_name, cache[d].face_size);
        printf("[Rank %d]   idx_minus: %zu elements\n", rank, cache[d].idx_minus.size());
        printf("[Rank %d]   idx_plus: %zu elements\n", rank, cache[d].idx_plus.size());
        printf("[Rank %d]   idx_halo_minus: %zu elements\n", rank, cache[d].idx_halo_minus.size());
        printf("[Rank %d]   idx_halo_plus: %zu elements\n", rank, cache[d].idx_halo_plus.size());
    }

    printf("\n[Rank %d] ========== END FACE CACHE DEBUG ==========\n\n", rank);
}

// Calcola i rank dei vicini MPI usando la topologia cartesiana
inline void halo_index(MPI_Comm cart_comm, uint8_t N_dim, vector<vector<int>>& neighbors) {
    neighbors.resize(N_dim);
    for (uint8_t d = 0; d < N_dim; ++d) {
        neighbors[d].resize(2);
        MPI_Cart_shift(cart_comm, d, 1, &neighbors[d][0], &neighbors[d][1]);
    }
}


// Esegue lo scambio halo completo su tutte le dimensioni usando MPI_Sendrecv
inline void halo_exchange(
    vector<int8_t>& conf_local,
    const vector<vector<int>>& neighbors,
    MPI_Comm cart_comm, uint8_t N_dim,
    const vector<FaceCache>& cache)
{
    int my_rank;
    MPI_Comm_rank(cart_comm, &my_rank);

    for (uint8_t d = 0; d < N_dim; ++d) {
        const uint32_t face_size = cache[d].face_size;

        // Buffer temporanei per questa dimensione
        vector<int8_t> send_minus(face_size), send_plus(face_size);
        vector<int8_t> recv_minus(face_size), recv_plus(face_size);

        // Copia dati nei buffer di invio
        for (uint32_t i = 0; i < face_size; ++i) {
            send_minus[i] = conf_local[cache[d].idx_minus[i]];
            send_plus[i] = conf_local[cache[d].idx_plus[i]];
        }

        // Se sono il mio stesso vicino (1 rank in questa dimensione), copia locale
        if (neighbors[d][0] == my_rank) {
            for (uint32_t i = 0; i < face_size; ++i) {
                conf_local[cache[d].idx_halo_minus[i]] = send_plus[i];
                conf_local[cache[d].idx_halo_plus[i]] = send_minus[i];
            }
        } else {
            // Shift +1: invio plus boundary al plus neighbor, ricevo dal minus neighbor
            MPI_Sendrecv(send_plus.data(), face_size, MPI_INT8_T, neighbors[d][1], d,
                         recv_minus.data(), face_size, MPI_INT8_T, neighbors[d][0], d,
                         cart_comm, MPI_STATUS_IGNORE);

            // Shift -1: invio minus boundary al minus neighbor, ricevo dal plus neighbor
            MPI_Sendrecv(send_minus.data(), face_size, MPI_INT8_T, neighbors[d][0], d + N_dim,
                         recv_plus.data(), face_size, MPI_INT8_T, neighbors[d][1], d + N_dim,
                         cart_comm, MPI_STATUS_IGNORE);

            // Scrivi dati ricevuti nelle ghost cells
            for (uint32_t i = 0; i < face_size; ++i) {
                conf_local[cache[d].idx_halo_minus[i]] = recv_minus[i];
                conf_local[cache[d].idx_halo_plus[i]] = recv_plus[i];
            }
        }
    }
}

