#pragma once
#include <cstdint>
// Define R123_ASSERT before including philox.h
#ifndef R123_ASSERT
#include <cassert>
#define R123_ASSERT(x) assert(x)
#endif

#include "include/Random123/philox.h"
#include <cstdint>

class PhiloxRNG {
private:
    uint32_t base_key0_;
    uint32_t base_key1_;

    // Scrambling constants (large primes for good mixing)
    static constexpr uint64_t PRIME1 = 0x9e3779b97f4a7c15ULL;  // Golden ratio
    static constexpr uint64_t PRIME2 = 0xbf58476d1ce4e5b9ULL;  // Random prime

public:
    // Initialize with seed
    explicit PhiloxRNG(uint32_t seed) {
        base_key0_ = seed;
        base_key1_ = seed ^ 0x12345678;  // Derive second key from seed
    }

    // Get deterministic random uint32_t for (global_idx, iConf, sample_number)
    // sample_number: 0 for first random, 1 for second, etc.
    uint32_t get(uint64_t global_idx, uint32_t iConf, uint32_t sample_number) {
        // Scramble inputs to maximize statistical independence
        uint64_t scrambled = global_idx * PRIME1 + (uint64_t)iConf * PRIME2;

        // Split scrambled value into counter components
        uint32_t ctr0 = (uint32_t)(scrambled & 0xFFFFFFFF);
        uint32_t ctr1 = (uint32_t)(scrambled >> 32);
        uint32_t ctr2 = sample_number;  // Which random number for this site
        uint32_t ctr3 = 0;

        // Generate 4 random uint32_t values using Random123's Philox
        philox4x32_ctr_t ctr = {{ctr0, ctr1, ctr2, ctr3}};
        philox4x32_key_t key = {{base_key0_, base_key1_}};

        philox4x32_ctr_t result = philox4x32(ctr, key);

        // Return first value
        return result.v[0];
    }
   uint32_t get1(uint64_t global_idx, uint32_t iConf, uint32_t sample_number,bool flagPrint) {
        // Scramble inputs to maximize statistical independence
        uint64_t scrambled = global_idx * PRIME1 + (uint64_t)iConf * PRIME2;

        // Split scrambled value into counter components
        uint32_t ctr0 = (uint32_t)(scrambled & 0xFFFFFFFF);
        uint32_t ctr1 = (uint32_t)(scrambled >> 32);
        uint32_t ctr2 = sample_number;  // Which random number for this site
        uint32_t ctr3 = 0;

        // Split scrambled value into counter components
        // uint32_t ctr4 = (uint32_t)(scrambled & 0xEEEEEEEE);
        // uint32_t ctr5 = (uint32_t)(scrambled >> 16);
        // uint32_t ctr6 = sample_number;  // Which random number for this site
        // uint32_t ctr = 0;


        // Generate 4 random uint32_t values using Random123's Philox
        philox4x32_ctr_t ctr = {{ctr0, ctr1, ctr2, ctr3}};
        philox4x32_key_t key = {{base_key0_, base_key1_}};
        //philox4x32_key_t key = {{ctr4, ctr5}};
        if (flagPrint){
          std::cout <<"ctr"<<ctr<<"   key"<<key<<std::endl;
        }
        philox4x32_ctr_t result = philox4x32(ctr, key);

        // Return first value
        return result.v[0];
    }
    // Get random double in [0, 1)
    double get_double(uint64_t global_idx, uint32_t iConf, uint32_t sample_number) {
        uint32_t rand_uint = get(global_idx, iConf, sample_number);
        return (double)rand_uint / 4294967296.0;  // 2^32
    }

    // Get random spin (+1 or -1)
    int8_t get_spin(uint64_t global_idx, uint32_t iConf, uint32_t sample_number) {
        return (get(global_idx, iConf, sample_number) & 1) ? 1 : -1;
    }
};
