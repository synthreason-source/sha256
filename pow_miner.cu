// ═ QUANTUM PoW MINER — FULL CUDA GPU (GH/s SPEED) ═══════════════════════════════
// nvcc -O3 -arch=sm_75 pow_miner.cu -o pow_miner -lssl -lcrypto
// ./pow_miner [bits=25] [diff=20] [grid=4096] [block=256] [header="..."]
// ════════════════════════════════════════════════════════════════════════════════

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <cuda_runtime.h>
#include <openssl/sha.h>

// ── HOST GLOBALS ────────────────────────────────────────────────────────────────
uint64_t N_BITS, DIFF_BITS, GRID_SIZE, BLOCK_SIZE;
uint64_t N;
std::string BLOCK_HEADER = "First quantum sha256 by George W 28-4-2026";

std::mutex print_mutex;
std::atomic<uint64_t> total_found{0};

// ── DEVICE CONSTANTS ────────────────────────────────────────────────────────────
__constant__ char   header_const[128];
__constant__ uint64_t target_zeros;
__constant__ uint64_t c_N_BITS;
__constant__ uint64_t c_N;

// ── UTILITY ─────────────────────────────────────────────────────────────────────
std::string hash_to_hex(const unsigned char* hash) {
    std::ostringstream oss;
    for (int i = 0; i < 32; i++)
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    return oss.str();
}

// ── DEVICE SHA-256 ──────────────────────────────────────────────────────────────
__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ void sha256_transform(uint32_t state[8], const uint8_t data[64]) {
    const uint32_t K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };

    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
        w[i] = ((uint32_t)data[i*4] << 24) | ((uint32_t)data[i*4+1] << 16)
             | ((uint32_t)data[i*4+2] << 8) | (uint32_t)data[i*4+3];

    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19)  ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32_t a=state[0], b=state[1], c=state[2], d=state[3];
    uint32_t e=state[4], f=state[5], g=state[6], h=state[7];

    for (int i = 0; i < 64; ++i) {
        uint32_t S1    = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
        uint32_t ch    = (e & f) ^ (~e & g);
        uint32_t temp1 = h + S1 + ch + K[i] + w[i];
        uint32_t S0    = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
        uint32_t maj   = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;
        h=g; g=f; f=e; e=d+temp1;
        d=c; c=b; b=a; a=temp1+temp2;
    }

    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

// ── GPU SHA-256 with correct padding (supports up to 119-byte messages) ─────────
__device__ void sha256_gpu(uint64_t nonce, uint8_t* hash_out) {
    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    // Build nonce decimal string (right-aligned in tmp buf)
    char nonce_buf[21];
    int  nonce_pos = 20;
    nonce_buf[20] = '\0';
    uint64_t tmp = nonce;
    do {
        nonce_buf[--nonce_pos] = '0' + (int)(tmp % 10);
        tmp /= 10;
    } while (tmp);
    const char* nonce_str = nonce_buf + nonce_pos;
    int nonce_len = 20 - nonce_pos;

    // Assemble message: header_const + "|nonce=" + nonce_str
    uint8_t msg[128] = {0};
    int header_len = 0;
    while (header_len < 127 && header_const[header_len]) {
        msg[header_len] = (uint8_t)header_const[header_len];
        header_len++;
    }
    const char suffix[] = "|nonce=";
    for (int i = 0; i < 7; ++i) msg[header_len + i] = (uint8_t)suffix[i];
    for (int i = 0; i < nonce_len; ++i) msg[header_len + 7 + i] = (uint8_t)nonce_str[i];
    int msg_len = header_len + 7 + nonce_len;   // bytes of real message

    // --- SHA-256 padding ---
    // We fill two 64-byte blocks if msg_len >= 56, one block otherwise.
    // block[0..63] and block[64..127] in msg[] (msg is 128 bytes, zeroed)
    msg[msg_len] = 0x80;                         // append 1-bit

    // Write bit-length as 64-bit big-endian into last 8 bytes of the LAST block
    uint64_t bit_len = (uint64_t)msg_len * 8;
    int last_block_start = (msg_len >= 56) ? 64 : 0;  // offset of last 64-byte block
    int len_offset = last_block_start + 56;             // position of 8-byte length field

    msg[len_offset+0] = (uint8_t)(bit_len >> 56);
    msg[len_offset+1] = (uint8_t)(bit_len >> 48);
    msg[len_offset+2] = (uint8_t)(bit_len >> 40);
    msg[len_offset+3] = (uint8_t)(bit_len >> 32);
    msg[len_offset+4] = (uint8_t)(bit_len >> 24);
    msg[len_offset+5] = (uint8_t)(bit_len >> 16);
    msg[len_offset+6] = (uint8_t)(bit_len >>  8);
    msg[len_offset+7] = (uint8_t)(bit_len >>  0);

    // Process block(s)
    sha256_transform(h, msg);
    if (msg_len >= 56)
        sha256_transform(h, msg + 64);

    // Produce big-endian hash bytes
    for (int i = 0; i < 8; ++i) {
        hash_out[i*4+0] = (h[i] >> 24) & 0xff;
        hash_out[i*4+1] = (h[i] >> 16) & 0xff;
        hash_out[i*4+2] = (h[i] >>  8) & 0xff;
        hash_out[i*4+3] =  h[i]        & 0xff;
    }
}

// ── ORACLE KERNEL ────────────────────────────────────────────────────────────────
__global__ void quantum_oracle_kernel(uint64_t start_nonce,
                                      uint64_t* results,
                                      int*      result_count)
{
    uint64_t idx   = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = start_nonce + idx;
    if (nonce >= c_N) return;

    uint8_t hash[32];
    sha256_gpu(nonce, hash);

    // Count leading zero BITS
    uint64_t zeros = 0;
    for (int i = 0; i < 32; ++i) {
        if (hash[i] == 0) {
            zeros += 8;
        } else {
            uint8_t byte = hash[i];
            // count leading zero bits in this byte
            for (int b = 7; b >= 0; --b) {
                if ((byte >> b) & 1) break;
                zeros++;
            }
            break;
        }
        if (zeros >= target_zeros) break;
    }

    if (zeros >= target_zeros) {
        int pos = atomicAdd(result_count, 1);
        if (pos < 65536) results[pos] = nonce;
    }
}

// ── GPU MINING DRIVER ────────────────────────────────────────────────────────────
std::vector<uint64_t> gpu_mine() {
    std::string cache_name = "oracle_gpu_"
                           + std::to_string(N_BITS) + "_"
                           + std::to_string(DIFF_BITS) + ".bin";

    // Try cache first
    {
        std::ifstream fin(cache_name, std::ios::binary);
        if (fin) {
            std::vector<uint64_t> cached;
            uint64_t nonce;
            while (fin.read((char*)&nonce, sizeof(nonce))) cached.push_back(nonce);
            std::cout << "⚡ GPU CACHE HIT: " << cached.size() << " nonces\n\n";
            return cached;
        }
    }

    std::cout << "🚀 LAUNCHING GPU MINER: 2^" << N_BITS
              << " (" << N << ") | diff=" << DIFF_BITS << " leading zero bits\n";

    // Allocate GPU memory
    uint64_t* d_results = nullptr;
    int*      d_count   = nullptr;
    cudaMalloc(&d_results, 65536 * sizeof(uint64_t));
    cudaMalloc(&d_count,   sizeof(int));

    // Copy constants to device
    // header_const holds only the base header; nonce is appended inside the kernel
    cudaMemcpyToSymbol(header_const,  BLOCK_HEADER.c_str(), BLOCK_HEADER.size() + 1);
    cudaMemcpyToSymbol(target_zeros,  &DIFF_BITS, sizeof(uint64_t));
    cudaMemcpyToSymbol(c_N_BITS,      &N_BITS,    sizeof(uint64_t));
    cudaMemcpyToSymbol(c_N,           &N,         sizeof(uint64_t));

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<uint64_t> results;

    uint64_t batch_size = (uint64_t)GRID_SIZE * BLOCK_SIZE;
    for (uint64_t batch_start = 0; batch_start < N; batch_start += batch_size) {
        cudaMemset(d_count, 0, sizeof(int));

        quantum_oracle_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(batch_start, d_results, d_count);
        cudaDeviceSynchronize();

        int h_count = 0;
        cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_count > 0) {
            int safe_count = (h_count < 65536) ? h_count : 65536;
            std::vector<uint64_t> batch_results(safe_count);
            cudaMemcpy(batch_results.data(), d_results,
                       safe_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            results.insert(results.end(), batch_results.begin(), batch_results.end());
            total_found += safe_count;
        }

        // Progress
        double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();
        double pct  = 100.0 * (double)(batch_start + batch_size) / (double)N;
        double rate = (elapsed > 0) ? (double)(batch_start + batch_size) / elapsed / 1e6 : 0.0;
        std::cout << "\r\033[KGPU: "
                  << std::fixed << std::setprecision(1) << pct << "% | "
                  << total_found.load() << " found | "
                  << std::setprecision(0) << rate << " MH/s     " << std::flush;
    }

    cudaFree(d_results);
    cudaFree(d_count);

    std::cout << "\n✅ GPU MINING COMPLETE: " << results.size() << " solutions\n";

    // Cache results
    std::ofstream fout(cache_name, std::ios::binary);
    for (auto nonce : results) fout.write((char*)&nonce, sizeof(nonce));

    return results;
}


int main(int argc, char* argv[]) {
    N_BITS    = (argc > 1) ? (uint64_t)std::atoi(argv[1]) : 25;
    DIFF_BITS = (argc > 2) ? (uint64_t)std::atoi(argv[2]) : 20;
    GRID_SIZE = (argc > 3) ? (uint64_t)std::atoi(argv[3]) : 4096;
    BLOCK_SIZE= (argc > 4) ? (uint64_t)std::atoi(argv[4]) : 256;
    if (argc > 5) BLOCK_HEADER = argv[5];
    N = 1ULL << N_BITS;

    std::cout << "GPU QUANTUM MINER — GH/s BEAST\n";
    std::cout << "N_BITS=" << N_BITS << " | DIFF=" << DIFF_BITS
              << " leading zero bits | Grid=" << GRID_SIZE << "x" << BLOCK_SIZE << "\n";
    std::cout << "Header: \"" << BLOCK_HEADER << "\"\n\n";

    // GPU check
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cout << "❌ No CUDA GPU found.\n";
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "🖥️  GPU: " << prop.name
              << " | SM " << prop.major << "." << prop.minor
              << " | " << prop.multiProcessorCount << " SMs\n\n";

    auto marked = gpu_mine();

    if (marked.empty()) {
        std::cout << "❌ No solutions found.\n"
                  << "💡 Try lower difficulty: ./pow_miner 20 15\n";
        return 1;
    }

    // Grover quantum simulation statistics
    constexpr double PI = 3.141592653589793;
    double ratio     = (double)N / (double)marked.size();
    double grover_k  = PI / 4.0 * std::sqrt(ratio);

    std::cout << "\n⚛️  GROVER QUANTUM SIMULATION:\n";
    std::cout << "   Solutions found:    " << marked.size() << "\n";
    std::cout << "   Search space:       2^" << N_BITS << " = " << N << "\n";
    std::cout << "   Optimal iterations: " << (uint64_t)grover_k << "\n";
    std::cout << "   Quantum speedup:    " << std::sqrt(ratio) << "x\n\n";

    // Verify and display the winning block using OpenSSL for ground-truth check
    uint64_t winner = marked[0];
    std::string input = BLOCK_HEADER + "|nonce=" + std::to_string(winner);
    unsigned char ssl_hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(input.c_str()), input.size(), ssl_hash);

    std::cout << "🏆 GPU MINED BLOCK 🏆\n";
    std::cout << "Input:  " << input << "\n";
    std::cout << "Nonce:  " << winner << "\n";
    std::cout << "Hash:   " << hash_to_hex(ssl_hash) << "\n";

    // Count leading zero bits in the OpenSSL hash for verification
    int verify_zeros = 0;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        if (ssl_hash[i] == 0) { verify_zeros += 8; }
        else {
            uint8_t byte = ssl_hash[i];
            for (int b = 7; b >= 0; --b) {
                if ((byte >> b) & 1) break;
                verify_zeros++;
            }
            break;
        }
    }
    std::cout << "Leading zero bits: " << verify_zeros
              << " (required: " << DIFF_BITS << ") "
              << (verify_zeros >= (int)DIFF_BITS ? "✅ VALID" : "❌ INVALID") << "\n";

    return 0;
}