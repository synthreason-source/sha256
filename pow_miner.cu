// sha256_superminer.cu
// Super-miner prototype with CLI for header, difficulty, max attempts, etc.
// Compile:
//   nvcc -O3 -arch=sm_50 -std=c++14 -o sha256_superminer sha256_superminer.cu
//
// Run examples:
//   ./sha256_superminer -H "example header" -z 1 -n 10000 -b 4096 -m 5000
//   ./sha256_superminer -f header.bin -t 00000fffffffffffffffffffffffffffffffffffffffffffffffffffffffffff -n 500000
//
// Notes: research prototype for local benchmarking only.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <sstream>
#include <string>
#include <getopt.h>

#include <cuda_runtime.h>

// ----------------- Defaults -----------------
static const std::string DEFAULT_HEADER_STR = "example block header for super miner prototype";
static const std::string DEFAULT_LOG_CSV = "gpu_superminer_attempts.csv";
static const std::string DEFAULT_BEST_CSV = "gpu_superminer_hits.csv";
// --------------------------------------------

// Device constant memory for K
__constant__ uint32_t K_dev[64];

// ----------------- Device helpers -----------------
__device__ __forceinline__ uint32_t rotr_d(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ uint32_t shr_d(uint32_t x, int n) { return x >> n; }
__device__ __forceinline__ uint32_t Sigma0_d(uint32_t x) { return rotr_d(x,2) ^ rotr_d(x,13) ^ rotr_d(x,22); }
__device__ __forceinline__ uint32_t Sigma1_d(uint32_t x) { return rotr_d(x,6) ^ rotr_d(x,11) ^ rotr_d(x,25); }
__device__ __forceinline__ uint32_t sigma0_d(uint32_t x) { return rotr_d(x,7) ^ rotr_d(x,18) ^ shr_d(x,3); }
__device__ __forceinline__ uint32_t sigma1_d(uint32_t x) { return rotr_d(x,17) ^ rotr_d(x,19) ^ shr_d(x,10); }
__device__ __forceinline__ uint32_t Ch_d(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
__device__ __forceinline__ uint32_t Maj_d(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }

// Kernel: one 512-bit block per thread
extern "C"
__global__ void sha256_compress_kernel(const uint8_t* __restrict__ blocks,
                                       const uint32_t* __restrict__ H_init,
                                       uint8_t* __restrict__ digests_out,
                                       uint64_t num_blocks) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_blocks) return;

    uint32_t a = H_init[0], b = H_init[1], c = H_init[2], d = H_init[3];
    uint32_t e = H_init[4], f = H_init[5], g = H_init[6], h = H_init[7];

    const uint8_t* block = blocks + tid * 64;
    uint32_t W[64];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int off = i * 4;
        W[i] = ((uint32_t)block[off] << 24) | ((uint32_t)block[off + 1] << 16)
             | ((uint32_t)block[off + 2] << 8) | ((uint32_t)block[off + 3]);
    }
    #pragma unroll
    for (int t = 16; t < 64; ++t) {
        W[t] = sigma1_d(W[t-2]) + W[t-7] + sigma0_d(W[t-15]) + W[t-16];
    }

    #pragma unroll
    for (int t = 0; t < 64; ++t) {
        uint32_t T1 = h + Sigma1_d(e) + Ch_d(e,f,g) + K_dev[t] + W[t];
        uint32_t T2 = Sigma0_d(a) + Maj_d(a,b,c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    uint32_t H_out[8];
    H_out[0] = a + H_init[0];
    H_out[1] = b + H_init[1];
    H_out[2] = c + H_init[2];
    H_out[3] = d + H_init[3];
    H_out[4] = e + H_init[4];
    H_out[5] = f + H_init[5];
    H_out[6] = g + H_init[6];
    H_out[7] = h + H_init[7];

    uint8_t* out = digests_out + tid * 32;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t v = H_out[i];
        out[i*4 + 0] = (v >> 24) & 0xff;
        out[i*4 + 1] = (v >> 16) & 0xff;
        out[i*4 + 2] = (v >> 8) & 0xff;
        out[i*4 + 3] = (v >> 0) & 0xff;
    }
}

// ----------------- Host helpers -----------------
static inline uint32_t rotr_h(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static inline uint32_t shr_h(uint32_t x, int n) { return x >> n; }
static inline uint32_t Sigma0_h(uint32_t x) { return rotr_h(x,2) ^ rotr_h(x,13) ^ rotr_h(x,22); }
static inline uint32_t Sigma1_h(uint32_t x) { return rotr_h(x,6) ^ rotr_h(x,11) ^ rotr_h(x,25); }
static inline uint32_t sigma0_h(uint32_t x) { return rotr_h(x,7) ^ rotr_h(x,18) ^ shr_h(x,3); }
static inline uint32_t sigma1_h(uint32_t x) { return rotr_h(x,17) ^ rotr_h(x,19) ^ shr_h(x,10); }
static inline uint32_t Ch_h(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
static inline uint32_t Maj_h(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }

void host_compress_block(const uint8_t block[64], const uint32_t K[64], uint32_t H[8]) {
    uint32_t W[64];
    for (int i = 0; i < 16; ++i) {
        W[i] = ((uint32_t)block[i*4] << 24) | ((uint32_t)block[i*4 + 1] << 16)
             | ((uint32_t)block[i*4 + 2] << 8) | ((uint32_t)block[i*4 + 3]);
    }
    for (int t = 16; t < 64; ++t) {
        uint32_t s1 = sigma1_h(W[t-2]);
        uint32_t s0 = sigma0_h(W[t-15]);
        W[t] = (s1 + W[t-7] + s0 + W[t-16]) & 0xffffffff;
    }
    uint32_t a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
    for (int t = 0; t < 64; ++t) {
        uint32_t T1 = (h + Sigma1_h(e) + Ch_h(e,f,g) + K[t] + W[t]) & 0xffffffff;
        uint32_t T2 = (Sigma0_h(a) + Maj_h(a,b,c)) & 0xffffffff;
        h = g; g = f; f = e; e = (d + T1) & 0xffffffff;
        d = c; c = b; b = a; a = (T1 + T2) & 0xffffffff;
    }
    H[0] = (H[0] + a) & 0xffffffff;
    H[1] = (H[1] + b) & 0xffffffff;
    H[2] = (H[2] + c) & 0xffffffff;
    H[3] = (H[3] + d) & 0xffffffff;
    H[4] = (H[4] + e) & 0xffffffff;
    H[5] = (H[5] + f) & 0xffffffff;
    H[6] = (H[6] + g) & 0xffffffff;
    H[7] = (H[7] + h) & 0xffffffff;
}

static inline uint32_t frac_cuberoot_const(uint32_t prime) {
    double r = cbrt((double)prime);
    double f = r - floor(r);
    uint32_t k = (uint32_t) floor(f * (double)(1ULL << 32));
    return k;
}

std::vector<uint32_t> first_n_primes(int n) {
    std::vector<uint32_t> primes; primes.reserve(n);
    uint32_t cand = 2;
    while ((int)primes.size() < n) {
        bool is_prime = true;
        for (uint32_t p : primes) {
            if ((uint64_t)p * p > cand) break;
            if (cand % p == 0) { is_prime = false; break; }
        }
        if (is_prime) primes.push_back(cand);
        cand += (cand == 2) ? 1 : 2;
    }
    return primes;
}

void build_standard_K(uint32_t K_out[64]) {
    auto primes = first_n_primes(64);
    for (int i = 0; i < 64; ++i) K_out[i] = frac_cuberoot_const(primes[i]);
}

std::vector<uint8_t> build_final_block_host(const std::vector<uint8_t>& header_remain,
                                            uint64_t nonce,
                                            int nonce_bytes,
                                            size_t total_header_len) {
    std::vector<uint8_t> msg;
    msg.insert(msg.end(), header_remain.begin(), header_remain.end());
    for (int i = nonce_bytes - 1; i >= 0; --i) {
        msg.push_back((uint8_t)((nonce >> (8 * i)) & 0xff));
    }
    msg.push_back(0x80);
    while ((msg.size() * 8) % 512 != 448) msg.push_back(0x00);
    uint64_t total_bits = ((uint64_t)total_header_len + (uint64_t)nonce_bytes) * 8ULL;
    for (int i = 7; i >= 0; --i) msg.push_back((uint8_t)((total_bits >> (8 * i)) & 0xff));
    return msg;
}

std::unordered_map<uint64_t,double> load_priority_csv(const std::string& path) {
    std::unordered_map<uint64_t,double> scores;
    if (path.empty()) return scores;
    std::ifstream ifs(path);
    if (!ifs) return scores;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.find(",") == std::string::npos) continue;
        size_t comma = line.find(',');
        try {
            uint64_t n = std::stoull(line.substr(0, comma));
            double s = std::stod(line.substr(comma+1));
            scores[n] = s;
        } catch (...) { continue; }
    }
    return scores;
}

std::vector<uint64_t> build_nonces(size_t count, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::vector<uint64_t> v; v.reserve(count);
    for (size_t i = 0; i < count; ++i) v.push_back(rng());
    return v;
}

inline bool digest_meets_target_prefix(const uint8_t* digest, const std::vector<uint8_t>& target_prefix) {
    for (size_t i = 0; i < target_prefix.size(); ++i) if (digest[i] != target_prefix[i]) return false;
    return true;
}

bool parse_hex_32bytes(const std::string& hex, std::vector<uint8_t>& out) {
    if (hex.size() != 64) return false;
    out.resize(32);
    for (int i = 0; i < 32; ++i) {
        unsigned val = 0;
        if (sscanf(hex.c_str() + i*2, "%2x", &val) != 1) return false;
        out[i] = (uint8_t)val;
    }
    return true;
}

void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "Options:\n"
        "  -H, --header TEXT           header text (bytes)\n"
        "  -f, --header-file PATH      header file (bytes)\n"
        "  -z, --leading-zero-bytes N  difficulty: N leading zero bytes\n"
        "  -t, --target-hex 64HEX      explicit 32-byte target hex (64 hex chars)\n"
        "  -m, --max-attempts N        stop after N attempts (0 = unlimited)\n"
        "  -n, --nonces N              number of nonces to generate (default 200000)\n"
        "  -b, --batch B               batch size (blocks per kernel launch)\n"
        "  -s, --seed S                RNG seed for nonces\n"
        "  -o, --nonce-bytes 1|2|4|8   nonce size in bytes (default 4)\n"
        "  -p, --priority PATH         CSV 'nonce,score' to prioritize nonces\n"
        "  -l, --log-csv PATH          attempts CSV path\n"
        "  -e, --best-csv PATH         hits CSV path\n"
        "  -?                         show this help\n", prog);
}

// ----------------- Main -----------------
int main(int argc, char** argv) {
    // CLI defaults
    std::string header_text;
    std::string header_file;
    int leading_zero_bytes = -1;
    std::string target_hex;
    uint64_t max_attempts = 0;
    size_t nonce_count = 200000;
    size_t batch = 1 << 12;
    unsigned seed = 20260514;
    int nonce_bytes = 4;
    std::string priority_csv;
    std::string log_csv = DEFAULT_LOG_CSV;
    std::string best_csv = DEFAULT_BEST_CSV;

    const struct option longopts[] = {
        {"header", required_argument, nullptr, 'H'},
        {"header-file", required_argument, nullptr, 'f'},
        {"leading-zero-bytes", required_argument, nullptr, 'z'},
        {"target-hex", required_argument, nullptr, 't'},
        {"max-attempts", required_argument, nullptr, 'm'},
        {"nonces", required_argument, nullptr, 'n'},
        {"batch", required_argument, nullptr, 'b'},
        {"seed", required_argument, nullptr, 's'},
        {"nonce-bytes", required_argument, nullptr, 'o'},
        {"priority", required_argument, nullptr, 'p'},
        {"log-csv", required_argument, nullptr, 'l'},
        {"best-csv", required_argument, nullptr, 'e'},
        {"help", no_argument, nullptr, '?'},
        {nullptr,0,nullptr,0}
    };

    while (true) {
        int idx = 0;
        int c = getopt_long(argc, argv, "H:f:z:t:m:n:b:s:o:p:l:e:?", longopts, &idx);
        if (c == -1) break;
        switch (c) {
            case 'H': header_text = optarg; break;
            case 'f': header_file = optarg; break;
            case 'z': leading_zero_bytes = atoi(optarg); break;
            case 't': target_hex = optarg; break;
            case 'm': max_attempts = strtoull(optarg, nullptr, 10); break;
            case 'n': nonce_count = strtoull(optarg, nullptr, 10); break;
            case 'b': batch = strtoull(optarg, nullptr, 10); break;
            case 's': seed = (unsigned)strtoul(optarg, nullptr, 10); break;
            case 'o': nonce_bytes = atoi(optarg); break;
            case 'p': priority_csv = optarg; break;
            case 'l': log_csv = optarg; break;
            case 'e': best_csv = optarg; break;
            case '?': usage(argv[0]); return 0;
            default: usage(argv[0]); return 1;
        }
    }

    if (nonce_bytes !=1 && nonce_bytes !=2 && nonce_bytes !=4 && nonce_bytes !=8) {
        std::fprintf(stderr, "nonce-bytes must be 1,2,4, or 8\n"); return 1;
    }

    // header bytes
    std::vector<uint8_t> header;
    if (!header_file.empty()) {
        std::ifstream ifs(header_file, std::ios::binary);
        if (!ifs) { std::fprintf(stderr, "Failed to open header-file '%s'\n", header_file.c_str()); return 1; }
        header.assign((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    } else if (!header_text.empty()) {
        header.assign(header_text.begin(), header_text.end());
    } else {
        header.assign(DEFAULT_HEADER_STR.begin(), DEFAULT_HEADER_STR.end());
    }

    // target handling
    std::vector<uint8_t> target_prefix;
    std::vector<uint8_t> target_full;
    if (!target_hex.empty()) {
        if (!parse_hex_32bytes(target_hex, target_full)) { std::fprintf(stderr,"Invalid target-hex\n"); return 1; }
    } else if (leading_zero_bytes >= 0) {
        target_prefix.assign((size_t)leading_zero_bytes, 0x00);
    } else {
        target_prefix.assign(1, 0x00);
    }

    std::printf("Config: header len=%zu, nonce_count=%zu, batch=%zu, nonce_bytes=%d, seed=%u, max_attempts=%llu\n",
                header.size(), nonce_count, batch, nonce_bytes, seed, (unsigned long long)max_attempts);
    if (!target_full.empty()) std::printf("Target: explicit 32-byte hex\n"); else std::printf("Target: %zu leading zero bytes\n", target_prefix.size());
    if (!priority_csv.empty()) std::printf("Priority CSV: %s\n", priority_csv.c_str());

    // Build K and copy to device constant memory
    uint32_t K_host[64];
    {
        auto primes = first_n_primes(64);
        for (int i = 0; i < 64; ++i) K_host[i] = frac_cuberoot_const(primes[i]);
    }
    cudaError_t cerr = cudaMemcpyToSymbol(K_dev, K_host, sizeof(K_host));
    if (cerr != cudaSuccess) { std::fprintf(stderr, "cudaMemcpyToSymbol(K_dev) failed: %s\n", cudaGetErrorString(cerr)); return 1; }

    // Precompute header full blocks
    uint32_t H_state[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                            0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
    size_t full_blocks = header.size() / 64;
    size_t consumed = 0;
    if (full_blocks > 0) {
        for (size_t i = 0; i < full_blocks; ++i) {
            uint8_t block[64];
            std::memcpy(block, &header[i*64], 64);
            host_compress_block(block, K_host, H_state);
        }
        consumed = full_blocks * 64;
    }
    std::vector<uint8_t> header_remain(header.begin() + consumed, header.end());
    size_t total_header_len = header.size();

    // Build nonces
    std::vector<uint64_t> nonces = build_nonces(nonce_count, seed);

    // Priority ordering
    if (!priority_csv.empty()) {
        auto scores = load_priority_csv(priority_csv);
        std::sort(nonces.begin(), nonces.end(), [&](uint64_t a, uint64_t b){
            double sa = scores.count(a) ? scores[a] : 0.0;
            double sb = scores.count(b) ? scores[b] : 0.0;
            if (sa != sb) return sa > sb;
            return a < b;
        });
    } else {
        std::mt19937_64 rng(seed);
        std::shuffle(nonces.begin(), nonces.end(), rng);
    }

    // Device buffers
    const size_t max_batch = batch;
    size_t blocks_bytes = max_batch * 64;
    size_t digests_bytes = max_batch * 32;
    uint8_t* d_blocks = nullptr;
    uint8_t* d_digests = nullptr;
    uint32_t* d_Hinit = nullptr;
    cudaMalloc((void**)&d_blocks, blocks_bytes);
    cudaMalloc((void**)&d_digests, digests_bytes);
    cudaMalloc((void**)&d_Hinit, sizeof(H_state));
    cudaMemcpy(d_Hinit, H_state, sizeof(H_state), cudaMemcpyHostToDevice);

    // Logs
    std::ofstream logf(log_csv);
    logf << "timestamp,nonce,digest_hex,passed\n";
    std::ofstream bestf(best_csv, std::ios::app);
    if (bestf.tellp() == 0) bestf << "timestamp,nonce,digest_hex\n";

    // Host buffers
    std::vector<uint8_t> host_blocks;
    host_blocks.resize(max_batch * 64);
    std::vector<uint8_t> host_digests;
    host_digests.resize(max_batch * 32);

    int threads_per_block = 256;
    if (threads_per_block > 1024) threads_per_block = 1024;

    // main loop
    size_t idx = 0, attempts = 0;
    auto t_start = std::chrono::high_resolution_clock::now();

    while (idx < nonces.size()) {
        if (max_attempts && attempts >= max_attempts) break;
        size_t this_batch = std::min((size_t)max_batch, nonces.size() - idx);

        for (size_t i = 0; i < this_batch; ++i) {
            auto b = build_final_block_host(header_remain, nonces[idx + i], nonce_bytes, total_header_len);
            if (b.size() != 64) {
                std::memset(&host_blocks[i*64], 0, 64);
            } else {
                std::memcpy(&host_blocks[i*64], b.data(), 64);
            }
        }

        cudaMemcpy(d_blocks, host_blocks.data(), this_batch * 64, cudaMemcpyHostToDevice);
        int grid = (int)((this_batch + threads_per_block - 1) / threads_per_block);
        sha256_compress_kernel<<<grid, threads_per_block>>>(d_blocks, d_Hinit, d_digests, (uint64_t)this_batch);
        cerr = cudaGetLastError();
        if (cerr != cudaSuccess) { std::fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(cerr)); break; }
        cudaDeviceSynchronize();

        cudaMemcpy(host_digests.data(), d_digests, this_batch * 32, cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < this_batch; ++i) {
            uint8_t* digest = &host_digests[i*32];
            bool ok = false;
            if (!target_full.empty()) {
                // full 32-byte lexicographic compare (big-endian)
                ok = (memcmp(digest, target_full.data(), 32) <= 0);
            } else {
                ok = digest_meets_target_prefix(digest, target_prefix);
            }
            double ts = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count();
            std::ostringstream oss; oss << std::fixed << std::setprecision(6) << ts;
            logf << oss.str() << "," << nonces[idx + i] << ",";
            for (int j = 0; j < 32; ++j) { char buf[3]; std::sprintf(buf, "%02x", digest[j]); logf << buf; }
            logf << "," << (ok ? 1 : 0) << "\n";
            if (ok) {
                bestf << oss.str() << "," << nonces[idx + i] << ",";
                for (int j = 0; j < 32; ++j){ char buf[3]; std::sprintf(buf, "%02x", digest[j]); bestf << buf; }
                bestf << "\n";
            }
            attempts++;
            if (max_attempts && attempts >= max_attempts) break;
        }

        idx += this_batch;
        if ((idx % (max_batch * 10)) == 0) {
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            std::printf("Processed %zu/%zu nonces, attempts=%zu, elapsed=%.2fs\n", idx, nonces.size(), attempts, elapsed);
        }
    }

    logf.close(); bestf.close();
    cudaFree(d_blocks); cudaFree(d_digests); cudaFree(d_Hinit);
    cudaDeviceReset();

    auto t_end = std::chrono::high_resolution_clock::now();
    double tot_s = std::chrono::duration<double>(t_end - t_start).count();
    std::printf("Done: processed %zu nonces in %.2fs\n", attempts, tot_s);

    return 0;
}
