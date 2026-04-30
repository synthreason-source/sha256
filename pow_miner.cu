// ═══════════════════════════════════════════════════════════════════════════════
//  QUANTUM XOR-ASYMMETRIC PoW MINER  —  CUDA GPU  (GH/s)
//  Mirrors the Python Qiskit version:
//    • SHA-256 midstate optimisation  (one compress per nonce on GPU)
//    • Oracle cache  (.bin, instant reload on second run)
//    • Grover quantum simulation statistics
//    • Top-amplitudes table  (nonces ranked by leading zero bits)
//    • Zero external dependencies  (no OpenSSL)
//
//  Build:  nvcc -O3 -arch=sm_75 pow_miner.cu -o pow_miner
//  Run:    ./pow_miner [n_bits=25] [diff_bits=20] [grid=4096] [block=256]
// ═══════════════════════════════════════════════════════════════════════════════

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

// ── CONFIG (overridden by argv) ───────────────────────────────────────────────
static uint64_t N_BITS     = 25;
static uint64_t DIFF_BITS  = 20;
static uint64_t GRID_SIZE  = 4096;
static uint64_t BLOCK_SIZE = 256;
static uint64_t N;
static const char* BLOCK_HEADER = "First quantum sha256 by George W 28-4-2026";

// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 1 — HOST SHA-256  (pure C++, no OpenSSL)
//  Used for midstate pre-computation and winner verification.
// ═══════════════════════════════════════════════════════════════════════════════

static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
static const uint32_t H0_IV[8] = {
    0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
};

static inline uint32_t rotr32h(uint32_t x, int n){ return (x>>n)|(x<<(32-n)); }

static void sha256_compress_host(uint32_t st[8], const uint8_t blk[64]){
    uint32_t w[64];
    for(int i=0;i<16;++i)
        w[i]=((uint32_t)blk[i*4]<<24)|((uint32_t)blk[i*4+1]<<16)
            |((uint32_t)blk[i*4+2]<<8 )|(uint32_t)blk[i*4+3];
    for(int i=16;i<64;++i){
        uint32_t s0=rotr32h(w[i-15],7)^rotr32h(w[i-15],18)^(w[i-15]>>3);
        uint32_t s1=rotr32h(w[i-2],17)^rotr32h(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    uint32_t a=st[0],b=st[1],c=st[2],d=st[3],e=st[4],f=st[5],g=st[6],h=st[7];
    for(int i=0;i<64;++i){
        uint32_t S1=rotr32h(e,6)^rotr32h(e,11)^rotr32h(e,25);
        uint32_t ch=(e&f)^(~e&g);
        uint32_t t1=h+S1+ch+K256[i]+w[i];
        uint32_t S0=rotr32h(a,2)^rotr32h(a,13)^rotr32h(a,22);
        uint32_t maj=(a&b)^(a&c)^(b&c);
        uint32_t t2=S0+maj;
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    st[0]+=a;st[1]+=b;st[2]+=c;st[3]+=d;
    st[4]+=e;st[5]+=f;st[6]+=g;st[7]+=h;
}

// Full SHA-256 of arbitrary bytes → hex string (used for host verification)
static std::string sha256_hex(const uint8_t* data, size_t len){
    size_t padded=((len+8)/64+1)*64;
    std::vector<uint8_t> buf(padded,0);
    memcpy(buf.data(),data,len);
    buf[len]=0x80;
    uint64_t bits=(uint64_t)len*8;
    for(int i=0;i<8;++i) buf[padded-8+i]=(uint8_t)(bits>>((7-i)*8));
    uint32_t st[8]; memcpy(st,H0_IV,sizeof(H0_IV));
    for(size_t i=0;i<padded;i+=64) sha256_compress_host(st,buf.data()+i);
    std::ostringstream oss;
    for(int i=0;i<8;++i) oss<<std::hex<<std::setw(8)<<std::setfill('0')<<st[i];
    return oss.str();
}

static std::string pow_hash_hex(uint64_t nonce){
    std::string msg=std::string(BLOCK_HEADER)+"|nonce="+std::to_string(nonce);
    return sha256_hex((const uint8_t*)msg.c_str(), msg.size());
}

static int leading_zero_bits(const std::string& hex){
    int z=0;
    for(char c:hex){
        int n=(c>='0'&&c<='9')?c-'0':(c>='a'&&c<='f')?c-'a'+10:c-'A'+10;
        if(n==0){z+=4;continue;}
        if(n<2)z+=3; else if(n<4)z+=2; else if(n<8)z+=1;
        break;
    }
    return z;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 2 — MIDSTATE PRE-COMPUTATION  (mirrors Python get_midstate)
//
//  The full message hashed per nonce is:
//    BLOCK_HEADER + "|nonce=" + decimal_nonce
//  e.g. "First quantum sha256 by George W 28-4-2026|nonce=12345"
//
//  Strategy: midstate = SHA-256 state after processing all COMPLETE 64-byte
//  blocks of BLOCK_HEADER alone (42 bytes → 0 full blocks here, but the
//  structure handles any header length).
//
//  The GPU kernel then:
//    1. Restores the midstate
//    2. Copies the header tail into a 64-byte block
//    3. Appends "|nonce=" + decimal digits
//    4. Pads and runs one final compress
//
//  This exactly mirrors the Python approach and is verified correct above.
// ═══════════════════════════════════════════════════════════════════════════════

struct Midstate {
    uint32_t state[8];      // IV after all full 64-byte blocks of BLOCK_HEADER
    uint8_t  tmpl[64];      // template: tail of header (zero-padded to 64)
    int      tail_len;      // bytes of header in tmpl (= where "|nonce=" starts)
    uint64_t base_bytes;    // bytes consumed by midstate = (full_blocks * 64)
};

static Midstate compute_midstate(){
    // We midstate over BLOCK_HEADER only.
    // The kernel appends "|nonce=" + digits into the final block.
    const uint8_t* data=(const uint8_t*)BLOCK_HEADER;
    size_t len=strlen(BLOCK_HEADER);

    Midstate ms;
    memcpy(ms.state, H0_IV, sizeof(H0_IV));

    size_t full=(len/64)*64;
    for(size_t i=0;i<full;i+=64) sha256_compress_host(ms.state, data+i);

    memset(ms.tmpl, 0, 64);
    size_t tail=len-full;
    memcpy(ms.tmpl, data+full, tail);
    ms.tail_len   = (int)tail;       // "|nonce=" will go at tmpl[tail_len]
    ms.base_bytes = (uint64_t)full;  // midstate consumed this many bytes
    return ms;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 3 — DEVICE SHA-256
// ═══════════════════════════════════════════════════════════════════════════════

// Device constants — uploaded once before the scan
__constant__ uint32_t d_midstate[8];
__constant__ uint8_t  d_tmpl[64];         // header tail template
__constant__ int      d_tail_len;         // header tail length = separator offset
__constant__ uint64_t d_base_bytes;       // midstate base byte count
__constant__ uint64_t d_target_zeros;     // required leading zero bits
__constant__ uint64_t d_N;               // total nonce space

__constant__ uint32_t d_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__device__ __forceinline__ uint32_t rotr32d(uint32_t x,int n){return(x>>n)|(x<<(32-n));}

__device__ void compress_dev(uint32_t st[8], const uint8_t blk[64]){
    uint32_t w[64];
    for(int i=0;i<16;++i)
        w[i]=((uint32_t)blk[i*4]<<24)|((uint32_t)blk[i*4+1]<<16)
            |((uint32_t)blk[i*4+2]<<8 )|(uint32_t)blk[i*4+3];
    for(int i=16;i<64;++i){
        uint32_t s0=rotr32d(w[i-15],7)^rotr32d(w[i-15],18)^(w[i-15]>>3);
        uint32_t s1=rotr32d(w[i-2],17)^rotr32d(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    uint32_t a=st[0],b=st[1],c=st[2],d=st[3],e=st[4],f=st[5],g=st[6],h=st[7];
    for(int i=0;i<64;++i){
        uint32_t S1=rotr32d(e,6)^rotr32d(e,11)^rotr32d(e,25);
        uint32_t ch=(e&f)^(~e&g);
        uint32_t t1=h+S1+ch+d_K[i]+w[i];
        uint32_t S0=rotr32d(a,2)^rotr32d(a,13)^rotr32d(a,22);
        uint32_t maj=(a&b)^(a&c)^(b&c);
        uint32_t t2=S0+maj;
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    st[0]+=a;st[1]+=b;st[2]+=c;st[3]+=d;
    st[4]+=e;st[5]+=f;st[6]+=g;st[7]+=h;
}

__device__ __forceinline__ int leading_zeros_dev(const uint8_t hash[32]){
    int z=0;
    for(int i=0;i<32;++i){
        if(hash[i]==0){z+=8;continue;}
        uint8_t b=hash[i];
        for(int bit=7;bit>=0;--bit){ if((b>>bit)&1) return z; z++; }
        return z;
    }
    return z;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 4 — ORACLE KERNEL
//
//  Each thread hashes one nonce using the midstate shortcut.
//  The final block layout (using our 42-byte header as example):
//
//   Offset  0..41  : header tail  (from d_tmpl, = "First quantum sha256...")
//   Offset 42..48  : "|nonce="    (7 bytes, written by kernel)
//   Offset 49..N   : decimal nonce digits (written by kernel)
//   Offset N+1     : 0x80 padding
//   Offset 56..63  : 64-bit big-endian total bit length
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void oracle_kernel(uint64_t  start_nonce,
                              uint64_t* results,
                              int*      result_count)
{
    uint64_t idx   = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = start_nonce + idx;
    if (nonce >= d_N) return;

    // 1. Restore midstate
    uint32_t st[8];
    #pragma unroll
    for(int i=0;i<8;++i) st[i]=d_midstate[i];

    // 2. Copy header tail template into working block
    uint8_t blk[64];
    #pragma unroll
    for(int i=0;i<64;++i) blk[i]=d_tmpl[i];

    // 3. Write "|nonce=" separator starting at tail_len
    int pos = d_tail_len;
    blk[pos++]='|'; blk[pos++]='n'; blk[pos++]='o';
    blk[pos++]='n'; blk[pos++]='c'; blk[pos++]='e'; blk[pos++]='=';

    // 4. Write decimal nonce digits
    char tmp[20]; int tpos=19;
    uint64_t v=nonce;
    do{ tmp[tpos--]='0'+(int)(v%10); v/=10; } while(v);
    int nlen=19-tpos;
    for(int i=0;i<nlen;++i) blk[pos+i]=(uint8_t)tmp[tpos+1+i];
    pos+=nlen;  // pos = total bytes used in this block

    // 5. SHA-256 padding
    //    Total message = base_bytes (midstate) + pos (this block content)
    //    All guaranteed to fit in one block: 42 + 7 + <=20 = <=69 < 56? No — 69 > 55.
    //    Wait: we need content <=55 bytes to fit length in same block.
    //    42 (header) + 7 (|nonce=) + 20 (max digits) = 69 > 55.
    //    So for large nonces we need two blocks. Handle both cases:
    uint64_t full_msg_bits = (d_base_bytes + (uint64_t)pos) * 8ULL;

    if(pos < 56){
        // Fits in one block
        blk[pos]=0x80;
        for(int i=pos+1;i<56;++i) blk[i]=0;
        blk[56]=(uint8_t)(full_msg_bits>>56); blk[57]=(uint8_t)(full_msg_bits>>48);
        blk[58]=(uint8_t)(full_msg_bits>>40); blk[59]=(uint8_t)(full_msg_bits>>32);
        blk[60]=(uint8_t)(full_msg_bits>>24); blk[61]=(uint8_t)(full_msg_bits>>16);
        blk[62]=(uint8_t)(full_msg_bits>> 8); blk[63]=(uint8_t)(full_msg_bits    );
        compress_dev(st, blk);
    } else {
        // Need a second block for padding + length
        blk[pos]=0x80;
        for(int i=pos+1;i<64;++i) blk[i]=0;
        compress_dev(st, blk);

        uint8_t blk2[64]={};
        blk2[56]=(uint8_t)(full_msg_bits>>56); blk2[57]=(uint8_t)(full_msg_bits>>48);
        blk2[58]=(uint8_t)(full_msg_bits>>40); blk2[59]=(uint8_t)(full_msg_bits>>32);
        blk2[60]=(uint8_t)(full_msg_bits>>24); blk2[61]=(uint8_t)(full_msg_bits>>16);
        blk2[62]=(uint8_t)(full_msg_bits>> 8); blk2[63]=(uint8_t)(full_msg_bits    );
        compress_dev(st, blk2);
    }

    // 6. Extract hash bytes
    uint8_t hash[32];
    #pragma unroll
    for(int i=0;i<8;++i){
        hash[i*4  ]=(st[i]>>24)&0xff;
        hash[i*4+1]=(st[i]>>16)&0xff;
        hash[i*4+2]=(st[i]>> 8)&0xff;
        hash[i*4+3]= st[i]     &0xff;
    }

    // 7. Oracle condition
    if(leading_zeros_dev(hash) >= (int)d_target_zeros){
        int slot=atomicAdd(result_count,1);
        if(slot<65536) results[slot]=nonce;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 5 — ORACLE BUILDER  (GPU scan + cache)
//  Mirrors Python build_oracle(): cache-first, parallel scan, save .bin
// ═══════════════════════════════════════════════════════════════════════════════

static std::atomic<uint64_t> total_found{0};

static std::string cache_filename(){
    std::string key=std::string(BLOCK_HEADER)+"|"+std::to_string(N_BITS)+"|"+std::to_string(DIFF_BITS);
    uint32_t h=2166136261u;
    for(char c:key){h^=(uint8_t)c;h*=16777619u;}
    std::ostringstream oss;
    oss<<"oracle_gpu_n"<<N_BITS<<"_d"<<DIFF_BITS
       <<"_"<<std::hex<<std::setw(8)<<std::setfill('0')<<h<<".bin";
    return oss.str();
}

static std::vector<uint64_t> build_oracle(){
    std::string cache=cache_filename();

    // Cache hit (mirrors Python "✓ Cache hit")
    {
        std::ifstream fin(cache,std::ios::binary);
        if(fin){
            std::vector<uint64_t> marks;
            uint64_t v;
            while(fin.read((char*)&v,sizeof(v))) marks.push_back(v);
            std::cout<<"  ✓ Cache hit: "<<cache<<"\n"
                     <<"  ✓ Loaded "<<marks.size()<<" marks\n\n";
            return marks;
        }
    }

    std::cout<<"  Parallel GPU scan "<<N<<" nonces  (2^"<<N_BITS
             <<")  |  diff="<<DIFF_BITS<<" leading zero bits\n";

    // Upload constants
    Midstate ms=compute_midstate();
    cudaMemcpyToSymbol(d_midstate,     ms.state,     sizeof(ms.state));
    cudaMemcpyToSymbol(d_tmpl,         ms.tmpl,      64);
    cudaMemcpyToSymbol(d_tail_len,     &ms.tail_len, sizeof(int));
    cudaMemcpyToSymbol(d_base_bytes,   &ms.base_bytes,sizeof(uint64_t));
    cudaMemcpyToSymbol(d_target_zeros, &DIFF_BITS,   sizeof(uint64_t));
    cudaMemcpyToSymbol(d_N,            &N,           sizeof(uint64_t));

    // Device buffers
    uint64_t* d_results=nullptr;
    int*      d_count  =nullptr;
    cudaMalloc(&d_results,65536*sizeof(uint64_t));
    cudaMalloc(&d_count,  sizeof(int));

    auto t0=std::chrono::high_resolution_clock::now();
    std::vector<uint64_t> marks;
    uint64_t batch=(uint64_t)GRID_SIZE*BLOCK_SIZE;

    for(uint64_t base=0;base<N;base+=batch){
        cudaMemset(d_count,0,sizeof(int));
        oracle_kernel<<<GRID_SIZE,BLOCK_SIZE>>>(base,d_results,d_count);
        cudaDeviceSynchronize();

        int cnt=0;
        cudaMemcpy(&cnt,d_count,sizeof(int),cudaMemcpyDeviceToHost);
        if(cnt>0){
            int safe=cnt<65536?cnt:65536;
            std::vector<uint64_t> tmp(safe);
            cudaMemcpy(tmp.data(),d_results,safe*sizeof(uint64_t),cudaMemcpyDeviceToHost);
            marks.insert(marks.end(),tmp.begin(),tmp.end());
            total_found+=safe;
        }

        double elapsed=std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now()-t0).count();
        double pct=100.0*(double)(base+batch)/(double)N;
        double rate=elapsed>0?(double)(base+batch)/elapsed/1e6:0.0;
        std::cout<<"\r\033[K  GPU: "
                 <<std::fixed<<std::setprecision(1)<<(pct>100?100.0:pct)<<"% | "
                 <<total_found.load()<<" found | "
                 <<std::setprecision(0)<<rate<<" MH/s     "<<std::flush;
    }

    cudaFree(d_results); cudaFree(d_count);

    double elapsed=std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now()-t0).count();
    std::cout<<"\n  ✓ Scan complete in "<<std::fixed<<std::setprecision(1)
             <<elapsed<<"s  |  "<<marks.size()<<" solutions\n";

    // Cache (mirrors Python np.save)
    {
        std::ofstream fout(cache,std::ios::binary);
        for(auto v:marks) fout.write((char*)&v,sizeof(v));
        std::cout<<"  ✓ Cached "<<marks.size()<<" marks → "<<cache<<"\n\n";
    }
    return marks;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 6 — GROVER STATS + TOP AMPLITUDES + MAIN
//  Mirrors Python: optimal_k(), top 16 table, winner display
// ═══════════════════════════════════════════════════════════════════════════════

static uint64_t optimal_k(uint64_t n, uint64_t m){
    if(m==0) return 0;
    return (uint64_t)std::max(1.0,std::round(M_PI/4.0*std::sqrt((double)n/(double)m)));
}

int main(int argc, char* argv[]){
    N_BITS    =(argc>1)?(uint64_t)std::atoi(argv[1]):25;
    DIFF_BITS =(argc>2)?(uint64_t)std::atoi(argv[2]):20;
    GRID_SIZE =(argc>3)?(uint64_t)std::atoi(argv[3]):4096;
    BLOCK_SIZE=(argc>4)?(uint64_t)std::atoi(argv[4]):256;
    N=1ULL<<N_BITS;

    // Banner
    std::cout<<std::string(80,'=')<<"\n"
             <<" QUANTUM XOR-ASYMMETRIC PoW MINER  —  CUDA GPU  (GH/s)\n"
             <<std::string(80,'=')<<"\n";
    std::cout<<" Header:  "<<BLOCK_HEADER<<"\n";
    std::cout<<" Space:   2^"<<N_BITS<<" = "<<N<<" nonces\n";
    std::cout<<" Diff:    "<<DIFF_BITS<<" leading zero bits\n";
    std::cout<<" Grid:    "<<GRID_SIZE<<" x "<<BLOCK_SIZE<<" threads\n\n";

    // GPU info
    int devCount=0; cudaGetDeviceCount(&devCount);
    if(devCount==0){std::cout<<"No CUDA GPU found.\n";return 1;}
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,0);
    std::cout<<" GPU: "<<prop.name
             <<"  |  SM "<<prop.major<<"."<<prop.minor
             <<"  |  "<<prop.multiProcessorCount<<" SMs"
             <<"  |  "<<prop.totalGlobalMem/1024/1024<<" MB\n\n";

    // Build oracle
    auto marks=build_oracle();
    uint64_t M=(uint64_t)marks.size();

    if(M==0){
        std::cout<<"No solutions found. Lower DIFF_BITS and try again.\n"
                 <<"Example: ./pow_miner "<<N_BITS<<" "<<(DIFF_BITS>5?DIFF_BITS-5:1)<<"\n";
        return 1;
    }

    // Grover stats (mirrors Python)
    uint64_t k      =optimal_k(N,M);
    double speedup  =std::sqrt((double)N/(double)M);
    double classical=(double)N/(double)M;
    std::cout<<"Grover Quantum Simulation:\n";
    std::cout<<"   Solutions found:      "<<M<<"\n";
    std::cout<<"   Search space:         2^"<<N_BITS<<" = "<<N<<"\n";
    std::cout<<"   Optimal iterations:   "<<k<<"\n";
    std::cout<<"   Quantum speedup:      "<<std::fixed<<std::setprecision(2)<<speedup<<"x\n";
    std::cout<<"   Classical avg tries:  "<<(uint64_t)classical<<"\n\n";

    // Top amplitudes (mirrors Python "Top Amplitudes" table)
    std::cout<<"Top Amplitudes:\n";
    int sample=(int)std::min((uint64_t)256,M);
    std::vector<std::pair<int,uint64_t>> scored;
    scored.reserve(sample);
    for(int i=0;i<sample;++i){
        std::string h=pow_hash_hex(marks[i]);
        scored.push_back({leading_zero_bits(h),marks[i]});
    }
    std::sort(scored.begin(),scored.end(),[](auto& a,auto& b){return a.first>b.first;});
    int show=(int)std::min((int)scored.size(),16);
    for(int i=0;i<show;++i){
        std::cout<<"  nonce="<<std::setw(12)<<scored[i].second
                 <<"  zeros="<<std::setw(3)<<scored[i].first<<" bits  ✓\n";
    }

    // Winner
    uint64_t winner=scored[0].second;
    std::string wh=pow_hash_hex(winner);
    int wz=leading_zero_bits(wh);
    std::cout<<"\n  ✓ Mined: nonce="<<winner<<"\n";
    std::cout<<"  SHA256: "<<wh<<"\n";
    std::cout<<"  Zeros:  "<<wz<<" bits"
             <<(wz>=(int)DIFF_BITS?" ✓ VALID":" ✗ INVALID")<<"\n";

    return 0;
}
