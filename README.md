./pow_miner [bits] [diff] [grid] [block] [header]

# Example — custom header:
./pow_miner 25 20 4096 256 "My custom block header 2026"


nvcc -O3 -arch=sm_75 pow_miner.cu -o pow_miner -lssl -lcrypto

