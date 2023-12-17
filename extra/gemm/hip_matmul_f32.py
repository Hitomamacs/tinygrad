import time
import numpy as np
from tinygrad.helpers import dtypes, getenv, prod, flat_mv
from tinygrad.runtime.ops_hip import HIPAllocator, HIPProgram, compile_hip

N = getenv("N", 2048)
TSIZE = getenv("T", 16)
device = 0
hipallocator = HIPAllocator(device)
a = hipallocator.alloc(N*N*4) # 4 bytes per float32
b = hipallocator.alloc(N*N*4)
c = hipallocator.alloc(N*N*4)
FLOPS = N*N*N*2
BW = N*N*3*4
na = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32).astype(np.float32)
nb = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32).astype(np.float32)
nc = np.empty(N*N, np.float32)
hipallocator.copyin(a, bytearray(na))
hipallocator.copyin(b, bytearray(nb))
#ottimizzare i launch bounds
lib = compile_hip(f""" 
    extern "C" __global__ void __launch_bounds__ (128, 1) test(float* a, __float* b, __float* c)){{ 
    __shared__ float tileA[{TSIZE}][{TSIZE}];
    __shared__ float tileB[{TSIZE}][{TSIZE}];              
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    // Identifica la riga e la colonna della matrice C da lavorare
    int row = by * {TSIZE}+ ty;
    int col = bx * {TSIZE}+ tx;
    float sum = 0.0;
    for (int m = 0; m < {N} / {TSIZE}; ++m) {{
        // Caricamento coalescente nella memoria condivisa
        tileA[ty][tx] = A[row * {N}+ (m * {TSIZE} + tx)];
        tileB[ty][tx] = B[(m * {TSIZE} + ty) * {N}+ col];

        __syncthreads();

        // Calcolo del prodotto per il tile corrente
        for (int k = 0; k < {TSIZE}; ++k) {{
            sum += tileA[ty][k] * tileB[k][tx];
        }}

        __syncthreads();
    }}

    //Scrive il risultato nella matrice C
    C[row * {N} + col] = sum;
                  
                  }}""")


prog = HIPProgram(device, "test", lib)

# Funzione per misurare il tempo di esecuzione
def timeit(fxn):
    st = time.perf_counter()
    fxn()  # Esecuzione del kernel
    return time.perf_counter() - st

global_size, local_size = [N // TSIZE, N // TSIZE, 1], [TSIZE, TSIZE, 1] #controllare qui le dim sono un po a random
print("global/local size", global_size, local_size, f"local_size:{prod(local_size)} total_size:{prod(global_size+local_size)}")
tm = min([timeit(lambda: prog(a, b, c, global_size=global_size, local_size=local_size, wait=True)) for _ in range(1000)])
hipallocator.copyout(flat_mv(nc), c)
nc = nc.reshape(N, N)
comp = na @ nb
print(f"{N*N:10d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS matmul, {BW*1e-9/tm:.2f} GB/s")
np.testing.assert_allclose(na, comp, atol=1e-2, rtol=1e-2)