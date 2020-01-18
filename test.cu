#include "lock.cuh"
#include "errorcheck.h"
#include <stdio.h>

__global__
void testKernel(Lock* locks, int num_locks) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    Lock *lock = locks + id % num_locks;

    while (true) {
        if (lock -> lock()) {
            printf("[%d] Locked %d\n", id, id % num_locks);

            if (lock -> unlock()) {
                printf("[%d] Unlocked %d\n", id, id % num_locks);
                break;
            } else {
                printf("[%d] ERROR: Not able to unlock %d\n", id, id % num_locks);
            }
        } else {
            printf("[%d] Failed to lock %d\n", id, id % num_locks);
        }
    }
}

__global__
void initLocks(Lock *locks, int num_locks) {
    for (
        int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < num_locks;
        i += gridDim.x * blockDim.x
    ) {
        locks[i].init();
    }
}

int main() {
    Lock *locks;
    int num_locks = 2;

    gpuErrchk( cudaMalloc(&locks, num_locks * sizeof(Lock)) );
    initLocks<<<1, num_locks>>>(locks, num_locks);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    testKernel<<<1, 4>>>(locks, num_locks);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk( cudaFree(locks) );

    return 0;
}