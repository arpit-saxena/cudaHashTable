#include "lock.cuh"
#include "errorcheck.h"
#include <stdlib.h>

__device__
void Lock::init() {
    state = -1;
}

__device__
bool Lock::lock() {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    return atomicCAS(&state, -1, id) == -1;
}

__device__
bool Lock::unlock() {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    bool ret = atomicCAS(&state, id, -1) == id;

    #ifdef DEBUG
    
    if (!ret) {
        printf("Attempt to unlock by non-owning thread.\n");
    }

    #endif /* DEBUG */

    return ret;
}