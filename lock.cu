#include "lock.cuh"
#include "errorcheck.h"
#include <stdlib.h>

__device__
void Lock::init() {
    state = -1;
}

__device__
bool Lock::lock(Thread type) {
    return atomicCAS((int *) &state, -1, (int) type) == -1;
}

__device__
bool Lock::unlock() {
    __threadfence();
    state = -1;
    return true;
}

__device__
bool Lock::trylock() {
    return state == -1;
}