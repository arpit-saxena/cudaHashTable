#include "lock.cuh"
#include "errorcheck.h"
#include <stdlib.h>

__device__
void Lock::init() {
    state = (int) Thread::Null;
}

__device__
Thread Lock::lock(Thread type) {
    int old = atomicCAS((int *) &state, (int) Thread::Null, (int) type);
    return static_cast<Thread>(old);
}

__device__
void Lock::unlock(Thread type) {
    __threadfence();
    state = (int) Thread::Null;
}

__device__
bool Lock::trylock() {
    return state == -1;
}