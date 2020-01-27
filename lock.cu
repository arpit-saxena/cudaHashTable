#include "lock.cuh"
#include "errorcheck.h"
#include <stdlib.h>

__device__
void Lock::init() {
    state = 0ULL;
}

__device__ 
inline ULL atomicSub(ULL *address, ULL val) {
    ULL old = *address, assumed;
    do {
        assumed = old;
        old = atomicCAS(address, assumed, assumed - val); 
    } while (assumed != old);
     return old;
}

__device__
Thread Lock::lock(Thread type) {
    switch (type) {
        case Thread::Find:
            auto res = atomicAdd((ULL *) &state, 1ULL);

            auto threadType = static_cast<Thread>(state >> 62);
            switch (threadType) {
                case Thread::Insert:
                case Thread::Delete:
                    atomicSub((ULL *) &state, 1ULL);
                case Thread::Find:
                    if (res == 0) {
                        return Thread::Null;
                    }
                    return threadType;
            }
            break;
        case Thread::Insert:
        case Thread::Delete:
            auto res = atomicCAS((ULL *) &state, 0ULL, static_cast<ULL>(type) << 62);
            if (res == 0) {
                return Thread::Null;
            } else {
                return static_cast<Thread>(res >> 62);
            }
    }
    return Thread::Null;
}

__device__
void Lock::unlock(Thread type) {
    __threadfence();
    
    switch(type) {
        case Thread::Find:
            atomicSub((ULL *) &state, 1);
            break;
        case Thread::Insert:
        case Thread::Delete:
            ULL threadULL = ((ULL) type) << 62;
            atomicSub((ULL *) &state, threadULL);
    }
}

__device__
bool Lock::trylock() {
    return state == -1;
}