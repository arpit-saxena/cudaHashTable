#ifndef LOCK_H
#define LOCK_H

typedef unsigned long long int ULL;

enum class Thread {
    Insert = 1,
    Delete = 2,
    Find = 0,
    Null = -1
};

class Lock {
    volatile ULL state;

    public:
        // Initialises the lock
        __device__
        void init();

        // Returns type of thread which held the lock other than this thread
        __device__
        Thread lock(Thread type);

        // Returns true if it was able to unlock, false if some error occurred in
        // unlocking (perhaps another thread was holding the lock). If it returns
        // false, the state of the lock is left unchanged
        __device__
        void unlock(Thread type);

        // Returns true if the lock is available to acquire
        // Does not try to acquire the lock
        __device__ __host__
        bool trylock();
};

#endif /* LOCK_H */
