#ifndef LOCK_H
#define LOCK_H

enum class Thread {
    Insert,
    Delete,
    Find
};

class Lock {
    volatile int state;

    public:
        // Initialises the lock
        __device__
        void init();

        // Returns true if it was able to lock, false otherwise
        __device__
        bool lock(Thread type);

        // Returns true if it was able to unlock, false if some error occurred in
        // unlocking (perhaps another thread was holding the lock). If it returns
        // false, the state of the lock is left unchanged
        __device__
        bool unlock();

        // Returns true if the lock is available to acquire
        // Does not try to acquire the lock
        __device__ __host__
        bool trylock();
};

#endif /* LOCK_H */
