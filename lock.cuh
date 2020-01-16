#ifndef LOCK_H
#define LOCK_H

class Lock {
    int state;

    public:
        // Initialises the lock
        __device__
        void init();

        // Returns true if it was able to lock, false otherwise
        __device__
        bool lock();

        // Returns true if it was able to unlock, false if some error occurred in
        // unlocking (perhaps another thread was holding the lock). If it returns
        // false, the state of the lock is left unchanged
        __device__
        bool unlock();
};

#endif /* LOCK_H */
