#include "lock.cuh"
#include "errorcheck.h"
#include "hashtable.cuh"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <chrono>
#include <ctime>

__global__
void testKernel(Lock* locks, int num_locks) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    Lock *lock = locks + id % num_locks;

    while (true) {
        if (lock -> lock(Thread::Insert)) {
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

void checkLocks() {
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
}

std::pair<Instruction *, int> getInstructions(std::string name) {
    std::ifstream fin(name);
    int numIns; fin >> numIns;
    Instruction *ins = (Instruction *) malloc(numIns * sizeof(Instruction));
    for (int i = 0; i < numIns; i++) {
        std::string type; fin >> type;
        if (type == "INSERT") {
            ins[i].type = Instruction::Insert;
        } else if (type == "DELETE") {
            ins[i].type = Instruction::Delete;
        } else if (type == "FIND") {
            ins[i].type = Instruction::Find;
        } else {
            printf("Undefined instruction %s\n", type.c_str());
        }

        LL key; fin >> key;
        ins[i].key = key;
    }
    return std::make_pair(ins, numIns);
}

int main() {
    HashTable h_table(10);
    HashTable *table;
    gpuErrchk( cudaMalloc(&table, sizeof(HashTable)) );
    gpuErrchk( cudaMemcpy(table, &h_table, sizeof(HashTable), cudaMemcpyHostToDevice) );

    auto p = getInstructions("instructions.txt");
    Instruction *ins = p.first;
    int numIns = p.second;

    Instruction *d_ins;
    gpuErrchk( cudaMalloc(&d_ins, numIns * sizeof(Instruction)) );
    gpuErrchk( cudaMemcpy(d_ins, ins, numIns * sizeof(Instruction), cudaMemcpyDefault) );

    ThreadLog * statuses = (ThreadLog *)malloc(sizeof(ThreadLog)*numIns);
    for(int i = 0; i < numIns; ++i) {
        new (statuses + i) ThreadLog(h_table.size, ins[i]);
    }

    cudaEvent_t start;
    gpuErrchk( cudaEventCreate(&start) );

    cudaEvent_t stop;
    gpuErrchk( cudaEventCreate(&stop) );

    gpuErrchk( cudaEventRecord(start, NULL) );
    HashTable::performInstructs(table, d_ins, numIns, (ThreadLog *)statuses);
    gpuErrchk( cudaEventRecord(stop, NULL) );

    gpuErrchk( cudaEventSynchronize(stop) );
    float msecTotal = 0.0f;
    gpuErrchk( cudaEventElapsedTime(&msecTotal, start, stop) );


    std::ofstream fout("log.txt");
    auto time_now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(time_now);
    HashTable::print(table, statuses, numIns, fout << "Logged at: " << std::ctime(&time) << std::endl);
    std::cout << "Time taken by performInstructs: " << msecTotal << std::endl;

    gpuErrchk( cudaDeviceSynchronize() );
    free(ins);
    cudaFree(d_ins);
    cudaFree(table);
    for(int i = 0; i < numIns; ++i) {
        (statuses + i)->~ThreadLog();
    }
    free(statuses);

    return 0;
}