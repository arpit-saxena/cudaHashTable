#include "lock.cuh"
#include "errorcheck.h"
#include "hashtable.cuh"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <chrono>
#include <ctime>

const int hashTable_size = 20;
const bool logging = false;

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

using std::vector;

vector<vector<Instruction>> getInstructions(std::string name) {
    std::ifstream fin(name);
    int numBlocks; fin >> numBlocks;
    vector<vector<Instruction>> instructions(numBlocks);
    for(int blockNum = 0; blockNum < numBlocks; blockNum++) {
        int numIns; fin >> numIns;
        for (int i = 0; i < numIns; i++) {
            Instruction ins;
            std::string type; fin >> type;
            if (type == "INSERT") {
                ins.type = Instruction::Insert;
            } else if (type == "DELETE") {
                ins.type = Instruction::Delete;
            } else if (type == "FIND") {
                ins.type = Instruction::Find;
            } else {
                printf("Undefined instruction %s\n", type.c_str());
            }

            ULL key; fin >> key;
            ins.key = key;
            instructions[blockNum].push_back(ins);
        }
    }
    return instructions;
}

int main(int argc, char **argv) {
    HashTable h_table(hashTable_size);
    HashTable *table;
    gpuErrchk( cudaMalloc(&table, sizeof(HashTable)) );
    gpuErrchk( cudaMemcpy(table, &h_table, sizeof(HashTable), cudaMemcpyHostToDevice) );

    auto p = getInstructions(argv[1]);

    std::ofstream fout;
    if (logging) fout.open("log.txt");
    for (auto &v_ins : p) {
        int numIns = v_ins.size();
        Instruction *ins = (Instruction *) malloc(sizeof(Instruction) * numIns);
        std::copy(v_ins.begin(), v_ins.end(), ins);

        Instruction *d_ins;
        gpuErrchk( cudaMalloc(&d_ins, numIns * sizeof(Instruction)) );
        gpuErrchk( cudaMemcpy(d_ins, ins, numIns * sizeof(Instruction), cudaMemcpyDefault) );

        ThreadLog * statuses = nullptr;
        if (logging) {
            statuses = (ThreadLog *)malloc(sizeof(ThreadLog)*numIns);
            for(int i = 0; i < numIns; ++i) {
                new (statuses + i) ThreadLog(h_table.size, ins[i]);
            }
        }

        cudaEvent_t start;
        gpuErrchk( cudaEventCreate(&start) );

        cudaEvent_t stop;
        gpuErrchk( cudaEventCreate(&stop) );

        gpuErrchk( cudaEventRecord(start, NULL) );
        HashTable::performInstructs(table, d_ins, numIns, statuses);
        gpuErrchk( cudaEventRecord(stop, NULL) );

        gpuErrchk( cudaEventSynchronize(stop) );
        float msecTotal = 0.0f;
        gpuErrchk( cudaEventElapsedTime(&msecTotal, start, stop) );

        auto time_now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(time_now);
        if (statuses)
            HashTable::print(table, statuses, numIns, fout << "\n\nLogged at: " << std::ctime(&time));
        std::cout << "Time taken by performInstructs: " << msecTotal << " ms" << std::endl;

        gpuErrchk( cudaDeviceSynchronize() );
        free(ins);
        cudaFree(d_ins);
        for(int i = 0; i < numIns; ++i) {
            (statuses + i)->~ThreadLog();
        }
    }

    cudaFree(table);
    return 0;
}