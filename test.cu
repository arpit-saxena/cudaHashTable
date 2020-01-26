#include "lock.cuh"
#include "errorcheck.h"
#include "hashtable.cuh"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

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

            LL key; fin >> key;
            ins.key = key;
            instructions[blockNum].push_back(ins);
        }
    }
    return instructions;
}

int main() {
    HashTable h_table(100);
    HashTable *table;
    gpuErrchk( cudaMalloc(&table, sizeof(HashTable)) );
    gpuErrchk( cudaMemcpy(table, &h_table, sizeof(HashTable), cudaMemcpyHostToDevice) );

    auto p = getInstructions("instructions.txt");

    for (auto &v_ins : p) {
        int numIns = v_ins.size();
        Instruction ins[numIns];
        std::copy(v_ins.begin(), v_ins.end(), ins);

        Instruction *d_ins;
        gpuErrchk( cudaMalloc(&d_ins, numIns * sizeof(Instruction)) );
        gpuErrchk( cudaMemcpy(d_ins, ins, numIns * sizeof(Instruction), cudaMemcpyDefault) );

        HTResult * statuses = (HTResult *)malloc(sizeof(HTResult)*numIns);
        for(int i = 0; i < numIns; ++i) {
            new (statuses + i) HTResult(h_table.size);
        }
        HashTable::performInstructs(table, d_ins, numIns, (HTResult *)statuses);
        HashTable::print(table);

        gpuErrchk( cudaDeviceSynchronize() );
        free(ins);
        cudaFree(d_ins);
        cudaFree(table);
        for(int i = 0; i < numIns; ++i) {
            (statuses + i)->~HTResult();
        }
        free(statuses);

    }

    return 0;
}