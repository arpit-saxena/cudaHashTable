#ifndef HASHTABLE_H
#define HASHTABLE_H

#include "lock.cuh"

typedef long long int LL;

enum State {EMPTY, DELETED, FULL};

struct Data {
	volatile State state;
	volatile LL key;
	Lock lock;
};

struct Instruction {
	enum Type {
		Insert,
		Delete,
		Find
	};

	Type type;
	LL key;
};

class HashTable {
		Data * table;
		int size;

	public:
		HashTable(int size);
		~HashTable();

		__device__ bool insert(LL key);
		__device__ bool deleteKey(LL key);
		__device__ bool findKey(LL key);
		static void performInstructs(HashTable *table, Instruction *instructions,
			int numInstruction, bool *ret);
		static void print(HashTable *table);
};

// Contains all the CUDA kernels
namespace cu {
	// Insert array of keys into table given. Stores insert statuses in ret
	__global__ void performInstructs(
		HashTable * table,
		Instruction *instructions,
		int numInstructions,
		bool * ret);
}

namespace init_table {
	__global__
	void init_empty_table(Data *, int);
}

// Temporary hash functions
namespace HashFunction {
	__device__
	int h1(LL key, int size);

	__device__
	int h2(LL key, int size);
}

#endif /* HASHTABLE_H */