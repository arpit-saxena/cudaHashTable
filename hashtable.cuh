#ifndef HASHTABLE_H
#define HASHTABLE_H

#include "lock.cuh"
#include <iostream>

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

class HTResult {
	
	public:
		int * iterations, * h_iterations, final_index, size;
		bool returned;

		HTResult(int size);

		~HTResult();

		void to_string(std::ostream &);
		void fillhostarray();
};

class HashTable {
	
	
	public:
		Data * table;
		int size;
		HashTable(int size);
		~HashTable();

		__device__ void insert(LL key, HTResult *);
		__device__ void deleteKey(LL key, HTResult *);
		__device__ void findKey(LL key, HTResult *);
		static void performInstructs(HashTable *table, Instruction *instructions,
			int numInstruction, HTResult *);
			static void print(HashTable *table);
		};
		
namespace init_table {
	__global__
	void init_empty_table(Data *, int);
}

// Contains all the CUDA kernels
namespace cu {
	// Insert array of keys into table given. Stores insert statuses in ret
	__global__ void performInstructs(
		HashTable * table,
		Instruction *instructions,
		int numInstructions,
		HTResult *);
}

// Temporary hash functions
namespace HashFunction {
	__device__
	int h1(LL key, int size);

	__device__
	int h2(LL key, int size);
}

#endif /* HASHTABLE_H */