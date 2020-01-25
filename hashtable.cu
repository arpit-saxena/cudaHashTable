#include "hashtable.cuh"
#include "errorcheck.h"
#include "lock.cuh"
#include <stdio.h>

__global__
void init_table::init_empty_table(Data * table, int size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < size) {
		auto ptr = table + i;
		ptr->lock.init();
		ptr->key = 0;
		ptr->state = EMPTY;
		i += gridDim.x * blockDim.x;
	}
}

__device__
int HashFunction::h1(LL key, int size) {
	return key % size;
}

__device__
int HashFunction::h2(LL key, int size) {
	return 1;
}

HashTable::HashTable(int size) {
	this->size = size;
	gpuErrchk( cudaMalloc(&table, size * sizeof(Data)) );
	int threads_per_block = 32,
		blocks = (size/threads_per_block) + (size % threads_per_block != 0);
	init_table::init_empty_table<<<blocks, threads_per_block>>>(table, size);
}

__device__
bool HashTable::insert(LL key) {
	int N = this->size, h1 = HashFunction::h1(key, size), h2 = HashFunction::h2(key, size);
	int index = h1;
	while(N > 0){
		auto current = (table+index);
		if( current->lock.lock(Thread::Insert) ) {
			__threadfence(); // Not sure if it is needed
			if(current->state != FULL) {
				current->state = FULL;
				current->key = key;
				current->lock.unlock();

				return true;
				// Can't guarantee that the element will be there after insert returns...
			}

			index = (index + h2) % size;
			N--;
			current->lock.unlock();
		}
	}
	return false;
}

__device__
bool HashTable::deleteKey(LL key) {
	int N = this->size;
	int h1 = HashFunction::h1(key, size);
	int h2 = HashFunction::h2(key, size);
	int index = h1;

	while (N > 0) {
		Data *current = table + index;
		if (current->lock.lock(Thread::Delete)) {
			__threadfence(); // Not sure if it is needed
			switch(current->state) {
				case FULL:
					if (current->key == key) {
						current->state = DELETED;
						current->lock.unlock();
						return true;
					}
					index = (index + h2) % size; N--;
					break;
				case DELETED:
					index = (index + h2) % size; N--;
					break;
				case EMPTY:
					current->lock.unlock();
					return false;
				default:
					printf("Unrecognized thread type\n");
			}
			current->lock.unlock();
		}
	}

	return false;
}

__device__
bool HashTable::findKey(LL key) {
	int N = this->size;
	int h1 = HashFunction::h1(key, size);
	int h2 = HashFunction::h2(key, size);
	
	int index = h1;

	while (N > 0) {
		Data *current = table + index;
		if (current->lock.lock(Thread::Find)) {
			switch(current->state) {
				case FULL:
					if (current->key == key) {
						current->lock.unlock();
						return true;
					}
					// No break; moves to next case
				case DELETED:
					index = (index + h2) % size;
					N--;
					break;
				case EMPTY:
					current->lock.unlock();
					return false;
			}
		}
	}

	return false;
}

void HashTable::performInstructs(HashTable *table, Instruction *ins, int numIns, bool *ret) {
	int threads_per_block = 32;
	int blocks = (numIns + threads_per_block - 1) / threads_per_block;
	cu::performInstructs<<<blocks, threads_per_block>>>(table, ins, numIns, ret);
}

__global__
void cu::performInstructs(
	HashTable * table,
	Instruction *instructions,
	int numInstructions,
	bool * ret) {
		for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < numInstructions;
			id += blockDim.x * gridDim.x) {
				bool ans;
				switch(instructions[id].type) {
					case Instruction::Insert:
						ans = table -> insert(instructions[id].key);
						break;
					case Instruction::Delete:
						ans = table -> deleteKey(instructions[id].key);
						break;
					case Instruction::Find:
						ans = table -> findKey(instructions[id].key);
						break;
				}
				if (ret) ret[id] = ans;
			}
}

__global__
void printtt(HashTable *hashTable) {
	Data *table = hashTable->table;
	int size = hashTable->size;
	for (int i = 0; i < size; i++) {
		switch(table[i].state) {
			case FULL:
				printf("Idx%d: %lld\n", i, table[i].key);
				break;
			case DELETED:
				printf("Idx%d: DELETED\n", i);
				break;
		}
	}
}

void HashTable::print(HashTable *d_hashTable) {
	gpuErrchk( cudaDeviceSynchronize() );

	printtt<<<1, 1>>>(d_hashTable);
}

HashTable::~HashTable() {
	gpuErrchk( cudaFree(table) );
}