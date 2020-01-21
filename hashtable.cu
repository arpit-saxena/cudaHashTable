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
	while(N--){
		auto current = (table+index);
		if(current->state != FULL){
			if( current->lock.lock() ) {
				if(current->state != FULL) {
					current->state = FULL;
					current->key = key;
					current->lock.unlock();
					return true;
					// Can't guarantee that the element will be there after insert returns...
				}
			}
		}
		index += h2;
	}
	return false;
}

void HashTable::insert(LL *keys, int numKeys, bool *ret) {
	int threads_per_block = 32;
	int blocks = (numKeys + threads_per_block - 1) / threads_per_block;
	cu::insert<<<blocks, threads_per_block>>>(this, keys, numKeys, ret);
}

__global__
void cu::insert(HashTable *table, LL *keys, int numKeys, bool *ret) {
	for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < numKeys;
		id += blockDim.x * gridDim.x) {
			bool ans = table -> insert(keys[id]);
			if (ret) ret[id] = ans;
		}
}

void HashTable::check() {
	Data * hostTable = new Data[size];
	gpuErrchk( cudaMemcpy(hostTable, table, size * sizeof(Data), cudaMemcpyDeviceToHost) );
	for(int i = 0; i < size; ++i) {
		if(!(hostTable+i)->lock.trylock()) {
			printf("Hashtable locks not initialized properly!!\n");
			break;
		}
		else if((hostTable+i)->key != 0 || (hostTable+i)->state != EMPTY){
			printf("That's weird...");
			break;
		}
	}
	delete [] hostTable;
	printf("yay!!\n");
}

HashTable::~HashTable() {
	gpuErrchk( cudaFree(table) );
}