#include "hashtable.cuh"
#include "errorcheck.h"
#include "lock.cuh"
#include <stdio.h>
#include <cstring>

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
void HashTable::insert(LL key, ThreadLog * status) {
	int N = this->size, h1 = HashFunction::h1(key, size), h2 = HashFunction::h2(key, size);
	int index = h1;
	while(N > 0){
		auto current = (table+index);
		++(status -> iterations[index]);
		if( current->lock.lock(Thread::Insert) ) {
			__threadfence(); // Not sure if it is needed
			if(current->state != FULL) {
				current->state = FULL;
				current->key = key;
				current->lock.unlock();
				status->final_index = index;
				status->returned = true;
				return;
				// Can't guarantee that the element will be there after insert returns...
			}

			index = (index + h2) % size;
			N--;
			current->lock.unlock();
		}
	}
	status->final_index = index;
	status->returned = false;

}

__device__
void HashTable::deleteKey(LL key, ThreadLog * status) {
	int N = this->size;
	int h1 = HashFunction::h1(key, size);
	int h2 = HashFunction::h2(key, size);
	int index = h1;

	while (N > 0) {
		Data *current = table + index;
		++(status -> iterations[index]);
		if (current->lock.lock(Thread::Delete)) {
			__threadfence(); // Not sure if it is needed
			switch(current->state) {
				case FULL:
					if (current->key == key) {
						current->state = DELETED;
						current->lock.unlock();
						status->final_index = index;
						status->returned = true;		
						return;
					}
					index = (index + h2) % size; N--;
					break;
				case DELETED:
					index = (index + h2) % size; N--;
					break;
				case EMPTY:
					current->lock.unlock();
					status->final_index = index;
					status->returned = false;
					return;
				default:
					printf("Unrecognized thread type\n");
			}
			current->lock.unlock();
		}
	}
	status->final_index = index;
	status->returned = false;
}

__device__
void HashTable::findKey(LL key, ThreadLog * status) {
	int N = this->size;
	int h1 = HashFunction::h1(key, size);
	int h2 = HashFunction::h2(key, size);
	
	int index = h1;

	while (N > 0) {
		Data *current = table + index;
		++(status -> iterations[index]);
		if (current->lock.lock(Thread::Find)) {
			switch(current->state) {
				case FULL:
					if (current->key == key) {
						current->lock.unlock();
						status->final_index = index;
						status->returned = true;		
						return;
					}
					// No break; moves to next case
				case DELETED:
					current->lock.unlock();
					index = (index + h2) % size;
					N--;
					break;
				case EMPTY:
					current->lock.unlock();
					status->final_index = index;
					status->returned = false;
					return;
			}
		}
	}
	status->final_index = index;
	status->returned = false;
}

void HashTable::performInstructs(HashTable *table, Instruction *ins, int numIns, ThreadLog * status) {
	int threads_per_block = 32;
	int blocks = (numIns + threads_per_block - 1) / threads_per_block;
	ThreadLog * d_status;
	gpuErrchk( cudaMalloc(&d_status, numIns*sizeof(ThreadLog)) );
	gpuErrchk( cudaMemcpy(d_status, status, numIns*sizeof(ThreadLog), cudaMemcpyDefault) );
	cu::performInstructs<<<blocks, threads_per_block>>>(table, ins, numIns, d_status);
	gpuErrchk( cudaMemcpy(status, d_status, numIns*sizeof(ThreadLog), cudaMemcpyDefault) );
	gpuErrchk( cudaFree(d_status) );
	for (int i = 0; i < numIns; ++i) { (status + i)->fillhostarray(); }
}

__global__
void cu::performInstructs(
	HashTable * table,
	Instruction *instructions,
	int numInstructions,
	ThreadLog * status) {
		for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < numInstructions;
			id += blockDim.x * gridDim.x) {
				switch(instructions[id].type) {
					case Instruction::Insert:
						table -> insert(instructions[id].key, status + id);
						break;
					case Instruction::Delete:
						table -> deleteKey(instructions[id].key, status + id);
						break;
					case Instruction::Find:
						table -> findKey(instructions[id].key, status + id);
						break;
				}
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

void HashTable::print(HashTable *d_hashTable, ThreadLog * statuses, int statuses_size, std::ostream & out) {
	gpuErrchk( cudaDeviceSynchronize() );

	printtt<<<1, 1>>>(d_hashTable);

	for (int i = 0; i < statuses_size; ++i) {
		(statuses+i)->to_string(out << std::endl << i << ". \n");
	}
}

HashTable::~HashTable() {
	gpuErrchk( cudaFree(table) );
}

ThreadLog::ThreadLog(int size, Instruction ins) {
	this->size = size;
	gpuErrchk( cudaMalloc(&iterations, size*sizeof(int)) );
	gpuErrchk( cudaMemset(iterations, 0, size*sizeof(int)) );
	final_index = -1;
	returned = false;
	h_iterations = new int[size];
	instruction = ins;
}

ThreadLog::~ThreadLog() {
	cudaFree(this->iterations);
	delete [] h_iterations;
}

void ThreadLog::to_string(std::ostream & out) {
	out << "Instruction: ";
	switch (instruction.type) {
		case Instruction::Insert :
			out << "INSERT ";
			break;
		case Instruction::Delete:
			out << "DELETE ";
			break;
		case Instruction::Find:
			out << "FIND ";
			break;
		default:
			out << "Unrecognized instruction given to thread!!\n";
			return;
	}
	out << instruction.key << "\n" << (returned ? "Success\n" : "Failure\n");
	out << "Iterations this thread spent per index:\n";
	for(int i = 0; i < size; ++i) {
		out << h_iterations[i] << " | ";
	}
	out << "\nFinal Index = ";
	out << final_index << std::endl;
}

void ThreadLog::fillhostarray() {
	gpuErrchk( cudaMemcpy(h_iterations, iterations, size*sizeof(int), cudaMemcpyDefault) );
}