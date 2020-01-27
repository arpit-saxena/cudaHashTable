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
int HashFunction::h2(ULL x, int size) {
	x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x % size;
}

__device__
int HashFunction::h1(ULL a, int size) {
	a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a % size;
}

HashTable::HashTable(int size) {
	this->size = size;
	gpuErrchk( cudaMalloc(&table, size * sizeof(Data)) );
	int threads_per_block = 32,
		blocks = (size/threads_per_block) + (size % threads_per_block != 0);
	init_table::init_empty_table<<<blocks, threads_per_block>>>(table, size);
}

__device__
void HashTable::insert(ULL key, ThreadLog * status) {
	int N = this->size, h1 = HashFunction::h1(key, size), h2 = HashFunction::h2(key, size);
	int index = h1;
	while(N > 0){
		auto current = (table+index);
		if (status) ++(status -> iterations[index]);
		if(current->state == FULL) {
			index = (index + h2) % size;
			N--;
			continue;
		}
		Thread oldThread = current->lock.lock(Thread::Insert);
		switch(oldThread) {
			case Thread::Null:
				break;
			default:
				index = (index + h2) % size;
				N--;
				continue;
		}
		
		__threadfence(); // Not sure if it is needed
		if(current->state != FULL) {
			current->state = FULL;
			current->key = key;
			current->lock.unlock(Thread::Insert);
			if (status) {
				status->final_index = index;
				status->returned = true;
			}
			return;
			// Can't guarantee that the element will be there after insert returns...
		}

		index = (index + h2) % size;
		N--;
		current->lock.unlock(Thread::Insert);
	}
	if (status) {
		status->final_index = index;
		status->returned = false;
	}
}

__device__
void HashTable::deleteKey(ULL key, ThreadLog * status) {
	int N = this->size;
	int h1 = HashFunction::h1(key, size);
	int h2 = HashFunction::h2(key, size);
	int index = h1;

	while (N > 0) {
		Data *current = table + index;
		if(status) ++(status -> iterations[index]);
		if(current->state != FULL) {
			index = (index + h2) % size;
			N--;
			continue;
		}
		Thread oldThread = current->lock.lock(Thread::Delete);
		switch(oldThread) {
			case Thread::Null:
				break;
			case Thread::Insert:
				index = (index + h2) % size;
				N--;
			case Thread::Delete:
			case Thread::Find:
				continue;
		}

		__threadfence(); // Not sure if it is needed
		switch(current->state) {
			case FULL:
				if (current->key == key) {
					current->state = DELETED;
					current->lock.unlock(Thread::Delete);
					if (status) {
						status->final_index = index;
						status->returned = true;
					}
					return;
				}
				index = (index + h2) % size; N--;
				break;
			case DELETED:
				index = (index + h2) % size; N--;
				break;
			case EMPTY:
				current->lock.unlock(Thread::Delete);
				if (status) {
					status->final_index = index;
					status->returned = false;
				}
				return;
			default:
				printf("Unrecognized thread type\n");
		}
		current->lock.unlock(Thread::Delete);
	}
	if (status) {
		status->final_index = index;
		status->returned = false;
	}
}

__device__
void HashTable::findKey(ULL key, ThreadLog * status) {
	int N = this->size;
	int h1 = HashFunction::h1(key, size);
	int h2 = HashFunction::h2(key, size);
	
	int index = h1;

	while (N > 0) {
		Data *current = table + index;
		if(status) ++(status -> iterations[index]);
		if( (auto currst = current->state) != FULL ) {
			if(currst == EMPTY)	break;
			index = (index + h2) % size;
			N--;
			continue;
		}
		Thread oldThread = current->lock.lock(Thread::Find);
		switch(oldThread) {
			case Thread::Null:
			case Thread::Find:
				break;
			case Thread::Insert:
				index = (index + h2) % size;
				N--;
			case Thread::Delete:
				continue;
		}

		switch(current->state) {
			case FULL:
				if (current->key == key) {
					current->lock.unlock(Thread::Find);
					if (status) {
						status->final_index = index;
						status->returned = true;
					}
					return;
				}
				// No break; moves to next case
			case DELETED:
				current->lock.unlock(Thread::Find);
				index = (index + h2) % size;
				N--;
				break;
			case EMPTY:
				current->lock.unlock(Thread::Find);
				if (status) {
					status->final_index = index;
					status->returned = false;
				}
				return;
		}
	}
	if (status) {
		status->final_index = index;
		status->returned = false;
	}
}

void HashTable::performInstructs(HashTable *table, Instruction *ins, int numIns, ThreadLog * status) {
	int threads_per_block = 32;
	int blocks = (numIns + threads_per_block - 1) / threads_per_block;
	ThreadLog * d_status = nullptr;
	if (status) {
		gpuErrchk( cudaMalloc(&d_status, numIns*sizeof(ThreadLog)) );
		gpuErrchk( cudaMemcpy(d_status, status, numIns*sizeof(ThreadLog), cudaMemcpyDefault) );
	}
	cu::performInstructs<<<blocks, threads_per_block>>>(table, ins, numIns, d_status);
	if (status) {
		gpuErrchk( cudaMemcpy(status, d_status, numIns*sizeof(ThreadLog), cudaMemcpyDefault) );
		gpuErrchk( cudaFree(d_status) );
		for (int i = 0; i < numIns; ++i) { (status + i)->fillhostarray(); }
	}
}

__global__
void cu::performInstructs(
	HashTable * table,
	Instruction *instructions,
	int numInstructions,
	ThreadLog * status) {
		for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < numInstructions;
			id += blockDim.x * gridDim.x) {
				auto curr_status = status ? status + id : nullptr;
				switch(instructions[id].type) {
					case Instruction::Insert:
						table -> insert(instructions[id].key, curr_status);
						break;
					case Instruction::Delete:
						table -> deleteKey(instructions[id].key, curr_status);
						break;
					case Instruction::Find:
						table -> findKey(instructions[id].key, curr_status);
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