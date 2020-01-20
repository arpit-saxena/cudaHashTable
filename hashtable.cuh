#ifndef HASHTABLE_H
#define HASHTABLE_H

#include "lock.cuh"

typedef long long int LL;

enum State {EMPTY, DELETED, FULL};

struct Data {
	State state;
	LL key;
	Lock lock;
};

class HashTable {
		Data * table;
		int size;

	public:
		HashTable(int size);
		~HashTable();

		__device__
		bool insert(LL key);

		void check();
};

namespace init_table {
	__global__
	void init_empty_table(Data *, int);
}

#endif /* HASHTABLE_H */