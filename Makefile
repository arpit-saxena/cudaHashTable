objects = lock.o test.o hashtable.o

all: $(objects)
	nvcc -arch=sm_61 $(objects) -o test -g -G

%.o: %.cu
	nvcc -x cu -arch=sm_61 -I. -dc $< -o $@ -DDEBUG -g -G

clean:
	rm -f *.o *.exe *.pdb *.exp *.lib test
