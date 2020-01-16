objects = lock.o test.o

all: $(objects)
	nvcc -arch=sm_61 $(objects) -o test -g

%.o: %.cu
	nvcc -x cu -arch=sm_61 -I. -dc $< -o $@ -DDEBUG -g

clean:
	rm -f *.o test
