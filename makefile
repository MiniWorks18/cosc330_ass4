COMPILER = nvcc
CFLAGS = -I /usr/local/cuda-11.4/samples/common/inc
COBJS = bmpfile.o
EXES = cuda_mandelbrot test
all: ${EXES}

cuda_mandelbrot: cuda_mandelbrot.cu ${COBJS}
	${COMPILER} ${CFLAGS} cuda_mandelbrot.cu ${COBJS} -o cuda_mandelbrot

test: test.cu ${COBJS}
	${COMPILER} ${CFLAGS} test.cu ${COBJS} -o test

clean:
	rm -f *.o *~ ${EXES} ${CFILES}