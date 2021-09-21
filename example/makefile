COMPILER = gcc
CFLAGS = -Wall
COBJS = bmpfile.o
CEXES =  mandelbrot

all: ${CEXES}

mandelbrot: mandelbrot.c ${COBJS}
	${COMPILER} ${CFLAGS} mandelbrot.c ${COBJS} -o mandelbrot -lm

%.o: %.c %.h  makefile
	${COMPILER} ${CFLAGS} -lm $< -c

clean:
	rm -f *.o *~ ${CEXES}
