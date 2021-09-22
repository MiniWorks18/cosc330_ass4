# COSC330 - Assignment 4
## Purpose
Calculate the Mandelbrot set and visualise it by colouring pixels and placing them in a BMP file
## Usage
### Recommended:
Use the makefile, type `make` then run the appropriate executable

### Alternative
Compile using `nvcc -I /usr/local/cuda-11.4/samples/common/inc <file>.cu bmpfile.o <file>`
Run using `./<executable> <width> <height>`

### Output
The results from the execution should be found in the `mandelbrot.bmp` file
