# Assignment 4 - Mandelbrot Sets in CUDA

* Worth 10% for COSC330, 7% for COSC530
* **An overall mark of at least 50% across across all assessment tasks is required to pass COSC330/530**

## Aims
* Construct a CUDA C program, that makes use of the basic GPU features
* Develop a parallelisation strategy for a computationally intensive problem.
* Build some fractals!

## The Mandelbrot Set
The Mandelbrot Set, originally published by Benoit Mandelbrot in 1977 in his
book *The Fractal Geometry of Nature*, is the most famous fractal object ever
discovered. The Mandelbrot set consists of a set of points on a plain defined by:

Given a point `\((x,y)\)`, compute a sequence of other points `\((a_i,b_i)\)` for `\(i = 0,1,2 ...\)`
according to the formulas:

<div style="width:250px; margin:auto">
\(a_0 = 0 \\ b_0 = 0 \\ a_{i+1} = (a_{i})^2 (b_i)^2 + x \\ b_{i+1} = 2a_{i} b_{i} + y \)
</div>

If each point in the sequence `\((a_ib_i)\)` stays finite, then `\((x,y)\)` is a member of the set.
If the sequence of points `\((a_ib_i)\)` shoots off into infinity, then `\((x,y)\)` is not a member
of the set. 

Figure 1 shows part of the Mandelbrot set represented using a
high resolution image generated from the supplied sample
implementation.

<image src ="./assignments/images/a4/mendelbrot.png"/ style="max-width:800px">

**Figure 1** Part of the Mandelbrot set represented as an image.

## Computing the Mandelbrot set
As it is not possible to compute an infinite sequence of points and still get an answer in a finite time, an alternate criterion is needed to determine membership in the Mandelbrot set. The Mandelbrot set is a compact set, contained in the closed disk radius of 2 around the origin. In fact (x, y) only belongs to the Mandelbrot set if and only if

`\(\sqrt{a_i^2 + b_i^2} \leq 2\)`

for all `\(i\)`.
That is, if :

`\(\sqrt{a_i^2 + b_i^2} \gt 2\)`

for some `\(i\)`, then the sequence will inevitably shoot off into infinity. If we select an
arbitrary, but large enough limit on `\(i\)`, (say 1000), then we can determine if `\((x,y)\)`
is a member of the set: If `\(i\)` reaches 1000 and `\(\sqrt{a_i^2 + b_i^2} \leq 2\)`, then `\((x, y)\)` is a member of the set. If `\(\sqrt{a_i^2 + b_i^2} \gt 2\)` before `\(i\)` reaches 1000, then `\((x, y)\)` is not a member of the set.

This approach is used in the following algorithm:

```c
// 4.0 is used as the limit as the square root is not used
while(iter < MAX_ITER && zmagsqr <= 4.0){
	++iter;
	a = (aold * aold) - (bold * bold) + x;
	b = 2.0 * aold*bold + y;
	zmagsqr = a*a + b*b;
	aold = a;
	bold = b;
}
```

In this approach, the number of completed iterations (iter) before `\(\sqrt{a_i^2 + b_i^2} \gt 2\)` is used to generate the colour for a specific pixel (where the
pixel coordinates are used for the `\(x, y\)` values)

## My Single Threaded Implementation
The algorithm in the previous section has been applied in my sample application -
[mandelbrot.c](http://turing.une.edu.au/~cosc330/assignments/assignment_04/mandelbrot.c)

This program uses libbmp (i.e. [bmpfile.h](http://turing.une.edu.au/~cosc330/assignments/assignment_04/bmpfile.h), [bmpfile.c](http://turing.une.edu.au/~cosc330/assignments/assignment_04/bmpfile.c) to produce bitmap formatted
images from the Mandelbrot set. The program contains a number of constants
defined:
 
* RESOLUTION - The resolution of the fractal, effectively the zoom level
* XCENTER - X Position of the fractal in the image
* YCENTER - Y Position of the fractal in the image
* MAX_ITER - Maximum iter value from the algorithm in the previous section
* WIDTH - Image size (i.e. image resolution)
* HEIGHT - Image size (i.e. image resolution)

Analyze the example, make changes to the values and review your results to
understand how these are used in the algorithm. Compile and run my example
using the supplied makefile


## Your Task

Your task is construct a parallel implementation of the Mandelbrot algorithm using Nvidia CUDA C that produces bitmap images which contain the fractal representation of the Mandelbrot set. Your implementation should allow the user to specify any arbitary dimensions for the images produced. You are free to pull apart the example provided and make use of any other the code provided.

Your implementation should make efficient use of the various CUDA memory stores and *pitched* memory features when working with 2-dimensional structures.

The coding and exact approach used to achieve this is up to you - remember that your program will need to run on `bourbaki` and the Tesla K80 GPU installed there.

## Submission

* Your assignment will need to be submitted through the `submit` program on turing.
* Make sure that you record a script of your program compiling and working correctly. 
* Confirm that the file sizes listed in the submission receipt are not 0Kb!

## Tentative Marking Scheme

**Solution Correctness - 70%**

* Does your solution calculate the Mandelbrot set using a CUDA kernel?
* Is all memory freed at completion?
* Are the CUDA threads that are initiated used effectively? 
* Is the processing evenly distributed on the GPU (e.g. full warps and even processesing)? 
* Does your program produce a bmp file containing the fractal image?
* Does your program scale correctly with different numbers of threads and different images sizes?
* Does your program make efficent use of the available CUDA memory stores (e.g. shared, global etc.)?
* Does your program make use of *pitched* memory when working with 2-dimensional structures? (note: you are not required to use 2D structures in your kernel processing if don't wish to)

**Quality of Solution - 15%**

* Is your code broken down into functions (e.g. not more than about 60 lines - excluding braces, comments and whitespace)
* Have you generated general-purpose/reusable functions?
* Have you grouped related functions into separate libraries?
* Have you included a complete makefile with `clean` and `run` targets? 
* Does the code compile without errors/warnings?
* Is there error checking on all system calls, user inputs or source file content?
* Does your solution take appropriate action if an error occurs (e.g. make every effort to save the situation)?
* Have you avoided the use of hard-coded literals? (e.g. use #define macros)

**Documentation - 10%**

* Does your header block contain the author's name, the purpose of the program and a description of how to compile and run the solution.
* Are identifiers named in a meaningful way?
* Are any obscure constructs fully explained?
* Does each function have a header block that explains the purpose, its arguments and return value?
* Have you recorded a submission script (in the `submit` program) showing your assignment compiling and running?

**Source Formatting - 5%**

* Is your indentation consistent? Make sure that you use the `indent` utility on all source files.
* Have blank lines been used so that the code is easy to read?
* Are any lines longer than 80 characters? (`indent` can sort this out)
* Are capitalisation and naming conventions consistent? 
 
