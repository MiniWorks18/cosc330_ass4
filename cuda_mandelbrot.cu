/*
Author: Tully McDonald
Purpose: Calculate and visualise the Mandelbrot set for a given width/height
Contact: tmcdon26@myune.edu.au

Credit: Portions of this program was inspired by UNE (COSC330)

*/

#include <stdio.h>
#include <cuda_runtime.h>
#include "bmpfile.h"

#define FILENAME "mandelbrot.bmp"
#define RESOLUTION 8700.0
#define XCENTER -0.55
#define YCENTER 0.6
#define MAX_ITER 1000

#define COLOUR_DEPTH 255
#define COLOUR_MAX 240.0
#define GRADIENT_COLOUR_MAX 230.0

// Device function called by the Mandelbrot kernel to calculate the 
// colors of our pixels
__device__ void GroundColorMix(float* color, float x, float min, float max)
{
  /*
   * Red = 0
   * Green = 1
   * Blue = 2
   */
    float posSlope = (max-min)/60;
    float negSlope = (min-max)/60;

    if( x < 60 )
    {
        color[0] = max;
        color[1] = posSlope*x+min;
        color[2] = min;
        return;
    }
    else if ( x < 120 )
    {
        color[0] = negSlope*x+2.0*max+min;
        color[1] = max;
        color[2] = min;
        return;
    }
    else if ( x < 180  )
    {
        color[0] = min;
        color[1] = max;
        color[2] = posSlope*x-2.0*max+min;
        return;
    }
    else if ( x < 240  )
    {
        color[0] = min;
        color[1] = negSlope*x+4.0*max+min;
        color[2] = max;
        return;
    }
    else if ( x < 300  )
    {
        color[0] = posSlope*x-4.0*max+min;
        color[1] = min;
        color[2] = max;
        return;
    }
    else
    {
        color[0] = max;
        color[1] = min;
        color[2] = negSlope*x+6*max;
        return;
    }
}

// Mandelbrot kernel, performs computations across all pixels to determine if
// coordinates fall under the mandelbrot set 
__global__ void Mandelbrot(rgb_pixel_t *A, const int width, const int height,
 const int xoffset, const int yoffset) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.x;
    int row = blockIdx.x;
    
    float x = XCENTER + (xoffset + col) / RESOLUTION;
    float y = YCENTER + (yoffset - row) / RESOLUTION;

    //Mandelbrot stuff
    float a = 0;
    float b = 0;
    float aold = 0;
    float bold = 0;
    float zmagsqr = 0;
    int iter = 0;
    float color[3];
    float x_col = 0;
    // Check the first MAX_ITER numbers to see if the results are probably
    // approaching infinity
    while(iter < MAX_ITER && zmagsqr <= 4.0){
        ++iter;
	    a = (aold * aold) - (bold * bold) + x;
        b = 2.0 * aold*bold + y;
        zmagsqr = a*a + b*b;
        aold = a;
        bold = b;	
    }
    // Fetch the color and set the pixel values
    x_col =  (COLOUR_MAX - (( ((float) iter / ((float) MAX_ITER) 
    * GRADIENT_COLOUR_MAX))));
    GroundColorMix(color, x_col, 1, COLOUR_DEPTH);
    A[id].red = color[0];
    A[id].green = color[1];
    A[id].blue = color[2];
}

// Check valid params and place into variables
void parse_args(int argc, char *argv[], int *width, int *height) {
    if (argc != 3 || (*width = atoi(argv[1])) == 0 
        || (*height = atoi(argv[2])) == 0) {
        printf("Usage: %s <width> <height>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
}

// Termination sequence, print error message and exit
void terminate(const char msg[], const char err[]) {
    fprintf(stderr, msg);
    fprintf(stderr, err);
    fprintf(stderr, "\n\n");
    exit(EXIT_FAILURE);
}
void terminate(const char msg[]) {
    fprintf(stderr, msg);
    exit(EXIT_FAILURE);
}

// Main function coordinates the program
int main(int argc, char *argv[]) {
    int width, height;
    parse_args(argc, argv, &width, &height);
    int numElements = width*height;
    int size = numElements*sizeof(rgb_pixel_t);

    rgb_pixel_t *h_A = (rgb_pixel_t *)malloc(size);
    rgb_pixel_t *d_A = NULL;
    cudaError_t err = cudaSuccess;
    bmpfile_t *bmp;
    bmp = bmp_create(width, height, 32);
    int xoffset = -(width - 1) /2;
    int yoffset = (height -1) / 2;

    // Error check h_A
    if (h_A == NULL)
        terminate("Failed to allocate host vectors\n");

    // Initialise h_A
    for (int i = 0; i < numElements; ++i)
        h_A[i] = {0,0,0,0};
    
    // Allocate cuda memory for A
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess)
        terminate("Failed to allocate device vector A\n", 
        cudaGetErrorString(err));

    // Copy h_A to d_A
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        terminate("Failed to copy vector A from host to device\n", 
        cudaGetErrorString(err));
    
    // Perform CUDA kernel call
    int threadsPerBlock = width;
    int blocksPerGrid = (numElements+threadsPerBlock-1)/threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", 
    blocksPerGrid, threadsPerBlock);
    Mandelbrot<<<blocksPerGrid, threadsPerBlock>>>(d_A, width, height, 
    xoffset, yoffset);

    // Error check
    err = cudaGetLastError();
    if (err != cudaSuccess)
        terminate("Failed to launch vectorAdd kernel\n", 
        cudaGetErrorString(err));

    // Copy d_A to h_A
    err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        terminate("Failed to copy A from device to host\n", 
        cudaGetErrorString(err));

    // Set the bmp file to returned pixel objects
    for (int col = 0; col < width; col++)
        for (int row = 0; row < height; row++)
            bmp_set_pixel(bmp, col, row, h_A[row*width+col]);

    // Save file and unload it
    bmp_save(bmp, FILENAME);
    bmp_destroy(bmp);

    free(h_A);

    err = cudaFree(d_A);
    if (err != cudaSuccess)
        terminate("Failed to free device vector A\n", 
        cudaGetErrorString(err));

    return 0;
}