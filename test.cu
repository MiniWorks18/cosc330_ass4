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

__global__ void mandelbrot(float* devPtr, size_t pitch, int width, int height) {
    int thread_id = threadIdx.x;
    int block_dim = blockDim.x;
    int block_id = blockIdx.x;
    int grid_dim = gridDim.x;
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Thread_id: %d, Block_dim: %d, Block_id: %d, Grid_dim: %d, Id: %d\n", 
    thread_id, block_dim, block_id, grid_dim, id);

    for (int r = 0; r < height; ++r) {
        /*Tricky pointer arithmetic to get the pointer to the row*/  
        float* row = (float*)((char*)devPtr + r * pitch); 
        for (int c = 0; c < width; ++c) { 
            float element = row[c]; 
            printf("%d\n", element);
        } 
    }
}

__global__ void MyKernel(float *A, const int width, const int height, bmpfile_t* bmp) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    // int thread_id = threadIdx.x;
    // int block_dim = blockDim.x;
    // int block_id = blockIdx.x;
    // int grid_dim = gridDim.x;
    int xoffset = -(width - 1) /2;
    int yoffset = (height -1) / 2;
    int col = threadIdx.x;
    int row = blockIdx.x;

    // for (int col = 0; col < width; col++) {
    //     for (int row = 0; row < height; row++) {

    
    float x = XCENTER + (xoffset + col) / RESOLUTION;
    float y = YCENTER + (yoffset - row) / RESOLUTION;

    //Mandelbrot stuff
    float a = 0;
    float b = 0;
    float aold = 0;
    float bold = 0;
    float zmagsqr = 0;
    int iter = 0;
    float x_col = 0;
    while(iter < MAX_ITER && zmagsqr <= 4.0){
        ++iter;
	    a = (aold * aold) - (bold * bold) + x;
        b = 2.0 * aold*bold + y;

        zmagsqr = a*a + b*b;

        aold = a;
        bold = b;	
        // printf("iter: %d\n", iter);

    }
    x_col =  (240 - (( ((float) iter / ((float) 1000) * 230))));
    atomicAdd(&A[col*row], x_col);
    /////// Up to here, need to communicate the result from each thread back to the host and display the image
    // printf("ID (%d, %d): %d = %d\n", col, row, id, iter);
    //     }
    // }
}

void GroundColorMix(float* color, float x, float min, float max)
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

void parse_args(int argc, char *argv[], int *width, int *height) {
    if (argc != 3 || (*width = atoi(argv[1])) == 0 
        || (*height = atoi(argv[2])) == 0) {
        printf("Usage: %s <width> <height>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    int width, height;
    parse_args(argc, argv, &width, &height);
    int numElements = width*height;
    float *devPtr;
    size_t pitch;
    cudaMallocPitch(&devPtr, &pitch, width*sizeof(float), height);


    cudaError_t err = cudaSuccess;


    bmpfile_t *bmp;
    rgb_pixel_t pixel = {0, 0, 0, 0};
    int xoffset = -(width - 1) /2;
    int yoffset = (height -1) / 2;
    bmp = bmp_create(width, height, 32);
    int col = 0;
    int row = 0;
    int size = width*height*sizeof(float);



    float *h_A = (float *)malloc(size);

    if (h_A == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

     for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = 0;
    }

    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    
    int threadsPerBlock = width;
    int blocksPerGrid = (numElements+threadsPerBlock-1)/threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // mandelbrot<<<blocksPerGrid, threadsPerBlock>>>(d_A, pitch, width, height);

    MyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, width, height, bmp);
//     for(col = 0; col < width; col++){
//      for(row = 0; row < height; row++){
        
//         //Determine where in the mandelbrot set, the pixel is referencing
//         float x = XCENTER + (xoffset + col) / RESOLUTION;
//         float y = YCENTER + (yoffset - row) / RESOLUTION;

//         //Mandelbrot stuff

//         float a = 0;
//         float b = 0;
//         float aold = 0;
//         float bold = 0;
//         float zmagsqr = 0;
//         int iter =0;
// 	    float x_col;
//         float color[3];
//         //Check if the x,y coord are part of the mendelbrot set - refer to the algorithm
//         while(iter < MAX_ITER && zmagsqr <= 4.0){
//            ++iter;
// 	        a = (aold * aold) - (bold * bold) + x;
//            b = 2.0 * aold*bold + y;

//            zmagsqr = a*a + b*b;

//            aold = a;
//            bold = b;	

//         }
        
//         /* Generate the colour of the pixel from the iter value */
//         /* You can mess around with the colour settings to use different gradients */
//         /* Colour currently maps from royal blue to red */ 
//         x_col =  (COLOUR_MAX - (( ((float) iter / ((float) MAX_ITER) * GRADIENT_COLOUR_MAX))));
//         h_A[col*row] = x_col;
//         GroundColorMix(color, x_col, 1, COLOUR_DEPTH);
//         pixel.red = color[0];
//         pixel.green = color[1];
// 	    pixel.blue = color[2];
//         bmp_set_pixel(bmp, col, row, pixel);

//      }


//   }
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }



    err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy A from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }



    float color[3];
    for (int col = 0; col < width; col++) {
        for (int row = 0; row < height; row++) {
            int i = row*col;
            // printf("El %d = %f\n", i, h_A[i]);
            GroundColorMix(color, h_A[i], 1, COLOUR_DEPTH);
            pixel.red=color[0];
            pixel.green=color[1];
            pixel.blue=color[2];
            bmp_set_pixel(bmp, col, row, pixel);
        }
    }

    bmp_save(bmp, FILENAME);
    bmp_destroy(bmp);

   

    free(h_A);
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }




    return 0;
}