/**
 * Comparison of image processing speed on CPU and GPU.
 * Image size is IMAGE_SIZE by IMAGE_SIZE pixels and 
 * each pixel is pixelsize bytes
 *
 * This sample 
 */

#define NVS


#include <iostream>
#include <stdio.h>

#include <cuda_runtime.h>
#include <helper_cuda.h> /* for function checkCudaErrors() */
#include <cuda.h>
#include <helper_functions.h> /* for StopWatchInterface*/

// define, experiment
#define MEMORY_ITERATIONS   10
#define START_SIZE          ( 1 << 10 )     //1 KB
#define END_SIZE_NVS        ( 1 << 29 )     //512 MB
#define END_SIZE_TITAN      ( 1 << 32 )     //4 GB

// define, kernel functions
#define KERNEL_DEF(SIZE, TYPE)                                                                                  \
    __global__ void imageTest_##SIZE(unsigned TYPE *d_idata, unsigned TYPE *d_odata, unsigned int nr_pixels)    \
    {                                                                                                           \
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;                                               \
        if (tid < nr_pixels)                                                                                    \
            d_odata[tid] = (unsigned TYPE)(d_idata[tid] + 1);                                                   \
    }                                                                                                           \

#define KERNEL_CALL(SIZE)                                                   \
    imageTest_##SIZE<<<gridSize, blockSize>>>(d_idata, d_odata, nr_pixels); \

#define ALLOCATE_AND_INIT(TYPE)                                                     \
    unsigned TYPE *h_idata, *d_idata, *d_odata;                                     \
    h_idata = (unsigned TYPT)malloc(memSize);                                       \
    checkCudaErrors(cudaMalloc((void**)&d_idata, memSize));                         \
    checkCudaErrors(cudaMalloc((void**)&d_idata, memSize));                         \
    for (unsigned int i = 0 ; i < nr_pixels ; i ++)                                 \
        h_idata[i] = (unsigned TYPE)(i);                                            \
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, memSize, cudaMemcpyHostToDevice))  \

// enums
enum {  BYTE = 1,   // 8-bit pixels, unsigned char
        BYTE2 = 2,  // 16-bit pixels, unsigned short
        BYTE4 = 4,  // 32-bit pixels, unsigned int
        BYTE8 = 8   // 64-bit pixels, unsigned long long
};

////////////////////////////////////////////////////////////////////////////////////////
// Function declarations, forward
////////////////////////////////////////////////////////////////////////////////////////
float imageProcessingGPU(unsigned int memSize, unsigned int pixelSize); 
float imageProcessingCPU(unsigned int memSize, unsigned int pixelSize); 


// define kernel functions for different pixel sizes
KERNEL_DEF(BYTE, char)
KERNEL_DEF(BYTE2, short)
KERNEL_DEF(BYTE4, int)
KERNEL_DEF(BYTE8, long long)


int main(int argc, char **argv)
{
    /* check byte size assumptions 
     * unsigned char = 1 byte
     * unsigned short = 2 bytes
     * unsigned int = 4 bytes
     * unsigned long long = 8 bytes
     */
    if (sizeof(unsigned char) != 1){
        printf("unsigned char size != 1\n");
        exit(EXIT_FAILURE);
    }
    if (sizeof(unsigned short) != 2){
        printf("unsigned short size != 2\n");
        exit(EXIT_FAILURE);
    }
    if (sizeof(unsigned int) != 4){
        printf("unsigned int size != 4\n");
        exit(EXIT_FAILURE);
    }
    if (sizeof(unsigned long long) != 8){
        printf("unsigned long long size != 8\n");
        exit(EXIT_FAILURE);
    }

    int start = START_SIZE;
#ifdef NVS
    int end = END_SIZE_NVS;
#else
    int end = END_SIZE_TITAN;
#endif

    printf("%f\n", imageProcessingGPU(128, 4));

    exit(EXIT_SUCCESS);
}    

    

float imageProcessingGPU(unsigned int memSize, unsigned int pixelSize)
{
    StopWatchInterface *timer = NULL;
    float elapsedTimeInMs = 0.0f;
    
    unsigned int nr_pixels = memSize / pixelSize;
    // allocate host and device memory according to the pixel size
    switch(pixelSize){
        case 1:
            ALLOCATE_AND_INIT(char)
            break;
        case 2:
            ALLOCATE_AND_INIT(short)
            break;
        case 4:
            ALLOCATE_AND_INIT(int)
            break;
        case 8:
            ALLOCATE_AND_INIT(long long)
            break;
    }

    dim3 gridSize(1,1,1);
    dim3 blockSize(32,1,1);
    KERNEL_CALL(BYTE4)


    return 0.0f;
}


































