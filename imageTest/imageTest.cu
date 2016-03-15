/**
 * Comparison of image processing speed on CPU and GPU.
 * Image size is IMAGE_SIZE by IMAGE_SIZE pixels and 
 * each pixel is pixelsize bytes
 *
 * This sample 
 */

//#define NVS
#define TITAN


#include <iostream>
#include <stdio.h>

#include <cuda_runtime.h>
#include <helper_cuda.h> /* for function checkCudaErrors() */
#include <cuda.h>
#include <helper_functions.h> /* for StopWatchInterface*/

#include <math.h>   /* for log2 function */

// define, experiment
#define KERNEL_ITERATIONS   100
#define START_SIZE          ( 1 << 10 )     //1 KB
#define END_SIZE_NVS        ( 1 << 28 )     //256 MB    (1/4 of the device memory)
#define END_SIZE_TITAN      ( 0x80000000 )     //2 GB      (1/3 of the device memory)

#define BLOCK_SIZE          ( 1 << 7 )      //128 threads per block. This should be tested and tuned
#define GRID_X              ( 1 << 11 )     //2048
#define GRID_LIMIT          ( 1 << 16 - 1 ) //65535 for nvs (TITAN has a much larger limit: 2147483647)

// define, pixel operation
#define PIXEL_OP(TYPE, idata, odata)    odata = (unsigned TYPE)(idata + 1);  
// define, kernel functions
/* 
 * We first try to launch 1D block and 1D grid.
 * If the gridSize is bigger than the limit (65535), try to launch 2D grid
 */
#define KERNEL_DEF(SIZE, TYPE)                                                                                  \
    __global__ void imageTest_##SIZE(unsigned TYPE *d_idata, unsigned TYPE *d_odata, unsigned int nr_pixels)    \
    {                                                                                                           \
        unsigned int i = gridDim.x * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x;                        \
        if (i < nr_pixels)                                                                                      \
            d_odata[i] = (unsigned TYPE)(d_idata[i] + 1);                                                       \
    }                                                                                                           \

// define kernel functions for different pixel sizes
KERNEL_DEF(1, char)
KERNEL_DEF(2, short)
KERNEL_DEF(4, int)
KERNEL_DEF(8, long long)


// enums
enum {  BYTE = 1,   // 8-bit pixels, unsigned char
        BYTE2 = 2,  // 16-bit pixels, unsigned short
        BYTE4 = 4,  // 32-bit pixels, unsigned int
        BYTE8 = 8   // 64-bit pixels, unsigned long long
};

const char *sType[] = 
{
    "char",
    "short",
    "int",
    "long long",
    NULL
};

////////////////////////////////////////////////////////////////////////////////////////
// Function declarations, forward
////////////////////////////////////////////////////////////////////////////////////////
void printResults(double *GPUTime, double *CPUTime, unsigned int count);
// Declaration of imageProcessingGPU functions
/* cudaEventSynchronize(stop) should be called after recording stop
 * because kernel launching is asynchronous.
 */
#define IMAGEPROCESSINGGPU(TYPE, SIZE)                                          \
    float imageProcessingGPU_##SIZE(unsigned int memSize)                       \
    {                                                                           \
        float elapsedTimeInMs = 0.0f;                                           \
        cudaEvent_t start, stop;                                                \
        checkCudaErrors(cudaEventCreate(&start));                               \
        checkCudaErrors(cudaEventCreate(&stop));                                \
                                                                                \
        unsigned int nr_pixels = memSize / SIZE;                                \
        dim3 grid(1,1,1);                                                       \
        dim3 block(1,1,1);                                                      \
        block.x = (nr_pixels >= BLOCK_SIZE)? BLOCK_SIZE : nr_pixels;            \
        unsigned int nr_blocks = (nr_pixels + block.x - 1) / block.x;           \
        grid.x = (nr_blocks <= GRID_X)? nr_blocks : GRID_X;                     \
        grid.y = (nr_blocks + grid.x - 1) / grid.x;                             \
        /*printf("blocksize = %d\tgridx = %d\tgrid.y = %d\n", block.x, grid.x, grid.y);            */ \
                                                                                \
        unsigned TYPE *h_idata, *d_idata, *d_odata;                             \
        h_idata = (unsigned TYPE *)malloc(memSize);                             \
        checkCudaErrors(cudaMalloc((void**)&d_idata, memSize));                 \
        checkCudaErrors(cudaMalloc((void**)&d_odata, memSize));                 \
        for (unsigned int i = 0 ; i < nr_pixels ; i ++)                         \
            h_idata[i] = (unsigned TYPE)(i);                                    \
        checkCudaErrors(cudaMemcpy(d_idata, h_idata, memSize,                   \
                                   cudaMemcpyHostToDevice));                    \
                                                                                \
        checkCudaErrors(cudaEventRecord(start, 0));                             \
        imageTest_##SIZE<<<grid, block>>>(d_idata, d_odata, nr_pixels);         \
        checkCudaErrors(cudaEventRecord(stop, 0));                              \
        cudaEventSynchronize(stop);                                             \
        checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));   \
                                                                                \
        checkCudaErrors(cudaEventDestroy(start));                               \
        checkCudaErrors(cudaEventDestroy(stop));                                \
        free(h_idata);                                                          \
        checkCudaErrors(cudaFree(d_idata));                                     \
        checkCudaErrors(cudaFree(d_odata));                                     \
        return elapsedTimeInMs;                                                 \
}                                                                               \

IMAGEPROCESSINGGPU(char, 1)
IMAGEPROCESSINGGPU(short, 2)
IMAGEPROCESSINGGPU(int, 4)
IMAGEPROCESSINGGPU(long long, 8)

// Declaration of imageProcessingCPU functions
#define IMAGEPROCESSINGCPU(TYPE, SIZE)                                          \
    float imageProcessingCPU_##SIZE(unsigned int memSize)                       \
    {                                                                           \
        StopWatchInterface *timer = NULL;                                       \
        float elapsedTimeInMs = 0.0f;                                           \
        sdkCreateTimer(&timer);                                                 \
                                                                                \
        unsigned int nr_pixels = memSize / SIZE;                                \
        unsigned TYPE *h_idata, *h_odata;                                       \
        h_idata = (unsigned TYPE *)malloc(memSize);                             \
        h_odata = (unsigned TYPE *)malloc(memSize);                             \
                                                                                \
        for (unsigned int i = 0 ; i < nr_pixels ; i ++)                         \
            h_idata[i] = (unsigned TYPE)(i);                                    \
                                                                                \
        sdkStartTimer(&timer);                                                  \
        for (unsigned int i = 0 ; i < nr_pixels ; i ++)                         \
            PIXEL_OP(TYPE, h_idata[i], h_odata[i])                              \
        sdkStopTimer(&timer);                                                   \
        elapsedTimeInMs = sdkGetTimerValue(&timer);                             \
                                                                                \
        free(h_idata);                                                          \
        free(h_odata);                                                          \
        sdkDeleteTimer(&timer);                                                 \
        return elapsedTimeInMs;                                                 \
    }                                                                           \

IMAGEPROCESSINGCPU(char, 1)
IMAGEPROCESSINGCPU(short, 2)
IMAGEPROCESSINGCPU(int, 4)
IMAGEPROCESSINGCPU(long long, 8)


int main(const int argc, const char **argv)
{
    /* check byte size assumptions 
     * unsigned char = 1 byte
     * unsigned short = 2 bytes
     * unsigned int = 4 bytes
     * unsigned long long = 8 bytes
     */
    if (sizeof(unsigned char) != BYTE){
        printf("unsigned char size != 1\n");
        exit(EXIT_FAILURE);
    }
    if (sizeof(unsigned short) != BYTE2){
        printf("unsigned short size != 2\n");
        exit(EXIT_FAILURE);
    }
    if (sizeof(unsigned int) != BYTE4){
        printf("unsigned int size != 4\n");
        exit(EXIT_FAILURE);
    }
    if (sizeof(unsigned long long) != BYTE8){
        printf("unsigned long long size != 8\n");
        exit(EXIT_FAILURE);
    }

    unsigned long long start = START_SIZE;
#ifdef NVS
    unsigned long long end = END_SIZE_NVS;
#else
    unsigned long long end = END_SIZE_TITAN;
#endif
    
    unsigned int pixelSize = 0;
    if (checkCmdLineFlag(argc, argv, "char"))
        pixelSize = 1;
    if (checkCmdLineFlag(argc, argv, "short"))
        pixelSize = 2;
    if (checkCmdLineFlag(argc, argv, "int"))
        pixelSize = 4;
    if (checkCmdLineFlag(argc, argv, "long"))
        pixelSize = 8;
    // number of experiments
    unsigned int count = (unsigned int)(log2((float)end) - log2((float)start)) + 1;
    double *GPUTime = (double*)malloc(sizeof(double)*count);
    double *CPUTime = (double*)malloc(sizeof(double)*count);
    unsigned int i = 0;
    /* 
     * memSize should be as large as unsigned long long because 
     * end is set to 2 GB, at the last iteration, memSize will be
     * end * 2. If memSize is 32-bit, end * 2 will be 0, which is
     * smaller than end and the loop will never stop.
     */
    for (unsigned long long memSize = start ; memSize <= end ; memSize <<= 1){
        switch(pixelSize){
            case 1:
                GPUTime[i] = imageProcessingGPU_1((unsigned int)memSize);
                CPUTime[i] = imageProcessingCPU_1((unsigned int)memSize);
                break;
            case 2:
                GPUTime[i] = imageProcessingGPU_2((unsigned int)memSize);
                CPUTime[i] = imageProcessingCPU_2((unsigned int)memSize);
                break;
            case 4:
                GPUTime[i] = imageProcessingGPU_4((unsigned int)memSize);
                CPUTime[i] = imageProcessingCPU_4((unsigned int)memSize);
                break;
            case 8:
                GPUTime[i] = imageProcessingGPU_8((unsigned int)memSize);
                CPUTime[i] = imageProcessingCPU_8((unsigned int)memSize);
                break;
        }
        i ++;
    }

    printResults(GPUTime, CPUTime, count);

    // clean up memory
    free(GPUTime);
    free(CPUTime);

    exit(EXIT_SUCCESS);
}    

///////////////////////////////////////////////////////////////////////////
//print results in a database format
///////////////////////////////////////////////////////////////////////////
void printResults(double *GPUTime, double *CPUTime, unsigned int count)
{
    printf("GPUTime:");
    for (unsigned int i = 0 ; i < count ; i ++)
        printf("   %.5f", GPUTime[i]);
    printf("\n");

    printf("CPUTime:");
    for (unsigned int i = 0 ; i < count ; i ++)
        printf("   %.5f", CPUTime[i]);
    printf("\n");
}





















