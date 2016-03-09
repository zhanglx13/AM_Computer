/**
 * Test of an image processing
 *
 * This sample 
 */

#include <iostream>
#include <stdio.h>

#include <cuda_runtime.h>
#include <helper_cuda.h> /* for function checkCudaErrors() */

// defines, image size
#define IMAGE_SIZE     (1<<6)      //64

// define, kernel dimension
#define BLOCK_SIZE      (1<<8)      //256

// define, experiment
#define MEMORY_ITERATIONS   100


// enums
//enum pixelSize { BYTE, BYTE2, BYTE4, BYTE8 };

__global__ void 
imageTest(unsigned char *d_idata, unsigned char *d_odata, unsigned int nr_pixels)
{
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < nr_pixels){
        d_odata[tid] = (unsigned char)(d_idata[tid]+1);
    }
}

//void processImageGPU(unsigned char *h_idata, )


int main(int argc, char **argv)
{
    unsigned char *h_idata, *h_odata;   // host input and output data
    unsigned char *d_idata, *d_odata;   // device input and output data
    unsigned int imageSize = sizeof(unsigned char)*IMAGE_SIZE*IMAGE_SIZE;   // data size in bytes
    unsigned int nr_pixels = imageSize/sizeof(unsigned char);    // number of pixels

    h_idata = (unsigned char*)malloc(imageSize);    
    h_odata = (unsigned char*)malloc(imageSize);

    checkCudaErrors(cudaMalloc((void **)&d_idata, imageSize));
    checkCudaErrors(cudaMalloc((void **)&d_odata, imageSize));
    
    // initialize host input data
    for(unsigned int i = 0 ; i < nr_pixels ; i ++){
        // the range for an unsigned char is [0, 255];
        h_idata[i] = (unsigned char)(i & 0xff);
    }


    dim3 blockSize(BLOCK_SIZE,1,1);
    dim3 gridSize(nr_pixels/BLOCK_SIZE,1,1);
    // copy data from host to device
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, imageSize, 
                               cudaMemcpyHostToDevice));
    // image processing on device
    imageTest<<<gridSize, blockSize>>>(d_idata, d_odata, nr_pixels);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, imageSize, 
                               cudaMemcpyDeviceToHost));

    // verify result
    for(unsigned int i = 0 ; i < nr_pixels ; i ++){
        printf("%d \t %d\n", h_idata[i], h_odata[i]);
    }


    // clean up memory on host and device
    free(h_idata);
    free(h_odata);
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));
    exit(EXIT_SUCCESS);
}
