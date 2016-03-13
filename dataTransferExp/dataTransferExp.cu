/*
 * Running experiments based on the basic bandwidth test program.
 * Data transfer of D2H and H2D is tested for data size doubled from 
 * 1KB (1<<10) to 512MB (1<<29). (For nvs 5200m)
 * The experiments test the bandwidth for pageable, pinned without wc, 
 * and pinned with wc memory modes.
 * Transfer time can be calculated later with the memSize and bandwidth.
 */

#define NVS


// CUDA runtime
#include <cuda_runtime.h>

//includes
#include <helper_functions.h>    // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>        // helper functions for CUDA error checking and initialization

#include <cuda.h>

#include <memory>
#include <iostream>
#include <cassert>
#include <math.h> // for log2 function

// defines, project
#define MEMORY_ITERATIONS   10
#define START_SIZE          ( 1 << 10 )     //1 KB
#define END_SIZE_NVS        ( 1 << 29 )     //512 M (This is half the device memory of nvs 5200m)
#define END_SIZE_TITAN      ( 1 << 32 )     //4 GB (This is for TITAN, whose global memory is 6 GB)

// enums, project
enum memcpyKind { DEVICE_TO_HOST, HOST_TO_DEVICE, DEVICE_TO_DEVICE };
enum printMode  { USER_READABLE, CSV };
enum memoryMode { PINNED, PAGEABLE };
enum expMode    { BW, TT }; // run bandwidth(BW) mode or transfer time(TT) mode

const char *sMemoryCopyKind[] =
{
    "Device to Host",
    "Host to Device",
    "Device to Device",
    NULL
};

const char *sMemoryMode[] =
{
    "PINNED",
    "PAGEABLE",
    NULL
};


// if true, use CPU based timing for everything
static bool bDontUseGPUTiming;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
int runTest(const int argc, const char **argv);
void testBandwidthRange(unsigned int start, unsigned int end, unsigned int increment,
                        memcpyKind kind, printMode printmode, memoryMode memMode, bool wc);
float testDeviceToHostTransfer(unsigned int memSize, memoryMode memMode, bool wc);
float testHostToDeviceTransfer(unsigned int memSize, memoryMode memMode, bool wc);
float testDeviceToDeviceTransfer(unsigned int);
void printResultsReadable(unsigned int *memSizes, double *bandwidths, unsigned int count, memcpyKind kind, memoryMode memMode, bool wc);
void printResultsCSV(unsigned int *memSizes, double *bandwidths, unsigned int count, memcpyKind kind, memoryMode memMode, bool wc);
void printHelp(void);
////////////////////////////////////////////////////////////////////////////////


int main(int argc, char **argv)
{
    int iRetVal = runTest(argc, (const char **)argv);
    if (iRetVal < 0){
        checkCudaErrors(cudaSetDevice(0));
        cudaDeviceReset();
    }
    
    // finish
//    printf("%s\n", (iRetVal == 0)? "Result = PASS" : "Result = FAIL");

    exit((iRetVal == 0)? EXIT_SUCCESS : EXIT_FAILURE);
}


///////////////////////////////////////////////////////////////////////////////
//Parse args, run the appropriate tests
///////////////////////////////////////////////////////////////////////////////
int runTest(const int argc, const char **argv)
{
    int start = START_SIZE;
#ifdef NVS
    int end = END_SIZE_NVS;
#else
    int end = END_SIZE_TITAN; 
    /* For 4GB value, if the integer is 4 bytes, it is ok when it is cast to unsigned int */
#endif
    int increment = 0;
    bool htod = false;
    bool dtoh = false;
    bool wc = false;
    printMode printmode = CSV;
    memoryMode memMode = PINNED;
    expMode expmode = BW;
    char *memModeStr = NULL;
    char *expModeStr = NULL;

    // parse command line arguments
    if (checkCmdLineFlag(argc, argv, "help")){
        printHelp();
        return 0;
    }

    if (checkCmdLineFlag(argc, argv, "dtoh"))
        dtoh = true;

    if (checkCmdLineFlag(argc, argv, "htod"))
        htod = true;

    if (!htod && !dtoh){
        htod = true;
        dtoh = true;
    }
    // print result format
//    if (checkCmdLineFlag(argc, argv, "csv"))
//        printmode = CSV;
    // memory mode on host
    if (getCmdLineArgumentString(argc, argv, "memory", &memModeStr)){
        if (strcmp(memModeStr, "pageable") == 0)
            memMode = PAGEABLE;
        else if (strcmp(memModeStr, "pinned") == 0)
            memMode = PINNED;
        else{
            printf("Invalid memory mode\n");
            return -1000;
        }
    }
    else
        memMode = PINNED; // default memMode

    // experiment mode
    if (getCmdLineArgumentString(argc, argv, "exp", &expModeStr)){
        if (strcmp(expModeStr, "bw") == 0)
            expmode = BW;
        else if (strcmp(expModeStr, "tt") == 0)
            expmode = TT;
        else{
            printf("Invalid exp mode\n");
            return -1000;
        }
    }
    else
        expmode = BW; // default memMode
    
    // wc or not
    if (checkCmdLineFlag(argc, argv, "wc")) wc = true;
    if (checkCmdLineFlag(argc, argv, "cputiming")) bDontUseGPUTiming = true;

    // exp mode
    if (expmode == BW){ // bandwidth mode
        if (htod)
            testBandwidthRange((unsigned int)start, (unsigned int)end, (unsigned int)increment,
                                HOST_TO_DEVICE, printmode, memMode, wc);
        if (dtoh)
            testBandwidthRange((unsigned int)start, (unsigned int)end, (unsigned int)increment,
                                DEVICE_TO_HOST, printmode, memMode, wc);
    }


    // reset CUDA devices
    cudaSetDevice(0);
    cudaDeviceReset();
    
    return 0;
}

///////////////////////////////////////////////////////////////////////
//  Run a range mode bandwidth test
//////////////////////////////////////////////////////////////////////
void testBandwidthRange(unsigned int start, unsigned int end, unsigned int increment,
                        memcpyKind kind, printMode printmode, memoryMode memMode, bool wc)
{
    // count the number of copies we're going to run
    unsigned int count = log2((float)end) - log2((float)start) + 1;
    unsigned int *memSizes = (unsigned int *)malloc(sizeof(unsigned int)*count);
    double *bandwidths = (double*)malloc(sizeof(double)*count);
    // initialize the bandwidths
    for (unsigned int i = 0 ; i < count ; i ++)
        bandwidths[i] = 0.0;

    // run each of the copy
    unsigned int i = 0;
    for ( unsigned int memSize = start ; memSize <= end ; memSize <<= 1){
        memSizes[i] = memSize;
        switch(kind){
            case DEVICE_TO_HOST:
                bandwidths[i] += testDeviceToHostTransfer(memSizes[i], memMode, wc);
                break;
            case HOST_TO_DEVICE:
                bandwidths[i] += testHostToDeviceTransfer(memSizes[i], memMode, wc);
                break;
        }
        i ++;
    }
    if ( CSV == printmode)
        printResultsCSV(memSizes, bandwidths, count, kind, memMode, wc);
    else 
        printResultsReadable(memSizes, bandwidths, count, kind, memMode, wc);

    // clean up
    free(memSizes);
    free(bandwidths);
}


///////////////////////////////////////////////////////////////////////////////
//  test the bandwidth of a device to host memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
float testDeviceToHostTransfer(unsigned int memSize, memoryMode memMode, bool wc)
{
    StopWatchInterface *timer = NULL;
    float elapsedTimeInMs = 0.0f;
    float bandwidthInMBs = 0.0f;
    unsigned char *h_idata, *h_odata;
    cudaEvent_t start, stop;

    sdkCreateTimer(&timer);
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    if ( PINNED == memMode ){ // pinned memory allocation
        checkCudaErrors(cudaHostAlloc((void**)&h_idata, memSize, (wc)? cudaHostAllocWriteCombined : 0));
        checkCudaErrors(cudaHostAlloc((void**)&h_odata, memSize, (wc)? cudaHostAllocWriteCombined : 0));
    }
    else{ // pageable memory allocation
        h_idata = (unsigned char*)malloc(memSize);
        h_odata = (unsigned char*)malloc(memSize);
    }

    if (h_idata == 0 || h_odata == 0){
        fprintf(stderr, "Not enough memory space to run the test!!\n");
        exit(EXIT_FAILURE);
    }
    
    // device memory allocation
    unsigned char *d_idata;
    checkCudaErrors(cudaMalloc((void**)&d_idata, memSize));

    // initialize memory on host
    for (unsigned int i = 0 ; i < memSize/sizeof(unsigned char) ; i ++)
        h_idata[i] = (unsigned char)(i & 0xff);

    // copy data from host to device
    if ( PINNED == memMode )
        checkCudaErrors(cudaMemcpyAsync(d_idata, h_idata, memSize, 
                                        cudaMemcpyHostToDevice, 0));
    else
        checkCudaErrors(cudaMemcpy(d_idata, h_idata, memSize, 
                                   cudaMemcpyHostToDevice));

    // copy data from device to host
    sdkStartTimer(&timer);
    checkCudaErrors(cudaEventRecord(start, 0));

    if ( PINNED == memMode )
        for ( unsigned int i = 0 ; i < MEMORY_ITERATIONS ; i ++)
            checkCudaErrors(cudaMemcpyAsync(h_odata, d_idata, memSize, 
                                            cudaMemcpyDeviceToHost, 0));
    else
        for ( unsigned int i = 0 ; i < MEMORY_ITERATIONS; i ++)
            checkCudaErrors(cudaMemcpy(h_odata, d_idata, memSize, 
                                       cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaEventRecord(stop, 0));
    // sync device with host
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);

    // calculate the elapsed time
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    if ( bDontUseGPUTiming || memMode == PAGEABLE)
        elapsedTimeInMs = sdkGetTimerValue(&timer);

    // calculate the bandwidth
    bandwidthInMBs = ((float)(1<<10) * memSize * (float)MEMORY_ITERATIONS) / (elapsedTimeInMs * (float)(1<<20));
        
    // clean up memory
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    sdkDeleteTimer(&timer);

    if (PINNED == memMode){
        checkCudaErrors(cudaFreeHost(h_idata));
        checkCudaErrors(cudaFreeHost(h_odata));
    }
    else 
        free(h_idata);

    checkCudaErrors(cudaFree(d_idata));

    return bandwidthInMBs;

    

}

///////////////////////////////////////////////////////////////////////////////
//! test the bandwidth of a host to device memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
float testHostToDeviceTransfer(unsigned int memSize, memoryMode memMode, bool wc)
{
    StopWatchInterface *timer = NULL;
    float elapsedTimeInMs = 0.0f;
    float bandwidthInMBs = 0.0f;
    cudaEvent_t start, stop;

    sdkCreateTimer(&timer);
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // allocate host memory
    unsigned char *h_idata = NULL;

    if (memMode == PINNED)
        checkCudaErrors(cudaHostAlloc((void**)&h_idata, memSize, (wc) ? cudaHostAllocWriteCombined : 0));
    else{
        h_idata = (unsigned char*)malloc(memSize);
        if (h_idata == 0){
            fprintf(stderr, "Not enough memory on host to run test!\n");
            exit(EXIT_FAILURE);
        }
    }

    // initialize host memory
    for (unsigned int i = 0 ; i < memSize/sizeof(unsigned char) ; i ++)
        h_idata[i] = (unsigned char)(i & 0xff);

    // allocate device memory
    unsigned char *d_odata;
    checkCudaErrors(cudaMalloc((void**)&d_odata, memSize));

    sdkStartTimer(&timer);
    checkCudaErrors(cudaEventRecord(start, 0));

    // copy data from host to device
    if (PINNED == memMode)
        for (unsigned int i = 0 ; i < MEMORY_ITERATIONS ; i ++)
            checkCudaErrors(cudaMemcpyAsync(d_odata, h_idata, memSize,
                                            cudaMemcpyHostToDevice, 0));
    else 
        for (unsigned int i = 0 ; i < MEMORY_ITERATIONS ; i ++)
            checkCudaErrors(cudaMemcpy(d_odata, h_idata, memSize,
                                       cudaMemcpyHostToDevice));

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));
    
    if (PAGEABLE == memMode || bDontUseGPUTiming)
        elapsedTimeInMs = sdkGetTimerValue(&timer);

    sdkResetTimer(&timer);
    
    // calculate the bandwidth
    bandwidthInMBs = ((float)(1<<10) * memSize * (float)MEMORY_ITERATIONS) / ((float)(1<<20) * elapsedTimeInMs);

    // clean up memory
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    sdkDeleteTimer(&timer);

    if (PINNED == memMode)
        checkCudaErrors(cudaFreeHost(h_idata));
    else 
        free(h_idata);

    checkCudaErrors(cudaFree(d_odata));

    return bandwidthInMBs;
}


///////////////////////////////////////////////////////////////////////////////
//! test the bandwidth of a device to device memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
float testDeviceToDeviceTransfer(unsigned int memSize)
{
    StopWatchInterface *timer = NULL;
    float elapsedTimeInMs = 0.0f;
    float bandwidthInMBs = 0.0f;
    cudaEvent_t start, stop;

    sdkCreateTimer(&timer);
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // allocate host memory
    unsigned char *h_idata = (unsigned char*)malloc(memSize);
    if( h_idata == 0){
        fprintf(stderr, "Not enough memory available on host to run test!\n");
        exit(EXIT_FAILURE);
    }

    // initialize the host memory
    for (unsigned int i = 0 ; i < memSize/sizeof(unsigned char) ; i ++)
        h_idata[i] = (unsigned char)(i & 0xff);

    // allocate device memory
    unsigned char *d_idata, *d_odata;
    checkCudaErrors(cudaMalloc((void**)&d_idata, memSize));
    checkCudaErrors(cudaMalloc((void**)&d_odata, memSize));

    // initialize input device memory
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, memSize,
                               cudaMemcpyHostToDevice));

    // run the memcpy
    sdkStartTimer(&timer);
    checkCudaErrors(cudaEventRecord(start, 0));

    for (unsigned int i = 0 ; i < MEMORY_ITERATIONS ; i ++)
        checkCudaErrors(cudaMemcpy(d_odata, d_idata, memSize,
                                   cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaEventRecord(stop, 0));
    
    // since device to device memory copies are non-blocking,
    // cudaDeviceSynchronize() is required in order to get
    // proper timing
    checkCudaErrors(cudaDeviceSynchronize());

    // get the total elapsed time in ms
    sdkStopTimer(&timer);
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    if(bDontUseGPUTiming)
        elapsedTimeInMs = sdkGetTimerValue(&timer);

    // calculate the bandwidth in MB/s
	/* Why 2.0f here? */
    bandwidthInMBs = 2.0f * ((float)(1<<10) * memSize * (float)MEMORY_ITERATIONS) /
                    (elapsedTimeInMs * (float)(1<<20));

    // clean up memory
    sdkDeleteTimer(&timer);
    free(h_idata);
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    return bandwidthInMBs;

}

/////////////////////////////////////////////////////////
//print results in an easily read format
////////////////////////////////////////////////////////
void printResultsReadable(unsigned int *memSizes, double *bandwidths, unsigned int count, memcpyKind kind, memoryMode memMode, bool wc)
{
    printf(" %s Bandwidth\n", sMemoryCopyKind[kind]);
    printf(" %s Memory Transfers\n", sMemoryMode[memMode]);

    if (wc)
        printf(" Write-Combined Memory Writes are Enabled");

    printf("   Transfer Size (Bytes)\tBandwidth(MB/s)\n");
    unsigned int i;

    for (i = 0; i < (count - 1); i++)
    {
        printf("   %u\t\t\t%s%.1f\n", memSizes[i], (memSizes[i] < 10000)? "\t" : "", bandwidths[i]);
    }

    printf("   %u\t\t\t%s%.1f\n\n", memSizes[i], (memSizes[i] < 10000)? "\t" : "", bandwidths[i]);
}


///////////////////////////////////////////////////////////////////////////
//print results in a database format
///////////////////////////////////////////////////////////////////////////
void printResultsCSV(unsigned int *memSizes, double *bandwidths, unsigned int count, memcpyKind kind, memoryMode memMode, bool wc)
{
    std::string sConfig;

    // log config information
    if (kind == DEVICE_TO_HOST)
        sConfig += "D2H";
    else if (kind == HOST_TO_DEVICE)
        sConfig += "H2D";

    printf("%s:\t", sConfig.c_str());
    for ( unsigned int i = 0 ; i < count ; i ++)
        printf("%.1f\t", bandwidths[i]);
    printf("\n");
}

///////////////////////////////////////////////////////////////////////////
//Print help screen
///////////////////////////////////////////////////////////////////////////
void printHelp(void)
{
    printf("Usage:  bandwidthTest [OPTION]...\n");
    printf("Test the bandwidth for device to host, host to device, and device to device transfers\n");
    printf("\n");
    printf("Example:  measure the bandwidth of device to host pinned memory copies in the range 1024 Bytes to 102400 Bytes in 1024 Byte increments     \n");
    printf("./bandwidthTest --memory=pinned --start=1024 --end=102400 --increment=1024 --dtoh\n");

    printf("\n");
    printf("Options:\n");
    printf("--help\tDisplay this help menu\n");
    printf("--csv\tPrint results as a CSV\n");
    /* We use device 0 */
//    printf("--device=[deviceno]\tSpecify the device device to be used\n");
//    printf("  all - compute cumulative bandwidth on all the devices\n");
//    printf("  0,1,2,...,n - Specify any particular device to be used\n");
    printf("--memory=[MEMMODE]\tSpecify which memory mode to use\n");
    printf("  pageable - pageable memory\n");
    printf("  pinned   - non-pageable system memory\n");
    /* We use range*/
//    printf("--mode=[MODE]\tSpecify the mode to use\n");
//    printf("  quick - performs a quick measurement\n");
//    printf("  range - measures a user-specified range of values\n");
//    printf("  shmoo - performs an intense shmoo of a large range of values\n");
    printf("--htod\tMeasure host to device transfers\n");
    printf("--dtoh\tMeasure device to host transfers\n");
    printf("--dtod\tMeasure device to device transfers\n");
#if CUDART_VERSION >= 2020
    printf("--wc\tAllocate pinned memory as write-combined\n");
#endif
    printf("--cputiming\tForce CPU-based timing always\n");

    printf("Range mode options\n");
    printf("--start=[SIZE]\tStarting transfer size in bytes\n");
    printf("--end=[SIZE]\tEnding transfer size in bytes\n");
    printf("--increment=[SIZE]\tIncrement size in bytes\n");
}
