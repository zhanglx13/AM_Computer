/*
 * This is a simple test program to measure the memcopy bandwidth of the GPU.
 * It can measure device to device copy bandwidth, host to device copy bandwidth
 * for pageable and pinned memory, and device to host copy bandwidth for pageable
 * and pinned memory.
 *
 * Usage:
 * ./bandwidthTest [option]...
 */
/* A little information about write combined memory
 *  By default page-locked host memory is allocated as cacheable. 
 *  It can optionally be allocated as write-combining instead by 
 *  passing flag cudaHostAllocWriteCombined to cudaHostAlloc(). 
 *  Write-combining memory frees up the host's L1 and L2 cache 
 *  resources, making more cache available to the rest of the 
 *  application. In addition, write-combining memory is not 
 *  snooped during transfers across the PCI Express bus, which 
 *  can improve transfer performance by up to 40%.
 *  Reading from write-combining memory from the host is prohibitively 
 *  slow, so write-combining memory should in general be used for 
 *  memory that the host only writes to. 
 */

// CUDA runtime
#include <cuda_runtime.h>

//includes
#include <helper_functions.h>    // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>        // helper functions for CUDA error checking and initialization

#include <cuda.h>

#include <memory>
#include <iostream>
#include <cassert>

// defines, project
#define MEMORY_ITERATIONS   10
#define DEFAULT_SIZE        ( 32 * ( 1 << 20) )     //32 M
#define DEFAULT_INCREMENT   ( 1 << 22 )             //4 M
#define CACHE_CLEAR_SIZE    ( 1 << 24 )             //16 M

// defines, experiment
#define MEMSIZE_MAX         ( 1 << 26 )             //64 M
#define MEMSIZE_START       ( 1 << 10 )             //1 KB
#define INCREMENT_1KB       ( 1 << 10 )             //1 KB
#define INCREMENT_2KB       ( 1 << 11 )             //2 KB
#define INCREMENT_4KB       ( 1 << 12 )             //4 KB
#define INCREMENT_8KB       ( 1 << 13 )             //8 KB

// enums, project
enum memcpyKind { DEVICE_TO_HOST, HOST_TO_DEVICE, DEVICE_TO_DEVICE };
enum printMode  { USER_READABLE, CSV };
enum memoryMode { PINNED, PAGEABLE };

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
    printf("%s\n", (iRetVal == 0)? "Result = PASS" : "Result = FAIL");

    exit((iRetVal == 0)? EXIT_SUCCESS : EXIT_FAILURE);
}


///////////////////////////////////////////////////////////////////////////////
//Parse args, run the appropriate tests
///////////////////////////////////////////////////////////////////////////////
int runTest(const int argc, const char **argv)
{
    int start = DEFAULT_SIZE;
    int end = DEFAULT_SIZE;
    int increment = DEFAULT_INCREMENT;
    bool htod = false;
    bool dtoh = false;
    bool dtod = false;
    bool wc = false;
    printMode printmode = USER_READABLE;
    memoryMode memMode = PINNED;
    char * memModeStr = NULL;


    // parse command line arguments
    if (checkCmdLineFlag(argc, argv, "help")){
        printHelp();
        return 0;
    }
    // print result format
    if (checkCmdLineFlag(argc, argv, "csv"))
        printmode = CSV;
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

    // we will run on device 0 using rang mode

    if (checkCmdLineFlag(argc, argv, "htod")) htod = true;
    if (checkCmdLineFlag(argc, argv, "dtoh")) dtoh = true;
    if (checkCmdLineFlag(argc, argv, "dtod")) dtod = true;
    if (checkCmdLineFlag(argc, argv, "wc")) wc = true;
    if (checkCmdLineFlag(argc, argv, "cputiming")) bDontUseGPUTiming = true;
    if (!htod && !dtoh && !dtod){ // default
        htod = true;
        dtoh = true;
        dtod = true;
    }

    // for rang mode
    if (checkCmdLineFlag(argc, (const char **)argv, "start"))
        start = getCmdLineArgumentInt(argc, argv, "start");
    
    if (checkCmdLineFlag(argc, (const char **)argv, "end"))
        end = getCmdLineArgumentInt(argc, argv, "end");

    if (checkCmdLineFlag(argc, (const char **)argv, "increment"))
        increment = getCmdLineArgumentInt(argc, argv, "increment");

    
    if (htod)
        testBandwidthRange((unsigned int)start, (unsigned int)end, (unsigned int)increment,
                            HOST_TO_DEVICE, printmode, memMode, wc);
    if (dtod)
        testBandwidthRange((unsigned int)start, (unsigned int)end, (unsigned int)increment,
                            DEVICE_TO_DEVICE, printmode, memMode, wc);
    if (dtoh)
        testBandwidthRange((unsigned int)start, (unsigned int)end, (unsigned int)increment,
                            DEVICE_TO_HOST, printmode, memMode, wc);

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
    unsigned int count = (end - start)/increment + 1;
    unsigned int *memSizes = (unsigned int *)malloc(sizeof(unsigned int)*count);
    double *bandwidths = (double*)malloc(sizeof(double)*count);
    // initialize the bandwidths
    for (unsigned int i = 0 ; i < count ; i ++)
        bandwidths[i] = 0.0;

    // run each of the copy
    for (unsigned int i = 0 ; i < count ; i ++){
        memSizes[i] = start + i * increment;

        switch(kind){
            case DEVICE_TO_HOST:
                bandwidths[i] += testDeviceToHostTransfer(memSizes[i], memMode, wc);
                break;
            case HOST_TO_DEVICE:
                bandwidths[i] += testHostToDeviceTransfer(memSizes[i], memMode, wc);
                break;
            case DEVICE_TO_DEVICE:
                bandwidths[i] += testDeviceToDeviceTransfer(memSizes[i]);
                break;
        }
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
    if (kind == DEVICE_TO_DEVICE)
        sConfig += "D2D";
    else
    {
        if (kind == DEVICE_TO_HOST)
            sConfig += "D2H";
        else if (kind == HOST_TO_DEVICE)
            sConfig += "H2D";

        if (memMode == PAGEABLE)
            sConfig += "-Paged";
        else if (memMode == PINNED)
        {
            sConfig += "-Pinned";

            if (wc)
                sConfig += "WriteCombined";
        }
    }
    unsigned int i;
    double dSeconds = 0.0;

    for (i = 0 ; i < count ; i ++){
        dSeconds = (double)memSizes[i] / (bandwidths[i] * (double)(1<<20));
        printf("bandwidthTest-%s, Bandwidth = %.1f MB/s, Time = %.5f s, Size = %u bytes\n",
                sConfig.c_str(), bandwidths[i], dSeconds, memSizes[i]);
    }
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
