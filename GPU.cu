#include <pthread.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "structs.h"
#include <stdio.h>
#include "kernel.h"
#include <math.h>
#include "GPU.h"
#include <algorithm>
#include "omp.h"
#include <queue>
#include <unistd.h>
#include <parallel/algorithm>

//thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h> //for streams for thrust (added with Thrust v1.8)


//for warming up GPU:
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>



using namespace std;


cudaMemLocation kCpuMemLocation = { cudaMemLocationTypeHost, 0 };


//Error checking GPU calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//sort descending
bool compareWorkArrayByNumPointsInCell(const workArray &a, const workArray &b)
{
    return a.pntsInCell > b.pntsInCell;
}


//sort descending
bool compareKeyValPairs(const keyValPair& a, const keyValPair& b)
{
	return a.key < b.key;
}


//sort ascending
bool compareByPointValue(const key_val_sort &a, const key_val_sort &b)
{
    return a.value_at_dim < b.value_at_dim;
}


unsigned long long callGPUBatchEst(unsigned int * DBSIZE, DTYPE* dev_database, DTYPE* dev_epsilon, struct grid * dev_grid, 
	unsigned int * dev_indexLookupArr, struct gridCellLookup * dev_gridCellLookupArr, DTYPE* dev_minArr, 
	unsigned int * dev_nCells, unsigned int * dev_nNonEmptyCells, unsigned int * dev_gridCellNDMask, 
	unsigned int * dev_gridCellNDMaskOffsets, unsigned int * dev_orderedQueryPntIDs, unsigned int * retNumBatches, unsigned int * retGPUBufferSize)
{



	//CUDA error code:
	cudaError_t errCode;

	printf("\n\n***********************************\nEstimating Batches:");
	cout<<"\n** BATCH ESTIMATOR: Sometimes the GPU will error on a previous execution and you won't know. \nLast error start of function: "<<cudaGetLastError();



//////////////////////////////////////////////////////////
	//ESTIMATE THE BUFFER SIZE AND NUMBER OF BATCHES ETC BY COUNTING THE NUMBER OF RESULTS
	//TAKE A SAMPLE OF THE DATA POINTS, NOT ALL OF THEM
	//Use sampleRate for this
	/////////////////////////////////////////////////////////

	
	// printf("\nDon't estimate: calculate the entire thing (for testing)");
	//Parameters for the batch size estimation.
	double sampleRate=SAMPLERATE; //sample 1.5% of the points in the dataset sampleRate=0.01. 
						//Sample the entire dataset(no sampling) sampleRate=1
						//0.015- used in Journal version of high-D paper for reorderqueries
						//0.01- used for all otherpapers (HPBDC, JPDC, DaMoN)
	int offsetRate=1.0/sampleRate;
	printf("\nOffset: %d", offsetRate);



	/////////////////
	//N GPU threads
	////////////////

	unsigned int * dev_N_batchEst; 
	dev_N_batchEst=(unsigned int*)malloc(sizeof(unsigned int));

	unsigned int * N_batchEst; 
	N_batchEst=(unsigned int*)malloc(sizeof(unsigned int));
	*N_batchEst=*DBSIZE*sampleRate;


	//allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_N_batchEst, sizeof(unsigned int)));
	
	//copy N to device 
	//N IS THE NUMBER OF THREADS
	gpuErrchk(cudaMemcpy( dev_N_batchEst, N_batchEst, sizeof(unsigned int), cudaMemcpyHostToDevice));
	
	/////////////
	//count the result set size 
	////////////

	unsigned int * dev_cnt_batchEst; 
	dev_cnt_batchEst=(unsigned int*)malloc(sizeof(unsigned int));

	unsigned int * cnt_batchEst; 
	cnt_batchEst=(unsigned int*)malloc(sizeof(unsigned int));
	*cnt_batchEst=0;


	//allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_cnt_batchEst, sizeof(unsigned int)));

	//copy cnt to device 
	gpuErrchk(cudaMemcpy( dev_cnt_batchEst, cnt_batchEst, sizeof(unsigned int), cudaMemcpyHostToDevice));

	//////////////////
	//SAMPLE OFFSET - TO SAMPLE THE DATA TO ESTIMATE THE TOTAL NUMBER OF KEY VALUE PAIRS
	/////////////////

	//offset into the database when batching the results
	unsigned int * sampleOffset; 
	sampleOffset=(unsigned int*)malloc(sizeof(unsigned int));
	*sampleOffset=offsetRate;


	unsigned int * dev_sampleOffset; 
	dev_sampleOffset=(unsigned int*)malloc(sizeof(unsigned int));

	//allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_sampleOffset, sizeof(unsigned int)));

	//copy offset to device 
	gpuErrchk(cudaMemcpy( dev_sampleOffset, sampleOffset, sizeof(unsigned int), cudaMemcpyHostToDevice));

	////////////////////////////////////
	//TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////			

	//debug values
	unsigned int * dev_debug1; 
	dev_debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug1=0;

	unsigned int * dev_debug2; 
	dev_debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug2=0;

	unsigned int * debug1; 
	debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug1=0;

	unsigned int * debug2; 
	debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug2=0;



	//allocate on the device
	gpuErrchk(cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int )));
	
	gpuErrchk(cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int )));
	
	//copy debug to device
	gpuErrchk(cudaMemcpy( dev_debug1, debug1, sizeof(unsigned int), cudaMemcpyHostToDevice ));
	

	gpuErrchk(cudaMemcpy( dev_debug2, debug2, sizeof(unsigned int), cudaMemcpyHostToDevice ));
	
	////////////////////////////////////
	//END TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////	




	const int TOTALBLOCKSBATCHEST=ceil((1.0*(*DBSIZE)*sampleRate)/(1.0*BLOCKSIZE));	
	printf("\ntotal blocks: %d",TOTALBLOCKSBATCHEST);

	

	kernelNDGridIndexBatchEstimator<<< TOTALBLOCKSBATCHEST, BLOCKSIZE>>>(dev_debug1, dev_debug2, dev_N_batchEst, 
		dev_sampleOffset, dev_database, dev_epsilon, dev_grid, dev_indexLookupArr, 
		dev_gridCellLookupArr, dev_minArr, dev_nCells, dev_cnt_batchEst, dev_nNonEmptyCells, dev_gridCellNDMask, 
		dev_gridCellNDMaskOffsets, dev_orderedQueryPntIDs);
		cout<<"\n** ERROR FROM KERNEL LAUNCH OF BATCH ESTIMATOR: "<<cudaGetLastError();
		// find the size of the number of results
		errCode=cudaMemcpy( cnt_batchEst, dev_cnt_batchEst, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		if(errCode != cudaSuccess) {
		cout << "\nError: getting cnt for batch estimate from GPU Got error with code " << errCode << endl; 
		}
		else
		{
			printf("\nGPU: result set size for estimating the number of batches (sampled): %u",*cnt_batchEst);
		}


	#if COUNTMETRICS==1	
	gpuErrchk(cudaMemcpy( debug1, dev_debug1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	printf("\nGPU: sampled number of cells visited: %u",*debug1);
	#endif	

	uint64_t estimatedNeighbors=(uint64_t)*cnt_batchEst*(uint64_t)offsetRate;	
	printf("\nFrom gpu cnt: %d, offset rate: %d", *cnt_batchEst,offsetRate);
	
	
	unsigned int GPUBufferSize=GPUBUFFERSIZE;

	#ifndef PYTHON	
	double alpha=0.05; //overestimation factor
	#endif

	#ifdef PYTHON
	double alpha=0.10;
	#endif
	
	uint64_t estimatedTotalSizeWithAlpha=estimatedNeighbors*(1.0+alpha*1.0);
	printf("\nEstimated total result set size: %lu", estimatedNeighbors);
	printf("\nEstimated total result set size (with Alpha %f): %lu", alpha,estimatedTotalSizeWithAlpha);	
	


	if (estimatedNeighbors<(GPUBufferSize*GPUSTREAMS))
	{
		printf("\nSmall buffer size, increasing alpha to: %f",alpha*3.0);
		GPUBufferSize=estimatedNeighbors*(1.0+(alpha*3.0))/(GPUSTREAMS);		//we do 3*alpha for small datasets because the
																		//sampling will be worse for small datasets
																		//but we fix the number of streams.			
	}

	unsigned int numBatches=ceil(((1.0+alpha)*estimatedNeighbors*1.0)/((uint64_t)GPUBufferSize*1.0));
	
	//Make sure at least MINBATCHES are executed to mitigate against large pinned memory allocation overheads
	if (numBatches<MINBATCHES)
	{
		GPUBufferSize=GPUBufferSize*((numBatches*1.0)/(MINBATCHES*1.0));
		numBatches=MINBATCHES;
	}

	printf("\nNumber of batches: %d, buffer size: %d", numBatches, GPUBufferSize);

	*retNumBatches=numBatches;
	*retGPUBufferSize=GPUBufferSize;
		

	printf("\nEnd Batch Estimator\n***********************************\n");




	cudaFree(dev_cnt_batchEst);	
	cudaFree(dev_N_batchEst);
	cudaFree(dev_sampleOffset);

return estimatedTotalSizeWithAlpha;

}

void distanceTableNDGridBatches(std::vector<std::vector<DTYPE> > * NDdataPoints, DTYPE* epsilon, struct grid * index, 
	struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, DTYPE* minArr, unsigned int * nCells, 
	unsigned int * indexLookupArr, struct neighborTableLookup * neighborTable, std::vector<struct neighborDataPtrs> * pointersToNeighbors, 
	uint64_t * totalNeighbors, unsigned int * gridCellNDMask, unsigned int * gridCellNDMaskOffsets, unsigned int * nNDMaskElems, CTYPE* workCounts,
	keyValPair ** keyValPairs, unordered_map<unsigned int, vector<struct keyValBin>> * keyBinsMap, 
	double * kernelExecutionTime, double * totalSortTime, double * tableConstructionTime )
{




	double tKernelResultsStart=omp_get_wtime();
	
	cout<<"\n** Sometimes the GPU will error on a previous execution and you won't know. \nLast error start of function: "<<cudaGetLastError();

	//CUDA error code:
	cudaError_t errCode;


	unsigned int * dev_orderedQueryPntIDs=NULL;
	//Reordering the query points based on work
	#if QUERYREORDER==1
	unsigned int * orderedQueryPntIDs=new unsigned int[NDdataPoints->size()];
	computeWorkDifficulty(orderedQueryPntIDs, gridCellLookupArr, nNonEmptyCells, indexLookupArr, index);
	//allocate memory on device:
	gpuErrchk(cudaMalloc( (void**)&dev_orderedQueryPntIDs, sizeof(unsigned int)*NDdataPoints->size()));
	gpuErrchk(cudaMemcpy(dev_orderedQueryPntIDs, orderedQueryPntIDs, sizeof(unsigned int)*NDdataPoints->size(), cudaMemcpyHostToDevice));

	#endif	

	///////////////////////////////////
	//COPY THE DATABASE TO THE GPU
	///////////////////////////////////
	unsigned int * DBSIZE;
	DBSIZE=(unsigned int*)malloc(sizeof(unsigned int));
	*DBSIZE=NDdataPoints->size();
	
	printf("\nIn main GPU method: DBSIZE is: %u",*DBSIZE);cout.flush();
	
	DTYPE* database= (DTYPE*)malloc(sizeof(DTYPE)*(*DBSIZE)*(GPUNUMDIM));  
	DTYPE* dev_database;
	
	//allocate memory on device:
	gpuErrchk(cudaMalloc( (void**)&dev_database, sizeof(DTYPE)*(GPUNUMDIM)*(*DBSIZE)));

	//copy the database from the ND vector to the array:
	for (int i=0; i<(*DBSIZE); i++){
		std::copy((*NDdataPoints)[i].begin(), (*NDdataPoints)[i].end(), database+(i*(GPUNUMDIM)));
	}


	//copy database to the device
	gpuErrchk(cudaMemcpy(dev_database, database, sizeof(DTYPE)*(GPUNUMDIM)*(*DBSIZE), cudaMemcpyHostToDevice));
	///////////////////////////////////
	//END COPY THE DATABASE TO THE GPU
	///////////////////////////////////




	///////////////////////////////////
	//COPY THE INDEX TO THE GPU
	///////////////////////////////////

	struct grid * dev_grid;
	//allocate memory on device:
	gpuErrchk(cudaMalloc( (void**)&dev_grid, sizeof(struct grid)*(*nNonEmptyCells)));
	
	//copy grid index to the device:
	gpuErrchk(cudaMemcpy(dev_grid, index, sizeof(struct grid)*(*nNonEmptyCells), cudaMemcpyHostToDevice));

	printf("\nSize of index sent to GPU (MiB): %f", (DTYPE)sizeof(struct grid)*(*nNonEmptyCells)/(1024.0*1024.0));


	///////////////////////////////////
	//END COPY THE INDEX TO THE GPU
	///////////////////////////////////


	///////////////////////////////////
	//COPY THE LOOKUP ARRAY TO THE DATA ELEMS TO THE GPU
	///////////////////////////////////

	#if SORT==1
	printf("\nSORTING ALL DIMENSIONS FOR VARIANCE NOW");
	printf("\nSORTIDU USES THE FIRST UNINXEDED DIMENSION");
	
	double tstartsort=omp_get_wtime();
	
	int sortDim=0;
	struct key_val_sort tmp;
	std::vector<struct key_val_sort> tmp_to_sort;
	unsigned int totalLength=0;

	if(GPUNUMDIM > NUMINDEXEDDIM)
		sortDim = NUMINDEXEDDIM;

	for (int i=0; i<(*nNonEmptyCells); i++)
	// for (int i=0; i<1; i++)
	{
		// if(index[i].indexmin < index[i].indexmax){
			// printf("Size cell: %d, %d\n", i,(index[i].indexmax-index[i].indexmin)+1);
			for (int j=0; j<(index[i].indexmax-index[i].indexmin)+1; j++)
			{
			unsigned int idx=index[i].indexmin+j;	
			tmp.pid=indexLookupArr[idx];
			tmp.value_at_dim=database[indexLookupArr[idx]*GPUNUMDIM+sortDim];
			tmp_to_sort.push_back(tmp);
			}

			
			totalLength+=tmp_to_sort.size();
			std::sort(tmp_to_sort.begin(),tmp_to_sort.end(),compareByPointValue); 

			//copy the sorted elements into the lookup array
			for (int x=0; x<tmp_to_sort.size(); x++)
			{
				indexLookupArr[index[i].indexmin+x]=tmp_to_sort[x].pid;
			}

			tmp_to_sort.clear();	
	}	

	double tendsort=omp_get_wtime();
	printf("\nSORT cells time (on host): %f", tendsort - tstartsort);

	// printf("\nTotal length: %u",totalLength);
	#endif
	
	unsigned int * dev_indexLookupArr;

	//allocate memory on device:
	gpuErrchk(cudaMalloc( (void**)&dev_indexLookupArr, sizeof(unsigned int)*(*DBSIZE)));

	//copy lookup array to the device:
	gpuErrchk(cudaMemcpy(dev_indexLookupArr, indexLookupArr, sizeof(unsigned int)*(*DBSIZE), cudaMemcpyHostToDevice));
	
	///////////////////////////////////
	//END COPY THE LOOKUP ARRAY TO THE DATA ELEMS TO THE GPU
	///////////////////////////////////



	///////////////////////////////////
	//COPY THE GRID CELL LOOKUP ARRAY 
	///////////////////////////////////

	
	
						
	struct gridCellLookup * dev_gridCellLookupArr;

	//allocate memory on device:
	gpuErrchk(cudaMalloc( (void**)&dev_gridCellLookupArr, sizeof(struct gridCellLookup)*(*nNonEmptyCells)));

	//copy lookup array to the device:
	gpuErrchk(cudaMemcpy(dev_gridCellLookupArr, gridCellLookupArr, sizeof(struct gridCellLookup)*(*nNonEmptyCells), cudaMemcpyHostToDevice));

	///////////////////////////////////
	//END COPY THE GRID CELL LOOKUP ARRAY 
	///////////////////////////////////
	
	///////////////////////////////////
	//COPY GRID DIMENSIONS TO THE GPU
	//THIS INCLUDES THE NUMBER OF CELLS IN EACH DIMENSION, 
	//AND THE STARTING POINT OF THE GRID IN THE DIMENSIONS 
	///////////////////////////////////

	//minimum boundary of the grid:
	DTYPE* dev_minArr;
	//Allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_minArr, sizeof(DTYPE)*(NUMINDEXEDDIM)));
	
	gpuErrchk(cudaMemcpy( dev_minArr, minArr, sizeof(DTYPE)*(NUMINDEXEDDIM), cudaMemcpyHostToDevice ));

	//number of cells in each dimension
	unsigned int * dev_nCells;

	//Allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_nCells, sizeof(unsigned int)*(NUMINDEXEDDIM)));

	gpuErrchk(cudaMemcpy( dev_nCells, nCells, sizeof(unsigned int)*(NUMINDEXEDDIM), cudaMemcpyHostToDevice ));

	///////////////////////////////////
	//END COPY GRID DIMENSIONS TO THE GPU
	///////////////////////////////////






	
	

	///////////////////////////////////
	//EPSILON
	///////////////////////////////////
	DTYPE* dev_epsilon;
	

	//Allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_epsilon, sizeof(DTYPE)));

	//copy to device
	gpuErrchk(cudaMemcpy( dev_epsilon, epsilon, sizeof(DTYPE), cudaMemcpyHostToDevice ));
	
	///////////////////////////////////
	//END EPSILON
	///////////////////////////////////


	///////////////////////////////////
	//NUMBER OF NON-EMPTY CELLS
	///////////////////////////////////
	unsigned int * dev_nNonEmptyCells;
	
	//Allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_nNonEmptyCells, sizeof(unsigned int)));
	//copy to device
	gpuErrchk(cudaMemcpy( dev_nNonEmptyCells, nNonEmptyCells, sizeof(unsigned int), cudaMemcpyHostToDevice ));

	///////////////////////////////////
	//NUMBER OF NON-EMPTY CELLS
	///////////////////////////////////
	


	//////////////////////////////////
	//ND MASK -- The array, the offsets, and the size of the array
	//////////////////////////////////

	//NDMASK
	unsigned int * dev_gridCellNDMask;

	//Allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_gridCellNDMask, sizeof(unsigned int)*(*nNDMaskElems)));
	
	gpuErrchk(cudaMemcpy( dev_gridCellNDMask, gridCellNDMask, sizeof(unsigned int)*(*nNDMaskElems), cudaMemcpyHostToDevice ));
	
	//NDMASKOFFSETS
	unsigned int * dev_gridCellNDMaskOffsets;

	//Allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_gridCellNDMaskOffsets, sizeof(unsigned int)*(2*NUMINDEXEDDIM)));

	gpuErrchk(cudaMemcpy( dev_gridCellNDMaskOffsets, gridCellNDMaskOffsets, sizeof(unsigned int)*(2*NUMINDEXEDDIM), cudaMemcpyHostToDevice ));

	//////////////////////////////////
	//End ND MASK -- The array, the offsets, and the size of the array
	//////////////////////////////////

	/*
	////////////////////////////////////
	//NUMBER OF THREADS PER GPU STREAM
	////////////////////////////////////

	//THE NUMBER OF THREADS THAT ARE LAUNCHED IN A SINGLE KERNEL INVOCATION
	//CAN BE FEWER THAN THE NUMBER OF ELEMENTS IN THE DATABASE IF MORE THAN 1 BATCH
	unsigned int * N;
	N=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	
	unsigned int * dev_N; 

	//allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_N, sizeof(unsigned int)*GPUSTREAMS));
	
	////////////////////////////////////
	//NUMBER OF THREADS PER GPU STREAM
	////////////////////////////////////


	////////////////////////////////////
	//OFFSET INTO THE DATABASE FOR BATCHING THE RESULTS
	//BATCH NUMBER 
	////////////////////////////////////
	unsigned int * batchOffset; 
	batchOffset=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	
	unsigned int * dev_offset; 
	

	//allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_offset, sizeof(unsigned int)*GPUSTREAMS));

	//Batch number to calculate the point to process (in conjunction with the offset)
	//offset into the database when batching the results
	unsigned int * batchNumber; 
	batchNumber=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	unsigned int * dev_batchNumber; 

	//allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_batchNumber, sizeof(unsigned int)*GPUSTREAMS));

	////////////////////////////////////
	//END OFFSET INTO THE DATABASE FOR BATCHING THE RESULTS
	//BATCH NUMBER
	////////////////////////////////////

	unsigned long long estimatedNeighbors=0;	
	unsigned int numBatches=0;
	unsigned int GPUBufferSize=0;

	double tstartbatchest=omp_get_wtime();
	estimatedNeighbors=callGPUBatchEst(DBSIZE, dev_database, dev_epsilon, dev_grid, dev_indexLookupArr,dev_gridCellLookupArr, dev_minArr, dev_nCells, dev_nNonEmptyCells, dev_gridCellNDMask,dev_gridCellNDMaskOffsets, dev_orderedQueryPntIDs, &numBatches, &GPUBufferSize);	
	double tendbatchest=omp_get_wtime();
	printf("\nTime to estimate batches: %f",tendbatchest - tstartbatchest);
	printf("\nIn Calling fn: Estimated neighbors: %llu, num. batches: %d, GPU Buffer size: %d",estimatedNeighbors, numBatches,GPUBufferSize);
	

	//initialize new neighbortable. resize to the number of batches	
	//Only use this if using unicomp
	#if STAMP==1
	double tstartinitneighbortable=omp_get_wtime();
	for (int i=0; i<NDdataPoints->size(); i++)
	{
	neighborTable[i].cntNDataArrays=0;
	neighborTable[i].vectindexmin.resize(numBatches);
	neighborTable[i].vectindexmax.resize(numBatches);
	neighborTable[i].vectdataPtr.resize(numBatches);
	pthread_mutex_init(&neighborTable[i].pointLock,NULL);
	}
	double tendinitneighbortable=omp_get_wtime();
	printf("\nTime to init neighbortable (tid: %d): %f",  omp_get_thread_num(), tendinitneighbortable - tstartinitneighbortable);
	#endif
		
	

	/////////////////////////////////////////////////////////	
	//END BATCH ESTIMATOR	
	/////////////////////////////////////////////////////////
	*/


	

	////////////////////////////////////
	//TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////			

	//debug values
	unsigned int * dev_debug1; 

	unsigned int * dev_debug2; 

	unsigned int * debug1; 
	debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug1=0;

	unsigned int * debug2; 
	debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug2=0;

	//allocate on the device
	gpuErrchk(cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int ) ));
	
	gpuErrchk(cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int ) ));
	
	//copy debug to device
	gpuErrchk(cudaMemcpy( dev_debug1, debug1, sizeof(unsigned int), cudaMemcpyHostToDevice ));

	gpuErrchk(cudaMemcpy( dev_debug2, debug2, sizeof(unsigned int), cudaMemcpyHostToDevice ));


	////////////////////////////////////
	//END TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////			

	

	///////////////////
	//ALLOCATE POINTERS TO INTEGER ARRAYS FOR THE VALUES FOR THE NEIGHBORTABLES
	///////////////////

	/*
	//THE NUMBER OF POINTERS IS EQUAL TO THE NUMBER OF BATCHES
	for (int i=0; i<numBatches; i++){
		
		struct neighborDataPtrs tmpStruct;
		tmpStruct.dataPtr=NULL;
		tmpStruct.sizeOfDataArr=0;
		
		pointersToNeighbors->push_back(tmpStruct);
	}
	*/
#if PROBEANDSORT==0 || USENEIGHBORTABLE==1 
	struct neighborDataPtrs tmpStruct;
	tmpStruct.dataPtr=NULL;
	tmpStruct.sizeOfDataArr=0;
#endif

	///////////////////
	//END ALLOCATE POINTERS TO INTEGER ARRAYS FOR THE VALUES FOR THE NEIGHBORTABLES
	///////////////////


	///////////////////////////////////
	//COUNT VALUES -- RESULT SET SIZE FOR EACH KERNEL INVOCATION
	///////////////////////////////////

	/*
	//total size of the result set as it's batched
	//this isnt sent to the GPU
	unsigned int * totalResultSetCnt;
	totalResultSetCnt=(unsigned int*)malloc(sizeof(unsigned int));
	*totalResultSetCnt=0;
	

	//count values - for an individual kernel launch
	//need different count values for each stream
	unsigned int * cnt;
	cnt=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	*cnt=0;
	*/

	unsigned long long int * dev_cnt; 

	//allocate on the device
	cudaMallocManaged(&dev_cnt, sizeof(unsigned long long int));
	cudaMemset(dev_cnt, 0, sizeof(unsigned long long int));
	

	///////////////////////////////////
	//END COUNT VALUES -- RESULT SET SIZE FOR EACH KERNEL INVOCATION
	///////////////////////////////////



	///////////////////////////////////
	//ALLOCATE MEMORY FOR THE RESULT SET USING THE BATCH ESTIMATOR
	///////////////////////////////////

	// unsigned int * dev_pointIDKey; //key
	// unsigned int * dev_pointInDistValue; //value

	size_t keyValElementsSize = ((size_t)(KEYVALUEMEM / 2) * (1024 * 1024 * 1024)) / sizeof(unsigned int);
	printf("\nNumber of allocated key value pairs: %zu", keyValElementsSize);
	
	// gpuErrchk(cudaMallocManaged((void **)&dev_pointIDKey, keyValElementsSize * sizeof(unsigned int)));
	// gpuErrchk(cudaMallocManaged((void **)&dev_pointInDistValue, keyValElementsSize * sizeof(unsigned int)));
	
	keyValPair * dev_keyValPairs;
	gpuErrchk(cudaMallocManaged((void **)&dev_keyValPairs, keyValElementsSize * sizeof(keyValPair)));

	//HOST RESULT ALLOCATION FOR THE GPU TO COPY THE DATA INTO A PINNED MEMORY ALLOCATION
	//ON THE HOST
	//pinned result set memory for the host
	//the number of elements are recorded for that batch in resultElemCountPerBatch
	//NEED PINNED MEMORY ALSO BECAUSE YOU NEED IT TO USE STREAMS IN THRUST FOR THE MEMCOPY OF THE SORTED RESULTS	

	/*
	//PINNED MEMORY TO COPY FROM THE GPU	
	int * pointIDKey[GPUSTREAMS]; //key
	int * pointInDistValue[GPUSTREAMS]; //value
		
	double tstartpinnedresults=omp_get_wtime();
	
	for (int i=0; i<GPUSTREAMS; i++)
	{
	cudaMallocHost((void **) &pointIDKey[i], sizeof(int)*GPUBufferSize);
	cudaMallocHost((void **) &pointInDistValue[i], sizeof(int)*GPUBufferSize);
	}

	double tendpinnedresults=omp_get_wtime();
	printf("\nTime to allocate pinned memory for results (tid: %d): %f", omp_get_thread_num(), tendpinnedresults - tstartpinnedresults);
	printf("\nmemory requested for results ON GPU (GiB): %f",(double)(sizeof(int)*2*GPUBufferSize*GPUSTREAMS)/(1024*1024*1024));
	printf("\nmemory requested for results in MAIN MEMORY (GiB): %f",(double)(sizeof(int)*2*GPUBufferSize*GPUSTREAMS)/(1024*1024*1024));
	*/

	
	///////////////////////////////////
	//END ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////
	
	/*
	/////////////////////////////////
	//SET OPENMP ENVIRONMENT VARIABLES
	////////////////////////////////
	omp_set_num_threads(GPUSTREAMS);
	/////////////////////////////////
	//END SET OPENMP ENVIRONMENT VARIABLES
	////////////////////////////////
	*/

	/////////////////////////////////
	//CREATE STREAMS
	////////////////////////////////

	cudaStream_t stream;
	
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

	/////////////////////////////////
	//END CREATE STREAMS
	////////////////////////////////
	
	

	///////////////////////////////////
	//LAUNCH KERNEL IN BATCHES
	///////////////////////////////////
		
	//since we use the strided scheme, some of the batch sizes
	//are off by 1 of each other, a first group of batches will
	//have 1 extra data point to process, and we calculate which batch numbers will 
	//have that.  The batchSize is the lower value (+1 is added to the first ones)

	CTYPE* dev_workCounts;
	cudaMalloc((void **)&dev_workCounts, sizeof(CTYPE)*2);
#if COUNTMETRICS == 1
		gpuErrchk(cudaMemcpy(dev_workCounts, workCounts, 2*sizeof(CTYPE), cudaMemcpyHostToDevice ));
#endif


	//the offset for batching, which keeps track of where to start processing at each batch
	unsigned int * batchOffset; //for the strided
	batchOffset = (unsigned int*)malloc(sizeof(unsigned int));
	*batchOffset = 1;
	unsigned int * dev_offset;
	gpuErrchk(cudaMalloc((void**)&dev_offset, sizeof(unsigned int)));
	gpuErrchk(cudaMemcpy( dev_offset, batchOffset, sizeof(unsigned int), cudaMemcpyHostToDevice ));

	//the batch number for batching with strided
	unsigned int * batchNumber;
	batchNumber = (unsigned int*)malloc(sizeof(unsigned int));
	*batchNumber = 0;
	unsigned int * dev_batchNumber;
	gpuErrchk(cudaMalloc((void**)&dev_batchNumber, sizeof(unsigned int)));
	gpuErrchk(cudaMemcpy( dev_batchNumber, batchNumber, sizeof(unsigned int), cudaMemcpyHostToDevice ));


	const int TOTALBLOCKS=ceil((1.0*(*DBSIZE))/(1.0*BLOCKSIZE));	
	printf("\ntotal blocks: %d",TOTALBLOCKS);

	// double tstart_kernel = omp_get_wtime();
	cudaEvent_t kernelStart;
	cudaEvent_t kernelStop;
    cudaEventCreate(&kernelStart);
	cudaEventCreate(&kernelStop);

	cudaEventRecord(kernelStart, stream);
	kernelNDGridIndexGlobal<<< TOTALBLOCKS, BLOCKSIZE, 0, stream>>>(dev_debug1, dev_debug2, *DBSIZE, 
dev_offset, dev_batchNumber, dev_database, dev_epsilon, dev_grid, dev_indexLookupArr, 
dev_gridCellLookupArr, dev_minArr, dev_nCells, dev_cnt, dev_nNonEmptyCells, dev_gridCellNDMask, 
dev_gridCellNDMaskOffsets, dev_keyValPairs, dev_orderedQueryPntIDs, dev_workCounts);
	cudaEventRecord(kernelStop, stream);

	// errCode=cudaDeviceSynchronize();
	// cout <<"\n\nError from device synchronize: "<<errCode;

	cout <<"\n\nKERNEL LAUNCH RETURN: "<<cudaGetLastError()<<endl<<endl;
	if ( cudaSuccess != cudaGetLastError() ){
		cout <<"\n\nERROR IN KERNEL LAUNCH. ERROR: "<<cudaSuccess<<endl<<endl;
	}

#if PROBEANDSORT==1
	probeAndSort(dev_keyValPairs, dev_cnt, keyValElementsSize, *DBSIZE, &kernelStop, keyBinsMap, totalSortTime);
#endif

	cudaEventSynchronize(kernelStop);
	
	// double tend_kernel = omp_get_wtime();
	float milliseconds;
	cudaEventElapsedTime(&milliseconds, kernelStart, kernelStop);
	printf("\nKernel execution time: %f", (milliseconds / 1000));
	fprintf(stderr,"\nTotal of total size of result array: %llu", *dev_cnt);
	printf("\n[After synchronization] Num elems generated in array (Fraction: %f): %llu", *dev_cnt*1.0/keyValElementsSize*1.0, *dev_cnt);
	if(keyValElementsSize < *dev_cnt) {
		cout << "\n\nWARNING: Total result set size exceeds elements allocated for key value pairs. Neighbor table will be inaccurate.\n" << endl;
	}
	cudaEventDestroy(kernelStart);
	cudaEventDestroy(kernelStop);

	*kernelExecutionTime = (milliseconds / 1000);

#if PROBEANDSORT==1 && USENEIGHBORTABLE==1
	double tableconstuctstart=omp_get_wtime();

	tmpStruct.sizeOfDataArr=*dev_cnt;    
	tmpStruct.dataPtr=new int[*dev_cnt];
	
	moveKeyBinsToNeighborTable(*DBSIZE, dev_keyValPairs, keyBinsMap, neighborTable, tmpStruct.dataPtr);

	double tableconstuctend=omp_get_wtime();	
	
	printf("\nTable construct time: %f", tableconstuctend - tableconstuctstart);

	*tableConstructionTime = (tableconstuctend - tableconstuctstart);
#endif

#if PROBEANDSORT==0
	// gnu parallel sort by key 
	double tstart_sort = omp_get_wtime();
	fprintf(stderr, "\nSorting pairs...");
	__gnu_parallel::sort(dev_keyValPairs, dev_keyValPairs+*dev_cnt, compareKeyValPairs);
	double tend_sort = omp_get_wtime();
	printf("\nSort time: %f", (tend_sort - tstart_sort));

	*totalSortTime = (tend_sort - tstart_sort);

	double tableconstuctstart=omp_get_wtime();
	//set the number of neighbors in the pointer struct:
	tmpStruct.sizeOfDataArr=*dev_cnt;    
	tmpStruct.dataPtr=new int[*dev_cnt]; // NOTE: Do not frees this from memory until program is finished

	constructNeighborTableKeyValueWithPtrs( dev_keyValPairs, neighborTable, tmpStruct.dataPtr, dev_cnt);
	
	double tableconstuctend=omp_get_wtime();	
	
	printf("\nTable construct time: %f", tableconstuctend - tableconstuctstart);

	*tableConstructionTime = (tableconstuctend - tableconstuctstart);
#endif


#if COUNTMETRICS == 1
	cudaMemcpy(workCounts, dev_workCounts, 2*sizeof(CTYPE), cudaMemcpyDeviceToHost );
	printf("\nPoint comparisons: %llu, Cell evaluations: %llu", workCounts[0],workCounts[1]);
#endif

	
	
	printf("\nTOTAL RESULT SET SIZE ON HOST:  %llu", *dev_cnt);
	*totalNeighbors=*dev_cnt;


	double tKernelResultsEnd=omp_get_wtime();
	
	printf("\nTime to launch kernel and execute everything (get results etc.) except freeing memory: %f",tKernelResultsEnd-tKernelResultsStart);




	///////////////////////////////////
	//END GET RESULT SET
	///////////////////////////////////



	///////////////////////////////////	
	//OPTIONAL DEBUG VALUES
	///////////////////////////////////
	
	// gpuErrchk(cudaMemcpy(debug1, dev_debug1, sizeof(unsigned int), cudaMemcpyDeviceToHost )); printf("\nDebug1 value: %u",*debug1);
	// gpuErrchk(cudaMemcpy(debug2, dev_debug2, sizeof(unsigned int), cudaMemcpyDeviceToHost ));printf("\nDebug2 value: %u",*debug2);	

	///////////////////////////////////	
	//END OPTIONAL DEBUG VALUES
	///////////////////////////////////
	

	///////////////////////////////////
	//FREE MEMORY FROM THE GPU
	///////////////////////////////////

	double tFreeStart=omp_get_wtime();

	errCode=cudaStreamDestroy(stream);
	if(errCode != cudaSuccess) {
	cout << "\nError: destroying stream" << errCode << endl; 
	}


	#if QUERYREORDER==1
	cudaFree(dev_orderedQueryPntIDs);
	#endif

	//free the data on the device
	cudaFree(dev_database);
	cudaFree(dev_debug1);
	cudaFree(dev_debug2);
	cudaFree(dev_epsilon);
	cudaFree(dev_grid);
	cudaFree(dev_gridCellLookupArr);
	cudaFree(dev_gridCellNDMask);
	cudaFree(dev_gridCellNDMaskOffsets);
	cudaFree(dev_indexLookupArr);
	cudaFree(dev_minArr);
	cudaFree(dev_nCells);
	cudaFree(dev_nNonEmptyCells);
	// cudaFree(dev_N); 	
	cudaFree(dev_cnt); 
	cudaFree(dev_offset); 
	cudaFree(dev_batchNumber); 
	free(batchOffset);
	free(batchNumber);

	
	//free data related to the individual streams for each batch
	// for (int i=0; i<GPUSTREAMS; i++){
		//free the data on the device
	/*
		//free on the host
		cudaFreeHost(pointIDKey[i]);
		cudaFreeHost(pointInDistValue[i]);
	}
	*/

	*keyValPairs = dev_keyValPairs;

	double tFreeEnd=omp_get_wtime();

	printf("\nTime freeing memory: %f", tFreeEnd - tFreeStart);
	// }
	cout<<"\n** last error at end of fn batches (could be from freeing memory): "<<cudaGetLastError();

}


void constructNeighborTableKeyValueWithPtrs(keyValPair * keyValPairs, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned long long int * cnt) {
	#if STAMP==0

	#pragma omp parallel for num_threads(8)
	for (unsigned int i=0; i<(*cnt); i++)
	{
		pointersToNeighbors[i]=keyValPairs[i].val;
	}


	std::vector<keyData> uniqueKeyData;

	keyData tmp;
	tmp.key=keyValPairs[0].key;
	tmp.position=0;
	uniqueKeyData.push_back(tmp);

	//we assign the ith data item when iterating over i+1th data item,
	//so we go 1 loop iteration beyond the number (*cnt)
	for (unsigned long long i=1; i<(*cnt)+1; i++){
		if (keyValPairs[i-1].key!=keyValPairs[i].key){
			tmp.key=keyValPairs[i].key;
			tmp.position=i;
			uniqueKeyData.push_back(tmp);
		}
		if ( i < (*cnt) && keyValPairs[i-1].key>keyValPairs[i].key ) {
			printf("\nERROR: The key-value pairs are not sorted correctly. Exiting table construction. "
				   "\nFound an instance of %d coming before %d at indexes %llu and %llu respectively", keyValPairs[i-1].key, keyValPairs[i].key, i-1, i);
			return;
		}
	}

	printf("\nUnique keys: %lu", uniqueKeyData.size());

	
	//insert into the neighbor table the values based on the positions of 
	//the unique keys obtained above. 
	for (int i=0; i<uniqueKeyData.size()-1; i++) {
		int keyElem=uniqueKeyData[i].key;
		neighborTable[keyElem].pointID=keyElem;
		neighborTable[keyElem].indexmin=uniqueKeyData[i].position;
		neighborTable[keyElem].indexmax=uniqueKeyData[i+1].position-1;
	
		//update the pointer to the data array for the values
		neighborTable[keyElem].dataPtr=pointersToNeighbors;	
	}
	#endif
}


void constructNeighborTableKeyValueWithPtrs(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned int * cnt)
{
	#if STAMP==0
	
	//copy the value data:
	std::copy(pointInDistValue, pointInDistValue+(*cnt), pointersToNeighbors);



	//Step 1: find all of the unique keys and their positions in the key array
	unsigned int numUniqueKeys=0;

	std::vector<keyData> uniqueKeyData;

	keyData tmp;
	tmp.key=pointIDKey[0];
	tmp.position=0;
	uniqueKeyData.push_back(tmp);

	//we assign the ith data item when iterating over i+1th data item,
	//so we go 1 loop iteration beyond the number (*cnt)
	for (int i=1; i<(*cnt)+1; i++){
		if (pointIDKey[i-1]!=pointIDKey[i]){
			numUniqueKeys++;
			tmp.key=pointIDKey[i];
			tmp.position=i;
			uniqueKeyData.push_back(tmp);
		}
	}

	
	//insert into the neighbor table the values based on the positions of 
	//the unique keys obtained above. 
	for (int i=0; i<uniqueKeyData.size()-1; i++) {
		int keyElem=uniqueKeyData[i].key;
		neighborTable[keyElem].pointID=keyElem;
		neighborTable[keyElem].indexmin=uniqueKeyData[i].position;
		neighborTable[keyElem].indexmax=uniqueKeyData[i+1].position-1;
	
		//update the pointer to the data array for the values
		neighborTable[keyElem].dataPtr=pointersToNeighbors;	
	}
	#endif

	}




//Fixing the issue with unicomp requiring multiple updates and overwriting the data
void constructNeighborTableKeyValueWithPtrsWithMultipleUpdatesMultipleDataArrays(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned long long int * cnt, int * uniqueKeys, int * uniqueKeyPosition, unsigned long long int numUniqueKeys)
{

	omp_set_max_active_levels(3);

	#pragma omp parallel for num_threads(8)
	for (unsigned int i=0; i<(*cnt); i++)
	{
		pointersToNeighbors[i]=pointInDistValue[i];
	}

	double tstartconstruct=omp_get_wtime();
	
	//if using unicomp we need to update different parts of the struct
	#if STAMP==1
	#pragma omp parallel for num_threads(8)
	for (unsigned int i=0; i<numUniqueKeys; i++) {

		int keyElem=uniqueKeys[i];
		//Update counter to write position in critical section
		pthread_mutex_lock(&neighborTable[keyElem].pointLock);
		
		int nextIdx=neighborTable[keyElem].cntNDataArrays;
		neighborTable[keyElem].cntNDataArrays++;
		pthread_mutex_unlock(&neighborTable[keyElem].pointLock);

		neighborTable[keyElem].vectindexmin[nextIdx]=uniqueKeyPosition[i];
		neighborTable[keyElem].vectdataPtr[nextIdx]=pointersToNeighbors;	

		//final value will be missing
		if (i==(numUniqueKeys-1))
		{
			neighborTable[keyElem].vectindexmax[nextIdx]=(*cnt)-1;
		}
		else
		{
			neighborTable[keyElem].vectindexmax[nextIdx]=(uniqueKeyPosition[i+1])-1;
		}
	}
	#endif


	//if not using unicomp we need to update the original parts of the struct
	#if STAMP==0
	#pragma omp parallel for num_threads(8)
	for (unsigned int i=0; i<numUniqueKeys; i++) {

			int keyElem=uniqueKeys[i];
			neighborTable[keyElem].pointID=keyElem;

			neighborTable[keyElem].indexmin=uniqueKeyPosition[i];
			neighborTable[keyElem].dataPtr=pointersToNeighbors;	

			//final value will be missing
			if (i==(numUniqueKeys-1))
			{
				neighborTable[keyElem].indexmax=(*cnt)-1;
			}
			else
			{
				neighborTable[keyElem].indexmax=(uniqueKeyPosition[i+1])-1;
			}
		}
	#endif

	double tendconstruct=omp_get_wtime();
	printf("\nTime to do the copy: %f", tendconstruct - tstartconstruct);
	
	return;


} //end function





void constructNeighborTableKeyValue(int * pointIDKey, int * pointInDistValue, struct table * neighborTable, unsigned int * cnt)
{
	
	//newer multithreaded way:
	//Step 1: find all of the unique keys and their positions in the key array
	
	//double tstart=omp_get_wtime();

	unsigned int numUniqueKeys=0;
	unsigned int count=0;

	

	std::vector<keyData> uniqueKeyData;

	keyData tmp;
	tmp.key=pointIDKey[0];
	tmp.position=0;
	uniqueKeyData.push_back(tmp);



	//we assign the ith data item when iterating over i+1th data item,
	//so we go 1 loop iteration beyond the number (*cnt)
	for (int i=1; i<(*cnt)+1; i++)
	{
		if (pointIDKey[i-1]!=pointIDKey[i])
		{
			numUniqueKeys++;
			tmp.key=pointIDKey[i];
			tmp.position=i;
			uniqueKeyData.push_back(tmp);
		}
	}



	//Step 2: In parallel, insert into the neighbor table the values based on the positions of 
	//the unique keys obtained above. Since multiple threads access this function, we don't want to 
	//do too many memory operations while GPU memory transfers are occurring, or else we decrease the speed that we 
	//get data off of the GPU
	omp_set_max_active_levels(3);
	#pragma omp parallel for reduction(+:count) num_threads(2) schedule(static,1)
	for (int i=0; i<uniqueKeyData.size()-1; i++) 
	{
		int keyElem=uniqueKeyData[i].key;
		int valStart=uniqueKeyData[i].position;
		int valEnd=uniqueKeyData[i+1].position-1;
		int size=valEnd-valStart+1;
		
		//seg fault from here: is it neighbortable mem alloc?
		neighborTable[keyElem].pointID=keyElem;
		neighborTable[keyElem].neighbors.insert(neighborTable[keyElem].neighbors.begin(),&pointInDistValue[valStart],&pointInDistValue[valStart+size]);
		count+=size;

	}
	

}





//Uses a brute force kernel to calculate the direct neighbors of the points in the database
void makeDistanceTableGPUBruteForce(std::vector<std::vector <DTYPE> > * NDdataPoints, DTYPE* epsilon, struct table * neighborTable, unsigned long long int * totalNeighbors)
{

	///////////////////////////////////
	//COPY THE DATABASE TO THE GPU
	///////////////////////////////////
	unsigned int * N;
	N=(unsigned int*)malloc(sizeof(unsigned int));
	*N=NDdataPoints->size();
	
	printf("\nIn main GPU method: Number of data points, (N), is: %u ",*N);cout.flush();



	
	//the database will just be a 1-D array, we access elemenets based on NDIM
	DTYPE* database= (DTYPE*)malloc(sizeof(DTYPE)*(*N)*GPUNUMDIM);  
	DTYPE* dev_database= (DTYPE*)malloc(sizeof(DTYPE)*(*N)*GPUNUMDIM);  
	

	//allocate memory on device:
	gpuErrchk(cudaMalloc( (void**)&dev_database, sizeof(DTYPE)*GPUNUMDIM*(*N)));
	//copy the database from the ND vector to the array:
	for (int i=0; i<*N; i++){
		std::copy((*NDdataPoints)[i].begin(), (*NDdataPoints)[i].end(), database+(i*GPUNUMDIM));
	}
	
	//copy database to the device:
	gpuErrchk(cudaMemcpy(dev_database, database, sizeof(DTYPE)*(*N)*GPUNUMDIM, cudaMemcpyHostToDevice));

	///////////////////////////////////
	//END COPY THE DATABASE TO THE GPU
	///////////////////////////////////

	


	///////////////////////////////////
	//ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////
	//NON-PINNED MEMORY FOR SINGLE KERNEL INVOCATION (NO BATCHING)


	//CHANGING THE RESULTS TO KEY VALUE PAIR SORT, WHICH IS TWO ARRAYS
	//KEY IS THE POINT ID
	//THE VALUE IS THE POINT ID WITHIN THE DISTANCE OF KEY

	int * dev_pointIDKey; //key
	int * dev_pointInDistValue; //value

	//num elements for the result set
	#define BUFFERELEM 300000000 


	gpuErrchk(cudaMalloc((void **)&dev_pointIDKey, sizeof(int)*BUFFERELEM));
	gpuErrchk(cudaMalloc((void **)&dev_pointInDistValue, sizeof(int)*BUFFERELEM));
	printf("\nmemory requested for results (GiB): %f",(double)(sizeof(int)*2*BUFFERELEM)/(1024*1024*1024));

	///////////////////////////////////
	//END ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////


	///////////////////////////////////
	//SET OTHER KERNEL PARAMETERS
	///////////////////////////////////

	
	
	//count values
	unsigned long long int * cnt;
	cnt=(unsigned long long int*)malloc(sizeof(unsigned long long int));
	*cnt=0;

	unsigned long long int * dev_cnt; 
	dev_cnt=(unsigned long long int*)malloc(sizeof(unsigned long long int));
	*dev_cnt=0;

	//allocate on the device
	gpuErrchk(cudaMalloc((unsigned long long int**)&dev_cnt, sizeof(unsigned long long int)));
	
	gpuErrchk(cudaMemcpy( dev_cnt, cnt, sizeof(unsigned long long int), cudaMemcpyHostToDevice ));
	
	//Epsilon
	DTYPE* dev_epsilon;
	dev_epsilon=(DTYPE*)malloc(sizeof( DTYPE));

	//Allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_epsilon, sizeof(DTYPE)));
		
	//size of the database:
	unsigned int * dev_N; 
	dev_N=(unsigned int*)malloc(sizeof( unsigned int ));

	//allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_N, sizeof(unsigned int)));

	//debug values
	unsigned int * dev_debug1; 
	unsigned int * dev_debug2; 
	dev_debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug1=0;
	dev_debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug2=0;




	//allocate on the device
	gpuErrchk(cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int ) ));
	
	gpuErrchk(cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int ) ));
	
	//copy N, epsilon to the device
	//epsilon
	gpuErrchk(cudaMemcpy( dev_epsilon, epsilon, sizeof(DTYPE), cudaMemcpyHostToDevice ));
	
	//N (DATASET SIZE)
	gpuErrchk(cudaMemcpy( dev_N, N, sizeof(unsigned int), cudaMemcpyHostToDevice ));

	///////////////////////////////////
	//END SET OTHER KERNEL PARAMETERS
	///////////////////////////////////


	


	///////////////////////////////////
	//LAUNCH KERNEL
	///////////////////////////////////

	const int TOTALBLOCKS=ceil((1.0*(*N))/(1.0*BLOCKSIZE));	
	printf("\ntotal blocks: %d",TOTALBLOCKS);


	//execute kernel	

	
	double tkernel_start=omp_get_wtime();
	kernelBruteForce<<< TOTALBLOCKS, BLOCKSIZE >>>(dev_N, dev_debug1, dev_debug2, dev_epsilon, dev_cnt, dev_database, dev_pointIDKey, dev_pointInDistValue);
	if ( cudaSuccess != cudaGetLastError() ){
    	printf( "Error in kernel launch!\n" );
    }


    cudaDeviceSynchronize();
    double tkernel_end=omp_get_wtime();
    printf("\nTime for kernel only: %f", tkernel_end - tkernel_start);
    ///////////////////////////////////
	//END LAUNCH KERNEL
	///////////////////////////////////
    


    ///////////////////////////////////
	//GET RESULT SET
	///////////////////////////////////

	//first find the size of the number of results
	gpuErrchk(cudaMemcpy( cnt, dev_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost ));
	printf("\nGPU: result set size on within epsilon: %llu",*cnt);
	

	*totalNeighbors=(*cnt);

	//get debug information (optional)
	unsigned int * debug1;
	debug1=(unsigned int*)malloc(sizeof(unsigned int));
	*debug1=0;
	unsigned int * debug2;
	debug2=(unsigned int*)malloc(sizeof(unsigned int));
	*debug2=0;

	gpuErrchk(cudaMemcpy(debug1, dev_debug1, sizeof(unsigned int), cudaMemcpyDeviceToHost ));printf("\nDebug1 value: %u",*debug1);
	

	gpuErrchk(cudaMemcpy(debug2, dev_debug2, sizeof(unsigned int), cudaMemcpyDeviceToHost ));
	printf("\nDebug2 value: %u",*debug2);
	
	///////////////////////////////////
	//END GET RESULT SET
	///////////////////////////////////


	///////////////////////////////////
	//FREE MEMORY FROM THE GPU
	///////////////////////////////////
    //free:
	cudaFree(dev_database);
	cudaFree(dev_debug1);
	cudaFree(dev_debug2);
	cudaFree(dev_cnt);
	cudaFree(dev_epsilon);

	////////////////////////////////////



}




//Order the work from most to least work
//Based on the points within each cell
void computeWorkDifficulty(unsigned int * outputOrderedQueryPntIDs, struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, unsigned int * indexLookupArr, struct grid * index)
{
	std::vector<workArray> totalWork; 

	//loop over each non-empty cell and find the points contained within
	//record the number of points in the cell
	for (int i=0; i<*nNonEmptyCells; i++)
	{
		unsigned int grid_cell_idx=gridCellLookupArr[i].idx;		

		unsigned int numPtsInCell=(index[grid_cell_idx].indexmax-index[grid_cell_idx].indexmin)+1;

			for (int j=index[i].indexmin; j<=index[i].indexmax;j++)
			{
				workArray tmp;
				tmp.queryPntID=indexLookupArr[j];
				tmp.pntsInCell=numPtsInCell;
				totalWork.push_back(tmp);
			}



	}

	//sort the array containing the total work:
	std::sort(totalWork.begin(), totalWork.end(), compareWorkArrayByNumPointsInCell);

	for (unsigned int i=0; i<totalWork.size(); i++)
	{
		outputOrderedQueryPntIDs[i]=totalWork[i].queryPntID;
	}

	return;	

}

void warmUpGPU(){
// initialize all ten integers of a device_vector to 1 
thrust::device_vector<int> D(10, 1); 
// set the first seven elements of a vector to 9 
thrust::fill(D.begin(), D.begin() + 7, 9); 
// initialize a host_vector with the first five elements of D 
thrust::host_vector<int> H(D.begin(), D.begin() + 5); 
// set the elements of H to 0, 1, 2, 3, ... 
thrust::sequence(H.begin(), H.end()); // copy all of H back to the beginning of D 
thrust::copy(H.begin(), H.end(), D.begin()); 
// print D 
for(int i = 0; i < D.size(); i++) 
std::cout << " D[" << i << "] = " << D[i]; 
return;
}


void bubbleSortByKey(int * keysPtr, int * valsPtr, unsigned long long int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (keysPtr[j] > keysPtr[j + 1])
                swap(keysPtr[j], keysPtr[j + 1]);
				swap(valsPtr[j], valsPtr[j + 1]);
        }
    }
}


void hostUniqueKeys(keyValPair * keyValPairs, unsigned long long int * size, keyValPair * uniqueKeyPosPairs, unsigned long long int * uniqueCnt) {
	for( int i=0; i<(*size); i++ ) {
		if (i==0)
		{	
			uniqueKeyPosPairs[*uniqueCnt].key=keyValPairs[0].key;
			uniqueKeyPosPairs[*uniqueCnt].val=0;
			*uniqueCnt +=1;
		}
		else if (keyValPairs[i-1].key!=keyValPairs[i].key)
		{
			uniqueKeyPosPairs[*uniqueCnt].key=keyValPairs[i].key;
			uniqueKeyPosPairs[*uniqueCnt].val=i;
			*uniqueCnt +=1;
		}
	}
}


void probeAndSort(
	keyValPair * keyValPairs,
	unsigned long long int * cnt,
	bool * completedArray,
	unsigned int *	countNeighbors,
	const unsigned long long int maxUnsortedNELEMS, 
	const unsigned int numElemsCompletedArray 
	)
{

	//offset into output sorted buffer
	// uint64_t idxOffset = 0;	
	//For clarity bufferToSort is just a pointer to the output array,
	//so we don't allocate unneeded memory and perform unnecessary memory copies.
	//But bufferToSort points to the location that needs to be sorted next
	// keyValPair * bufferToSort = &keyValPairs[idxOffset];

	//Should only probe-and-sort if the size of the array 
	//is using significant GPU memory
	// uint64_t thresholdElems = 1000;

	//NCOMPLETETHRESH should be a multiple of BLOCKSIZE
	unsigned int NCOMPLETETHRESH = BLOCKSIZE*((FRACTIONTHREADSCOMPLETE*numElemsCompletedArray*1.0)/(BLOCKSIZE*1.0));
	printf("\nNCOMPLETETHRESH: %u", NCOMPLETETHRESH);
	// unsigned int NCOMPLETETHRESH = BLOCKSIZE*1000;

	// get the number of elements in one page
	unsigned int elemsPerPage = ((PAGESIZE) * (1024)) / sizeof(unsigned int);
	printf("\nelemsPerPage: %u", elemsPerPage);

	unsigned int offSetComplete = 0;

	//Range of the "point ids" to read from the unsorted array:
	// unsigned int rangeMin = offSetComplete;
	// unsigned int rangeMax = min(numElemsCompletedArray, offSetComplete+NCOMPLETETHRESH);
	unsigned int rangeMin = offSetComplete;
	unsigned int rangeMax = min(numElemsCompletedArray, offSetComplete+NCOMPLETETHRESH);


	uint64_t cntOutputBuffer = 0;

	uint64_t totalSortedElems = 0;

	//used to short circuit the scan when copying elements from
	//unsorted buffer to a temp buffer for sorting;
	uint64_t elemsLowerBound = 0;
	// uint64_t elemsContinguousLowerBound = 0;
	uint64_t elemsUpperBound = 0;

	//probe until we can sort some of the elements
	bool ret = 0;
	unsigned long long int localCnt=0;
	// while(ret==0 && (*cnt)<maxUnsortedNELEMS)
	// while((*cnt)<maxUnsortedNELEMS)
	// while((*cnt)<maxUnsortedNELEMS)

	unsigned int retTrueIteration = 0;

	while(totalSortedElems<maxUnsortedNELEMS && localCnt<maxUnsortedNELEMS && !allComplete(completedArray, offSetComplete, numElemsCompletedArray))
	{

		// bufferToSort = &keyValPairs[idxOffset];

		//sleep for SLEEPSEC seconds
		//don't want to probe too much or else we'll cause too many page faults
		//if we need to leave the loop, then do not sleep
		usleep(SLEEPSEC*1000000);

		//use localCnt so we don't access it too much
		//because we will page fault more between CPU and GPU
		localCnt = *cnt;

		printf("\nNum elems generated in array on GPU: %llu", localCnt);
		ret = checkQueriesComplete(completedArray, offSetComplete, numElemsCompletedArray, NCOMPLETETHRESH);
		// ret = checkQueriesCompleteAsFraction(completedArray, offSetComplete, numElemsCompletedArray, FRACTIONTHREADSCOMPLETE);
		printf("\nret: %d", ret);

		// continue;

		//Output array:
		//If ret is true, need to scan for the elements and then sort in separate buffer
		
		if(ret==1)
		{
			printf("\nretTrueIteration: %u", retTrueIteration);

			uint64_t elemsToSort = computeElemsToSort(countNeighbors, offSetComplete, numElemsCompletedArray, NCOMPLETETHRESH);	
			// uint64_t elemsToSort = computeElemsToSortAsFraction(countNeighbors, offSetComplete, numElemsCompletedArray, FRACTIONTHREADSCOMPLETE);	
			printf("\nElems to sort: %lu", elemsToSort);	

			// tmp_elemsLowerBound = elemsContinguousLowerBound;
			// tmp_elemsContinguousLowerBound = getCopiedContiguousElemsLowerBound(keyValPairs, elemsToSort, localCnt, rangeMin, rangeMax, elemsLowerBound, tmp_elemsUpperBound);

			//Range of the "point ids" to read from the unsorted array:

			//original
			rangeMin = offSetComplete;
			rangeMax = min(numElemsCompletedArray, offSetComplete+NCOMPLETETHRESH);

			// GET UPPER BOUND
			elemsUpperBound = getElemsUpperBound(keyValPairs, elemsToSort, localCnt, rangeMin, rangeMax, elemsLowerBound);
			
			// printf("\nelemsLowerBound: %lu\nelemsUpperBound: %lu", elemsLowerBound, elemsUpperBound);

			// check if the kernel is finished writting to the page
			unsigned int elemsToSortPageIdx = (elemsUpperBound + elemsPerPage - 1) / (elemsPerPage);
			unsigned int cntPageIdx = (localCnt + elemsPerPage - 1) / (elemsPerPage);
			printf("\nelemsToSortPageIdx: %d\ncntPageIdx: %d", elemsToSortPageIdx, cntPageIdx);
			if( elemsToSortPageIdx < cntPageIdx )
			{
				// elemsLowerBound = tmp_elemsLowerBound;
				// elemsContinguousLowerBound = tmp_elemsContinguousLowerBound;
				// elemsUpperBound = tmp_elemsUpperBound;
				//printf("\nelemsLowerBound: %lu\nelemsUpperBound: %lu", elemsLowerBound, elemsUpperBound);

				//when using the fraction of work
				//TODO: return rangeMin/rangeMax using computeElemsToSortAsFraction above 
				// rangeMin = offSetComplete;
				// rangeMax = min((unsigned int)numElemsCompletedArray, (unsigned int)offSetComplete+(unsigned int)(numElemsCompletedArray*FRACTIONTHREADSCOMPLETE));
				
				//copy unsorted elems to bufferToSort using a scan
				cntOutputBuffer = 0;
				
				/*
				#if PREFETCH==1
				//Prefetch chunk of unsortedBuffer to CPU memory
				cudaMemPrefetchAsync(&unsortedBuffer[elemsLowerBound], sizeof(unsigned int)*localCnt, cudaCpuDeviceId, 0);
				#endif
				*/

				
				//sort buffer FROM LOWER BOUND TO UPPERBOUND
				printf("\nSorting %lu elements in bounds [%lu, %lu) corresponding to query points in the range [%u, %u)", 
																	elemsToSort, elemsLowerBound, elemsUpperBound, rangeMin, rangeMax);
				double tstart_sort = omp_get_wtime();
				__gnu_parallel::sort(keyValPairs+elemsLowerBound, keyValPairs+elemsUpperBound, compareKeyValPairs);
				double tend_sort = omp_get_wtime();
				printf("\nSort time: %f", tend_sort-tstart_sort);


				// GET THE NEXT LOWER BOUND (CURRENT LOWER BOUND + NUM ELEMS)
				elemsLowerBound += elemsToSort;

				//increase offset for output array
				// idxOffset+=elemsToSort;
				
				//increase offset
				offSetComplete+=NCOMPLETETHRESH;

				//increment total sorted elements
				totalSortedElems+=elemsToSort;

				printf("\nTotal sorted elements: %lu (frac: %f)", totalSortedElems, (totalSortedElems*1.0)/(maxUnsortedNELEMS*1.0));

				retTrueIteration++;
			}
		}

			
	} //end of while loop

	//////////////////////////
	//Sort the "leftovers" that were not sorted above

	// bufferToSort = &sortedKeyValPairs[idxOffset];

	localCnt = *cnt;

	NCOMPLETETHRESH = numElemsCompletedArray; // get the rest of the elements in the array

	// uint64_t prefixSumSize=0;
	uint64_t elemsToSort = computeElemsToSort(countNeighbors, offSetComplete, numElemsCompletedArray, NCOMPLETETHRESH);
	
	// unsigned int * bufferToSort = (unsigned int*)malloc(sizeof(unsigned int)*(elemsToSort));
	fprintf(stderr,"\n[Leftovers] Elems to sort: %lu", elemsToSort);

	//Range of the "point ids" to read from the unsorted array:
	rangeMin = offSetComplete;
	rangeMax = numElemsCompletedArray;

	// elemsLowerBound = elemsContinguousLowerBound;
	// elemsContinguousLowerBound = getCopiedContiguousElemsLowerBound(keyValPairs, elemsToSort, localCnt, rangeMin, rangeMax, elemsLowerBound, elemsUpperBound);
	// printf("\n[Leftovers] elemsLowerBound: %lu\n[Leftovers] elemsUpperBound: %lu", elemsLowerBound, elemsUpperBound);	
	
	// SET UPPERBOUND TO ARRAY END
	elemsUpperBound = localCnt;
	// elemsUpperBound = getElemsUpperBound(keyValPairs, elemsToSort, localCnt, rangeMin, rangeMax, elemsLowerBound);

	/*
	#if PREFETCH==1
	//Prefetch last chunk of unsortedBuffer to CPU memory
	cudaMemPrefetchAsync(&unsortedBuffer[elemsLowerBound], sizeof(unsigned int)*maxUnsortedNELEMS, cudaCpuDeviceId, 0);
	#endif
	*/

	// double tstart_cpy = omp_get_wtime();	
	// parallelCopyToBuffer(bufferToSort, dev_pointIDKey, dev_pointInDistValue, elemsToSort, localCnt, rangeMin, rangeMax, elemsLowerBound);
	// double tend_cpy = omp_get_wtime();
	// fprintf(stderr, "\n[Leftovers] Data transfer time: %f", tend_cpy-tstart_cpy);

	fprintf(stderr,"\n[Leftovers] cntOutputBuffer: %lu", cntOutputBuffer);
	
	printf("\n\n");

	//sort buffer
	printf("\nSorting %lu elements in bounds [%lu, %lu) corresponding to query points in the range [%u, %u)", 
														elemsToSort, elemsLowerBound, elemsUpperBound, rangeMin, rangeMax);
	double tstart_sort = omp_get_wtime();	
	__gnu_parallel::sort(keyValPairs+elemsLowerBound, keyValPairs+elemsUpperBound, compareKeyValPairs);
	double tend_sort = omp_get_wtime();
	fprintf(stderr, "\n[Leftovers] Sort time: %f", tend_sort-tstart_sort);

	// uint64_t lastIdxArray = idxOffset+elemsToSort;
	fprintf(stderr, "\n[Leftovers] Last idx: %lu (should be NELEMS)", elemsUpperBound);

}


bool allComplete(bool * completedArray, unsigned int offSetComplete, const unsigned int numElemsCompletedArray) {
	bool flag = 1;
	#pragma omp parallel for shared(flag) num_threads(8)
	for( unsigned int i=offSetComplete; i<numElemsCompletedArray; i++ ) {
		if( flag ) {
			if( completedArray[i]==0 ) {
				flag = 0;
			}
		}
	}

	return flag;
}

bool checkQueriesComplete(bool * completedArray, unsigned int offSetComplete, const unsigned int numElemsCompletedArray, const unsigned int NCOMPLETETHRESH)
{
	unsigned int ubound = min(numElemsCompletedArray, offSetComplete+NCOMPLETETHRESH);

	//flag that denotes that the array should be sorted based on completed work
	//assume true
	bool flag = 1;
	for(unsigned int i=offSetComplete; i<ubound; i++){
		if(completedArray[i]==0){
			flag = 0;
			break;
		}
	}

	return flag;
}


uint64_t computeElemsToSort(unsigned int * countNeighbors, unsigned int offSetComplete, const unsigned int numElemsCompletedArray, const unsigned int NCOMPLETETHRESH)
{
	uint64_t numElemsToSort=0;
	unsigned int ubound = min(numElemsCompletedArray, offSetComplete+NCOMPLETETHRESH);

	#pragma omp parallel for reduction(+:numElemsToSort) shared(countNeighbors, ubound) num_threads(NCOPYTHREADS)
	for(unsigned int i=offSetComplete; i<ubound; i++){
		numElemsToSort +=(uint64_t)countNeighbors[i];
	}

	return numElemsToSort;
}


uint64_t getElemsUpperBound(keyValPair * keyValPairs,
	uint64_t elemsToSort, unsigned long long int localCnt,
	unsigned int rangeMin, unsigned int rangeMax, uint64_t elemsLowerBound)
{
	uint64_t cntOutputBuffer = 0;
	uint64_t elemsUpperBound = 0;
	
	// for(uint64_t i=0; i<(localCnt) && (cntOutputBuffer<elemsToSort); i++)
	for(uint64_t i=elemsLowerBound; i<localCnt && (cntOutputBuffer<elemsToSort); i++)
	{
		if(keyValPairs[i].key>=rangeMin && keyValPairs[i].key<rangeMax)
		{
			cntOutputBuffer++;
		}
		
		// update upper bound
		elemsUpperBound = i;
	}

	elemsUpperBound += 1;

	// printf("\ncntOutputBuffer: %lu", cntOutputBuffer);

	return elemsUpperBound;
}


//Only count the number of elements that have already been copied in unsortedBuffer
// I believe this could be parallelized
uint64_t getCopiedContiguousElemsLowerBound(keyValPair * keyValPairs,
	uint64_t elemsToSort, unsigned long long int localCnt,
	unsigned int rangeMin, unsigned int rangeMax, uint64_t elemsLowerBound, uint64_t& elemsUpperBound)
{
	uint64_t cntOutputBuffer = 0;
	
	bool flagElemsLowerBound = 0;
	uint64_t copiedContiguousElemsLowerBound = elemsLowerBound;
	
	// for(uint64_t i=0; i<(localCnt) && (cntOutputBuffer<elemsToSort); i++)
	for(uint64_t i=elemsLowerBound; i<localCnt && (cntOutputBuffer<elemsToSort); i++)
	{

		if(keyValPairs[i].key>=rangeMin && keyValPairs[i].key<rangeMax)
		{
			// bufferToSort[cntOutputBuffer].key = dev_pointIDKey[i];
			// bufferToSort[cntOutputBuffer].val = dev_pointInDistValue[i];
			cntOutputBuffer++;

			//if the flag has not been set
			if(flagElemsLowerBound==0){
				copiedContiguousElemsLowerBound = i;
			}
		}
		
		//if an element exceeds rangeMax then we need to set the flag
		//because we need to go back and start from that index on a future iteration
		if(flagElemsLowerBound!=1 && keyValPairs[i].key>=rangeMax){
			flagElemsLowerBound = 1;
		}

		// update upper bound
		elemsUpperBound = i;
	}

	elemsUpperBound += 1;

	// printf("\nIn function copiedContiguousElemsLowerBound: %lu", copiedContiguousElemsLowerBound);

	return copiedContiguousElemsLowerBound;
}


//This variant counts the number of elements that have already been copied in unsortedBuffer
// so that it does not need to be re-scanned 
uint64_t sequentialCopyToBufferOutputLowerBound(keyValPair * bufferToSort, 	unsigned int * dev_pointIDKey,
	unsigned int * dev_pointInDistValue, uint64_t elemsToSort, unsigned long long int localCnt,
	unsigned int rangeMin, unsigned int rangeMax, uint64_t elemsLowerBound)
{
	uint64_t cntOutputBuffer = 0;
	
	bool flagElemsLowerBound = 0;
	uint64_t copiedContiguousElemsLowerBound = elemsLowerBound;
	
	// for(uint64_t i=0; i<(localCnt) && (cntOutputBuffer<elemsToSort); i++)
	for(uint64_t i=elemsLowerBound; i<localCnt && (cntOutputBuffer<elemsToSort); i++)
	{

		if(dev_pointIDKey[i]>=rangeMin && dev_pointIDKey[i]<rangeMax)
		{
			bufferToSort[cntOutputBuffer].key = dev_pointIDKey[i];
			bufferToSort[cntOutputBuffer].val = dev_pointInDistValue[i];
			cntOutputBuffer++;

			//if the flag has not been set
			if(flagElemsLowerBound==0){
				copiedContiguousElemsLowerBound = i;
			}
		}
		
		//if an element exceeds rangeMax then we need to set the flag
		//because we need to go back and start from that index on a future iteration
		if(flagElemsLowerBound!=1 && dev_pointIDKey[i]>=rangeMax){
			flagElemsLowerBound = 1;
		}



	}

	// printf("\nIn function copiedContiguousElemsLowerBound: %lu", copiedContiguousElemsLowerBound);

	return copiedContiguousElemsLowerBound;
}

void parallelCopyToBuffer(keyValPair * bufferToSort, unsigned int * dev_pointIDKey,
	unsigned int * dev_pointInDistValue, uint64_t elemsToSort, unsigned long long int localCnt,
	unsigned int rangeMin, unsigned int rangeMax, uint64_t elemsLowerBound)
{
	uint64_t cntOutputBuffer = 0;
	
	// for(uint64_t i=0; i<(localCnt) && (cntOutputBuffer<elemsToSort); i++)
	for(uint64_t i=elemsLowerBound; i<localCnt; i++)
	{
		if( cntOutputBuffer<elemsToSort ) {
			if(dev_pointIDKey[i]>=rangeMin && dev_pointIDKey[i]<rangeMax)
			{	
				bufferToSort[cntOutputBuffer].key = dev_pointIDKey[i];
				bufferToSort[cntOutputBuffer].val = dev_pointInDistValue[i];
				cntOutputBuffer++;
			}
		}
	}
}




// version of probe and sort that works with QUERYREORDER=1
void probeAndSort(
	keyValPair * keyValPairs,
	unsigned long long int * cnt,
	const unsigned long long int maxUnsortedNELEMS, 
	const unsigned int numElemsCompletedArray,
	cudaEvent_t * kernelStop,
	unordered_map<unsigned int, vector<struct keyValBin>> * keyBinsMap,
	double * totalSortTime
	)
{
	// set bounds
	uint64_t lowerBound = 0;
	uint64_t upperBound = 0;

	// get the number of elements in one page
	unsigned int elemsPerPage = ((PAGESIZE) * (1024)) / sizeof(unsigned int);
	printf("\nelemsPerPage: %u", elemsPerPage);

	// keep track of the number of pages sorted
	unsigned long long int pagesSorted = 0;

	unsigned long long int localCnt;
	unsigned long long int localPagesIdx;

	

	// loop while kernel is not complete
	while( cudaEventQuery(*kernelStop) == cudaErrorNotReady )
	{
		// sleep for SLEEPSEC seconds
		usleep(SLEEPSEC*1000000);

		// read cnt
		localCnt = *cnt;
		printf("\nNum elems generated in array on GPU: %llu", localCnt);

		// convert count to pages
		localPagesIdx = localCnt  / elemsPerPage;
		printf("\nPage index being updated on GPU: %llu", localPagesIdx);

		// sort up to localPageCnt each time

		// set bounds
		lowerBound = pagesSorted * elemsPerPage;
		upperBound = localPagesIdx * elemsPerPage;

		// prefetch
		#if PREFETCH==1
		printf("\nPrefetching...");
		cudaMemPrefetchAsync(keyValPairs+lowerBound, sizeof(keyValPair)*(upperBound-lowerBound), kCpuMemLocation, 0);
		#endif

		// sort
		printf("\nSorting %lu elements in bounds [%lu, %lu)", (upperBound-lowerBound), lowerBound, upperBound);
		double tstart_sort = omp_get_wtime();
		__gnu_parallel::sort(keyValPairs+lowerBound, keyValPairs+upperBound, compareKeyValPairs);
		double tend_sort = omp_get_wtime();
		printf("\nSort time: %f", tend_sort-tstart_sort);

		*totalSortTime += tend_sort-tstart_sort;

		// create bins and add to map
		createBinsAndAddToMap( keyValPairs, lowerBound, upperBound, keyBinsMap );

		// update pages sorted count
		pagesSorted = localPagesIdx;
	}

	// read cnt
	localCnt = *cnt;
	printf("\n[Leftover] Num elems generated in array on GPU: %llu", localCnt);

	// sort up to localPageCnt each time

	// set bounds
	lowerBound = pagesSorted * elemsPerPage;
	upperBound = localCnt;

	// prefetch
	#if PREFETCH==1
	printf("\nPrefetching...");
	cudaMemPrefetchAsync(keyValPairs+lowerBound, sizeof(keyValPair)*(upperBound-lowerBound), kCpuMemLocation, 0);
	#endif

	// sort
	printf("\n[Leftover] Sorting %lu elements in bounds [%lu, %lu)", (upperBound-lowerBound), lowerBound, upperBound);
	double tstart_sort = omp_get_wtime();
	__gnu_parallel::sort(keyValPairs+lowerBound, keyValPairs+upperBound, compareKeyValPairs);
	double tend_sort = omp_get_wtime();
	printf("\n[Leftover] Sort time: %f", tend_sort-tstart_sort);

	*totalSortTime += tend_sort-tstart_sort;

	// create bins and add to map
	createBinsAndAddToMap( keyValPairs, lowerBound, upperBound, keyBinsMap );
}

void createBinsAndAddToMap( keyValPair * keyValPairs, uint64_t& lowerBound, uint64_t& upperBound, unordered_map<unsigned int, vector<struct keyValBin>> * keyBinsMap ) {
	int numBinningThreads = 8;
	int threadSectionSize = ((upperBound-lowerBound) / numBinningThreads);

	#pragma omp parallel for num_threads(numBinningThreads)
	for(unsigned int i=0; i<numBinningThreads; i++)
	{
		uint64_t threadLowerBound = lowerBound + (threadSectionSize * i);
		uint64_t threadUpperBound = (i < numBinningThreads-1) ? threadLowerBound + threadSectionSize : upperBound;

		unordered_map<unsigned int, vector<struct keyValBin>> localMap;

		keyValBin tempBin;

		// create bins
		tempBin.indexmin = threadLowerBound;
		unsigned int currentKey = keyValPairs[threadLowerBound].key;
		for( unsigned long long int j=threadLowerBound+1; j<threadUpperBound; j++ )
		{
			if( keyValPairs[j].key != currentKey )
			{
				tempBin.indexmax = j;

				// add bin to local map
				localMap[currentKey].push_back(tempBin);

				tempBin.indexmin = j;
				currentKey = keyValPairs[j].key;
			}
		}
		// final bin
		tempBin.indexmax = threadUpperBound;

		// add bin to local map
		localMap[currentKey].push_back(tempBin);

		// Merge into global map
		#pragma omp critical
		{
			for (auto &p : localMap) {
				auto &vec = (*keyBinsMap)[p.first];
				vec.insert(vec.end(), p.second.begin(), p.second.end());
			}
		}
	}
}

void moveKeyBinsToNeighborTable( const unsigned int DBSIZE, keyValPair * dev_keyValPairs, unordered_map<unsigned int, vector<struct keyValBin>> * keyBinsMap, struct neighborTableLookup * neighborTable, int * pointersToNeighbors )
{
	#if STAMP==0

	// get then total number of values for each key
	int * keyCountsMap = new int[DBSIZE];
	#pragma omp parallel for num_threads(8)
	for (unsigned int i=0; i<DBSIZE; i++){
		int count = 0;

		for(auto bin : (*keyBinsMap)[i] ) {
			count += bin.indexmax - bin.indexmin;
			if( i ==0 ) {
				fprintf(stderr, "\nbin [%llu, %llu)", bin.indexmin, bin.indexmax);
			}
		}
		
		keyCountsMap[i] = count;
	}

	// get the index starts for the values of each key
	int * indexStarts = new int[DBSIZE+1];
	int indexCount = 0;
	indexStarts[0] = indexCount;
	for (unsigned int i=1; i<DBSIZE; i++){
		indexCount += keyCountsMap[i-1];
		indexStarts[i] = indexCount;
	}
	indexStarts[DBSIZE] = indexCount + keyCountsMap[DBSIZE-1];

	// move values to array and fill neighbor table
	#pragma omp parallel for num_threads(8)
	for (unsigned int i=0; i<DBSIZE; i++){
		unsigned int workingIdx = indexStarts[i];
		for(auto bin : (*keyBinsMap)[i]) {
			for (unsigned long long j = bin.indexmin; j < bin.indexmax; j++) {
				pointersToNeighbors[workingIdx] = dev_keyValPairs[j].val;
				workingIdx += 1;
			}
		}

		neighborTable[i].pointID=i;
		neighborTable[i].indexmin=indexStarts[i];
		neighborTable[i].indexmax=indexStarts[i+1] - 1; //index max in neighbortable is inclusive
	
		//update the pointer to the data array for the values
		neighborTable[i].dataPtr=pointersToNeighbors;
	}
	
	delete[] keyCountsMap;
	delete[] indexStarts;
	#endif
}