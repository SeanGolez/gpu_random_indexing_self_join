#include "gpu_sort.h"

#include <pthread.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "structs.h"
#include "GPU.h"
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "omp.h"
#include <queue>
#include <unistd.h>
#include <parallel/algorithm>

#include "tbb/tbb.h"
#include "tbb/concurrent_queue.h"
#include "tbb/concurrent_vector.h"
#include "tbb/mutex.h"

//thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h> //for streams for thrust (added with Thrust v1.8)

using namespace std;

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

// iterative GPU sort, concurrently merging on the cpu
void gpuRadixSortAndCpuMerge(unsigned int * unorderedPointIDKey, unsigned int * unorderedPointInDistValue, unsigned int * orderedPointIDKey, unsigned int * orderedPointInDistValue, unsigned long long int * N ) {
	cudaError_t errCode;

	// get free bytes on gpu
    // size_t free_bytes, total_bytes;
	// cudaMemGetInfo(&free_bytes, &total_bytes);
	// fprintf(stderr, "\nFree memory: %zu bytes, Total memory: %zu bytes\n", free_bytes, total_bytes);

    // calculate split
	unsigned int NUMBATCHES = ceil( (1.0*(*N)) / (1.0*GPUBUFFERSIZE) );
	
	int NUMGPU = 1;
	int STREAMSPERGPU = GPUSTREAMS;
	// int STREAMSPERGPU = 1;

	printf("\nNum batches: %d", NUMBATCHES);

	//array for recordkeeping for the batches that have been completed for pipelining merging pairs
	//before multiway
	bool batchCompleteArr[NUMBATCHES];
	for (int i=0; i<NUMBATCHES; i++)
	{
		batchCompleteArr[i]=0;
	}

	tbb::concurrent_queue<rangeorder> rangeQueue;
	int numBatchesMerged=0; //variable used for determining when to finish merging

	int gpustart=0;
	//only have the CPU worksteal a single batch at the beginning if
	//there's more than 3 batches per GPU
	if (NUMBATCHES>3)
	{
		gpustart=1;
	}

	cudaStream_t streams[STREAMSPERGPU];
	//create stream for the device
	for (int i=0; i<STREAMSPERGPU; i++)
	{
		cudaStreamCreate(&streams[i]);
	}


	// PINNED MEMORY TO COPY TO THE GPU
	unsigned int * batchToGPUPointIDKey;
	errCode= cudaMallocHost((void **) &batchToGPUPointIDKey, sizeof(unsigned int)*GPUBUFFERSIZE*STREAMSPERGPU);
	if(errCode != cudaSuccess) {
	cout << "Pinned mem alloc. memory on device got an error: " << errCode << endl; //2 means not enough memory
	}
	unsigned int * batchToGPUPointInDistValue;
	errCode= cudaMallocHost((void **) &batchToGPUPointInDistValue, sizeof(unsigned int)*GPUBUFFERSIZE*STREAMSPERGPU);
	if(errCode != cudaSuccess) {
	cout << "Pinned mem alloc. memory on device got an error: " << errCode << endl; //2 means not enough memory
	}

	// PINNED MEMORY TO COPY FROM THE GPU
	unsigned int * batchResultsPointIDKey;
	errCode= cudaMallocHost((void **) &batchResultsPointIDKey, sizeof(unsigned int)*GPUBUFFERSIZE*STREAMSPERGPU);
	if(errCode != cudaSuccess) {
	cout << "Pinned mem alloc. memory on device got an error: " << errCode << endl; //2 means not enough memory
	}
	unsigned int * batchResultsPointInDistValue;
	errCode= cudaMallocHost((void **) &batchResultsPointInDistValue, sizeof(unsigned int)*GPUBUFFERSIZE*STREAMSPERGPU);
	if(errCode != cudaSuccess) {
	cout << "Pinned mem alloc. memory on device got an error: " << errCode << endl; //2 means not enough memory
	}

	// allocate device arrays that will be sorted
	unsigned int * dev_pointIdKey;
	errCode= cudaMalloc((void **) &dev_pointIdKey, sizeof(unsigned int)*GPUBUFFERSIZE*STREAMSPERGPU);
	if(errCode != cudaSuccess) {
	cout << "Pinned mem alloc. memory on device got an error: " << errCode << endl; //2 means not enough memory	
	}

	unsigned int * dev_pointInDistValue;
	errCode= cudaMalloc((void **) &dev_pointInDistValue, sizeof(unsigned int)*GPUBUFFERSIZE*STREAMSPERGPU);
	if(errCode != cudaSuccess) {
	cout << "Pinned mem alloc. memory on device got an error: " << errCode << endl; //2 means not enough memory
	}

	// stop pipeline merge once all batches are sorted
	int sortDone =  0 - gpustart;

	double tstart_gpuSort = omp_get_wtime();

    #pragma omp parallel

	#pragma omp sections
	{

	//if work stealing using the CPU
	#pragma omp section
	{
		printf("\ncpu worksteal tid: %d", omp_get_thread_num());	
		//worksteal only if it has been calculated that a batch should be processed by the CPU
		// note: only executes when num batches > 3
		if (gpustart==1){
			uint64_t batchSize = GPUBUFFERSIZE;

			double tstartworksteal=omp_get_wtime();
			printf("\ncpu worksteal tid: %d executing/stealing first batch from the GPU", omp_get_thread_num());
			workStealSortingABatch(unorderedPointIDKey, unorderedPointInDistValue, orderedPointIDKey, orderedPointInDistValue, batchSize);
			double tendworksteal=omp_get_wtime();
			printf("\ncpu worksteal, time (since start): %f",tendworksteal - tstartworksteal);
			// t_cpuworksteal.push_back(tstartworksteal - tstartgpu);
			// t_cpuworksteal.push_back(tendworksteal - tstartgpu);

			//for merging v2 -- by ranges
			#pragma omp critical
			{
			struct rangeorder tmp;
			tmp.rangelower=0;
			tmp.rangeupper=0+batchSize;
			rangeQueue.push(tmp);

			batchCompleteArr[0]=1;

			sortDone += 1;
			}

		}
	}	

	//GPU section	
	#pragma omp section	
	{
		printf("\ngpu tid: %d", omp_get_thread_num());
		// fprintf(stderr,"\ngpu tid: %d", omp_get_thread_num());

		double timetestmaingpuloopstart=omp_get_wtime();

		//ordered: need to execute in order, because we want to merge in order
		//but we want to hide mem copies between batches which is a significant overhead
		#pragma omp parallel for num_threads(STREAMSPERGPU * NUMGPU) ordered schedule(static,1) // reduction(+:transferdatatime, gpusorttime, transferresulttime, pinnedcopydatatime)
		for (int i=gpustart; i<NUMBATCHES; i++)
		{
			uint64_t batchSize = GPUBUFFERSIZE;
			// check if last batch
			if( i == NUMBATCHES - 1 ) {
				batchSize = (*N) % GPUBUFFERSIZE;
			}

			int gpuid=i%NUMGPU;
			int streamid=0;

			//if 1 GPU then need to alternate streams
			// if (NUMGPU==1)
			// {
			streamid=i%STREAMSPERGPU;	
			// }
			
			// else
			// {
			// streamid=(i/STREAMSPERGPU)%STREAMSPERGPU; //this works for 2 GPUs 1 stream each, and 2 GPUs with 2 streams
			// 	streamid=(i/NUMGPU)%STREAMSPERGPU; //works on up to 3 streams and2 GPUs tested so far
			// }		

			int tid=omp_get_thread_num();

			// cudaSetDevice(gpuid);

			

			uint64_t offset=(uint64_t)i*GPUBUFFERSIZE;
			// printf("\nOffset: %lu", offset);

			// printf("\n\n\n");
			// for (int aaa=0; aaa<BATCHSIZE; aaa++)
			// {
			// 	printf("%f, ",array[offset+aaa]);	
			// }

			printf("\nExecuting batch # %d, tid: %d, on gpu: %d, stream: %d",i,tid,gpuid,streamid);
			// fprintf(stderr,"\nExecuting batch # %d, tid: %d, on gpu: %d, stream: %d",i,tid,gpuid,streamid);

			// printf("\nFirst   location tid: %d: %d",tid, (gpuid*GPUBUFFERSIZE)+(streamid*GPUBUFFERSIZE));

			// double t_start_transfer_data=omp_get_wtime();
			//copy array to the device


			// *****NOTE: This is not using the pinned memory buffers incrementally (assumes that the pinned memory buffers in large enough for each batch)*****
			printf("\nBatch size: %lu", batchSize);

			#pragma omp parallel for num_threads(4) shared(batchToGPUPointIDKey, batchResultsPointInDistValue, unorderedPointIDKey, unorderedPointInDistValue) 
			for (uint64_t z=0; z<batchSize; z++)
			{
				// uint64_t idx=(gpuid*GPUBUFFERSIZE)+(streamid*GPUBUFFERSIZE)+z;
				uint64_t idx=(streamid*batchSize)+z;
																																																											
				batchToGPUPointIDKey[idx]=unorderedPointIDKey[offset+z];
				batchResultsPointInDistValue[idx]=unorderedPointInDistValue[offset+z];
			}
			
			//copy from pinned to GPU
			gpuErrchk(cudaMemcpyAsync(dev_pointIdKey+(streamid*GPUBUFFERSIZE), batchToGPUPointIDKey+(streamid*GPUBUFFERSIZE), batchSize*sizeof(unsigned int), cudaMemcpyHostToDevice, streams[streamid]));
			gpuErrchk(cudaMemcpyAsync(dev_pointInDistValue+(streamid*GPUBUFFERSIZE), batchResultsPointInDistValue+(streamid*GPUBUFFERSIZE), batchSize*sizeof(unsigned int), cudaMemcpyHostToDevice, streams[streamid]));  
			cudaStreamSynchronize(streams[streamid]);



			// double t_end_transfer_data=omp_get_wtime();
			// transferdatatime+=t_end_transfer_data - t_start_transfer_data; 	

			// cudaStreamSynchronize(streams[gpuid][streamid]);
			
			
			

			// streamMutex[gpuid].lock();
			// double t_start_sort_gpu=omp_get_wtime();
			printf("\nSorting [%lu, %lu)...", offset, offset+batchSize);
			try{
			thrust::sort_by_key(thrust::cuda::par.on(streams[streamid]), dev_pointIdKey+(streamid*batchSize), dev_pointIdKey+(streamid*batchSize)+batchSize, dev_pointInDistValue+(streamid*batchSize));	
				// thrust::sort(thrust::cuda::par.on(testStream), dev_array_ptr[gpuid]+(streamid*BATCHSIZE), dev_array_ptr[gpuid]+(streamid*BATCHSIZE)+ BATCHSIZE);	
			}
			catch(std::bad_alloc &e)
				{
				std::cerr << "Ran out of memory while sorting" << std::endl;
				exit(-1);
			}
			// double t_end_sort_gpu=omp_get_wtime(); 
			// gpusorttime+=t_end_sort_gpu - t_start_sort_gpu;
			// streamMutex[gpuid].unlock();

			// cudaThreadSynchronize();

			//DO NOT NEED STREAM SYNC HERE BECAUSE THE MEMCPYASYNC WILL QUEUE THE COPIES AFTER THE SORT EXECUTED
			// cudaStreamSynchronize(streams[streamid]);
			printf("\nSorting Done.");

			
			// cudaStreamSynchronize(streams[gpuid][streamid]);
			//copy result back directly into the big result array that gets merged
			//times with Async are not going to be correct
			// double t_start_transfer_result=omp_get_wtime();

			// *****NOTE: This is not using the pinned memory buffers incrementally (assumes that the pinned memory buffers in large enough for each batch)*****

			//original
			gpuErrchk(cudaMemcpyAsync(batchResultsPointIDKey+(streamid*GPUBUFFERSIZE), thrust::raw_pointer_cast(dev_pointIdKey+(streamid*GPUBUFFERSIZE)), batchSize*sizeof(unsigned int), cudaMemcpyDeviceToHost, streams[streamid]));
			gpuErrchk(cudaMemcpyAsync(batchResultsPointInDistValue+(streamid*GPUBUFFERSIZE), thrust::raw_pointer_cast(dev_pointInDistValue+(streamid*GPUBUFFERSIZE)), batchSize*sizeof(unsigned int), cudaMemcpyDeviceToHost, streams[streamid])); 
			cudaStreamSynchronize(streams[streamid]);																
				
			//copy to the pageable memory
			//original-memcpy
			//slower by a factor of ~2.5 in comparison to copying element-wise in parallel!

			#pragma omp parallel for num_threads(4) shared (orderedPointIDKey, orderedPointInDistValue, batchResultsPointIDKey, batchResultsPointInDistValue)
			for (uint64_t y=0; y<batchSize; y++)
			{
				// uint64_t idx=(gpuid*GPUBUFFERSIZE)+(streamid*GPUBUFFERSIZE)+y;
				uint64_t idx=(streamid*batchSize)+y;
				orderedPointIDKey[offset+y]=batchResultsPointIDKey[idx];
				orderedPointInDistValue[offset+y]=batchResultsPointInDistValue[idx];
			}
			
			//memmove- slow/same as memcpy
			// std::memmove(resultsFromBatches+x, resultsFromBatchesPinned+(gpuid*NUMGPU*GPUBUFFERSIZE)+(streamid*GPUBUFFERSIZE), GPUBUFFERSIZE*sizeof(double));
			//HostToHost version -- SLOW dont use
			// cudaMemcpyAsync(resultsFromBatches+x,resultsFromBatchesPinned+(gpuid*NUMGPU*GPUBUFFERSIZE)+(streamid*GPUBUFFERSIZE), GPUBUFFERSIZE*sizeof(double), cudaMemcpyHostToHost, streams[gpuid][streamid]);
			// printf("\nDtoH: %d",x);
		

			// double t_end_transfer_result=omp_get_wtime();
			// transferresulttime+=t_end_transfer_result - t_start_transfer_result;
			// printf("\nTime to transfer batch %d: %f", i, t_end_transfer_result - t_start_transfer_result);
			

			//commented this, may not need it with the element-wise assignment above
			// cudaStreamSynchronize(streams[streamid]);
			
			


			

			#pragma omp critical
			{
				//for merging v2 -- by ranges
				struct rangeorder tmp;
				tmp.rangelower=offset;
				tmp.rangeupper=offset+batchSize;
				rangeQueue.push(tmp);

				//doesn't need to be in crit section but 
				//should be done after the rangeQueue insert
				batchCompleteArr[i]=1;
			}//end critical
		


		
		// double t_individualbatch=omp_get_wtime();
		// printf("\nTime component batch # %d, tid: %d, on gpu: %d, time finished (from start): %f",i,tid,gpuid,t_individualbatch- tstartgpu);
		
		//time start of batch
		// t_batch[gpuid].push_back(t_start_transfer_data - tstartgpu);
		// time end of batch
		// t_batch[gpuid].push_back(t_individualbatch - tstartgpu);
		

		
		
		

		} //end loop

		#pragma omp critical
		{
			sortDone += 1;

		}


		// double timetestmaingpuloopend=omp_get_wtime();
		// fprintf(stderr, "\nGPU thread %d finished generating batches, total time in main gpu loop: %f",omp_get_thread_num(),timetestmaingpuloopend-timetestmaingpuloopstart);
		// printf("\nGPU thread %d finished generating batches, total time in main gpu loop: %f, time from start: %f",omp_get_thread_num(),timetestmaingpuloopend-timetestmaingpuloopstart, timetestmaingpuloopend- tstartgpu);

		double tend_gpuSort = omp_get_wtime();
		printf("\nGPU sort portion time: %f", (tend_gpuSort - tstart_gpuSort));

	} //end GPU section

	//merging v3:
	//pipeline before multiway:
	//Heuristic: only merge the lists of batchsize, otherwise you get the problem with small and large
	//lists
	//total merges: floor((# batches-1)/2)

	#pragma omp section
	{

		printf("\nmerge tid: %d", omp_get_thread_num());
		// fprintf(stderr,"\nmerge tid: %d", omp_get_thread_num());
		
		//Number of batches to merge using the heuristic
		//1 GPU
		int NUMBATCHESTOMERGE=0;
		if (NUMGPU==1)
		{
			NUMBATCHESTOMERGE=(int)floor(((double)NUMBATCHES-1.0)/2.0);
		}
		else
		//2 GPUs	
		{
			NUMBATCHESTOMERGE=(int)floor(((double)NUMBATCHES)/(2.0*NUMGPU));
		}
		printf("\n Multiway: Num batches to be merged during pipeline (GPUs: %d): %d", NUMGPU, NUMBATCHESTOMERGE);
		
		//PIPELINING HERE: version 2 -- allow multiple cpu threads to merge
		//pipeline the merge
		
		//EACH THREAD NEEDS ITS OWN RANGES HERE
		uint64_t lower1, upper1,lower2, upper2;

		//each thread needs its own merge flag -- if we do multiple merge consumers
		bool mergeflag=0;

		while(numBatchesMerged<NUMBATCHESTOMERGE && sortDone != 1){
			mergeflag=0;
			//find if there are at least 2 items to merge that are each BATCHSIZE elems
			mergeflag=getMergeRangesForMergingPairsForMultiway(batchCompleteArr, &rangeQueue, N, NUMBATCHES,  &lower1, &upper1, &lower2, &upper2);
			//printf("\nmergeflag: %d, Num batches merged: %d",mergeflag, numBatchesMerged);

			if(mergeflag==1){
				printf("\nMerging [%lu, %lu) and [%lu, %lu)...", lower1, upper1,lower2, upper2);

				//inc num batches merged before merging so that the other merge thread (if we add one later with pipelined/multiway) can exit the loop if its on the second last one	
				#pragma omp atomic
				numBatchesMerged++;	


				// fprintf(stderr,"\nPipeline merge (for multiway later): tid: %d, merging: lower: %lu, %lu, upper: %lu, %lu",omp_get_thread_num(), lower1, upper1, lower2, upper2);
				
				

				
				


				// double merge_start=omp_get_wtime();	
				
				mergeConsumerWithRanges(orderedPointIDKey, orderedPointInDistValue, lower1, upper1, lower2, upper2);
				
				// double merge_end=omp_get_wtime();
				// fprintf(stderr, "\ntid: %d, time to merge batch: %f (Total merged: %d)", omp_get_thread_num(),merge_end - merge_start,numBatchesMerged);
				// printf("\ntid: %d, time to merge batch: %f (Total merged: %d), Time started from start: %f, Time finished from start: %f", omp_get_thread_num(),merge_end - merge_start,numBatchesMerged,merge_start-tstartgpu ,merge_end-tstartgpu);
				//Add merged result back to queue:
				struct rangeorder tmpmerged;
				tmpmerged.rangelower=lower1;
				tmpmerged.rangeupper=upper2;
				rangeQueue.push(tmpmerged);

				// pragma omp atomic
				// mergetime+=merge_end - merge_start;
				
				// printf("\nTime component tid: %d, merging: lower: %lu, %lu, upper: %lu, %lu, time finished (from start): %f",omp_get_thread_num(), lower1, upper1, lower2, upper2, merge_end-tstartgpu );
				//the start time of the merge
				// t_merge_1.push_back(merge_start-tstartgpu);
				//end end time of the merge
				// t_merge_1.push_back(merge_end-tstartgpu);

				printf("\nMerging Done.");

			} //end if mergeflag

			


		}

		fprintf(stderr, "\ntid: %d finished merging (batches merged before multiway: %d)", omp_get_thread_num(),numBatchesMerged);

	}
	}

	/*
	int sizeQueue= rangeQueue.unsafe_size();
	for (int j=0; j<sizeQueue; j++)
	{

		struct rangeorder tmp1;
		rangeQueue.try_pop(tmp1);
		printf("\nRange queue: %lu, %lu",tmp1.rangelower, tmp1.rangeupper);
		rangeQueue.push(tmp1);
	}
	*/

	printf("\nMultiway merging -- after some batches pipelined");


	//if theres only 1 batch do not merge! it's already sorted
	if ((NUMBATCHES-1)!=0)
	{
	    multiwayMergeBatchesAfterPipeline(N, &rangeQueue, orderedPointIDKey, orderedPointInDistValue);
	}
}

//this one merges give two sets of ranges of the batches, e.g., [50,100) [100,150)
void mergeConsumerWithRanges(unsigned int * orderedPointIDKey, unsigned int * orderedPointInDistValue, uint64_t lower1, uint64_t upper1,uint64_t lower2, uint64_t upper2)
{
	

	
	if (upper1!=lower2)
	fprintf(stderr,"\nerror, the two ranges do not have the same value between the upper1/lower2: %lu,%lu ",upper1,lower2);	

	
	
	//original in place:
	// std::inplace_merge(resultsFromBatches+lower1,resultsFromBatches+(upper1),resultsFromBatches+(upper2));
	
	//tmp vector:
	//not in place:
	uint64_t distance = (upper2-lower1);
	// printf("\ndistance: %lu", distance);

	keyValPair * tmp = new keyValPair[distance];
	keyValPair * sortedKeyValPairs = new keyValPair[distance];

	#pragma omp parallel for num_threads(NCOPYTHREADS)
	for( unsigned long long int i=0; i < distance; i++ ) {
		sortedKeyValPairs[i].key = orderedPointIDKey[i + lower1];
		sortedKeyValPairs[i].val = orderedPointInDistValue[i + lower1];
	}

	__gnu_parallel::merge(sortedKeyValPairs, sortedKeyValPairs + (upper1 - lower1), sortedKeyValPairs + (upper1 - lower1), sortedKeyValPairs + distance, tmp, compareKeyValPairs);		

	#pragma omp parallel for num_threads(NCOPYTHREADS)
	for( unsigned long long int i=0; i < distance; i++ ) {
		orderedPointIDKey[i + lower1] = tmp[i].key;
		orderedPointInDistValue[i + lower1] = tmp[i].val;
	}
	delete[] tmp;
	delete[] sortedKeyValPairs;
}

void workStealSortingABatch(unsigned int * unorderedPointIDKey, unsigned int * unorderedPointInDistValue, unsigned int * orderedPointIDKey, unsigned int * orderedPointInDistValue, uint64_t batchsize)
{
	keyValPair * sortedKeyValPairs = new keyValPair[batchsize];
	#pragma omp parallel for num_threads(NCOPYTHREADS)
	for( unsigned long long int i=0; i < batchsize; i++ ) {
		sortedKeyValPairs[i].key = unorderedPointIDKey[i];
		sortedKeyValPairs[i].val = unorderedPointInDistValue[i];
	}

	__gnu_parallel::sort(sortedKeyValPairs, sortedKeyValPairs+batchsize, compareKeyValPairs);

	#pragma omp parallel for num_threads(NCOPYTHREADS)
	for( unsigned long long int i=0; i < batchsize; i++ ) {
		orderedPointIDKey[i] = sortedKeyValPairs[i].key;
		orderedPointInDistValue[i] = sortedKeyValPairs[i].val;
	}
	delete[] sortedKeyValPairs;
}

//looks at the array of batches that have been completed and returns the 
//
bool getMergeRangesForMergingPairsForMultiway(bool * batchCompleteArr, tbb::concurrent_queue<rangeorder> * rangeQueue, unsigned long long int *N, int NUMBATCHES,  uint64_t * lower1, uint64_t * upper1, uint64_t * lower2, uint64_t * upper2)
{

	for (int i=0; i<NUMBATCHES; i+=2)
	{
		int firstBatchIdx = i;
		int secondBatchIdx = i + 1;

		if ( (batchCompleteArr[firstBatchIdx]==1 && batchCompleteArr[secondBatchIdx]==1) )
		{
			int cntRemoved=0;
			*lower1=(uint64_t)firstBatchIdx*GPUBUFFERSIZE;
			*upper1=(uint64_t)(firstBatchIdx+1)*GPUBUFFERSIZE;
			*lower2=(uint64_t)(secondBatchIdx)*GPUBUFFERSIZE;
			*upper2=(uint64_t)(secondBatchIdx+1)*GPUBUFFERSIZE;

			batchCompleteArr[firstBatchIdx]=0;
			batchCompleteArr[secondBatchIdx]=0;

			//NEED TO REMOVE THESE FROM THE RANGEQUEUE (the full range will be added later)
			//need to get the fixed size of the queue because you will add and remove inside the loop
			int sizeQueue=(*rangeQueue).unsafe_size();
			for (int j=0; j<sizeQueue; j++)
			{

				struct rangeorder tmp1;
				rangeQueue->try_pop(tmp1);
				// printf("\nRange queue: %lu, %lu",tmp1.rangelower, tmp1.rangeupper);

				if ((tmp1.rangelower==(*lower1))&&(tmp1.rangeupper==(*upper1)))
				{
					// printf("\nremoved (first): %lu, %lu", tmp1.rangelower, tmp1.rangeupper);
					cntRemoved++;
				}
				else if ((tmp1.rangelower==(*lower2))&&(tmp1.rangeupper==(*upper2)))
				{
					// printf("\nremoved (second): %lu, %lu", tmp1.rangelower, tmp1.rangeupper);
					cntRemoved++;
				}
				else
				{
					rangeQueue->push(tmp1);
					// printf("\n%lu, %lu", tmp1.rangelower, tmp1.rangeupper);
				}
			}

			if (cntRemoved!=2)
			printf("\n\n\nError: found 2 batches to merge but could not remove both from rangeQueue: cnt removed: %d\n\n",cntRemoved);
			
			return 1;
			
		}
	}
		
return 0;

}


//Merge after some of the batches were pipelined
void multiwayMergeBatchesAfterPipeline(unsigned long long * N, tbb::concurrent_queue<rangeorder> * rangeQueue, unsigned int * orderedPointIDKey, unsigned int * orderedPointInDistValue)
{
    //temp vector for output
    // double * tmp;
    // tmp = new double[BATCHSIZE*(uint64_t)NUMBATCHES]; 
    // out_vect.reserve(BATCHSIZE*NUMBATCHES);

	keyValPair * tmp = new keyValPair[*N];

	keyValPair * sortedKeyValPairs = new keyValPair[*N];
	#pragma omp parallel for num_threads(NCOPYTHREADS)
	for( unsigned long long int i=0; i < *N; i++ ) {
		sortedKeyValPairs[i].key = orderedPointIDKey[i];
		sortedKeyValPairs[i].val = orderedPointInDistValue[i];
	}

	delete[] orderedPointIDKey;
	delete[] orderedPointInDistValue;

    std::vector<std::pair<keyValPair *, keyValPair*> > seqs;

    struct rangeorder rangesToMultiway[rangeQueue->unsafe_size()];

    int sizeQueue=rangeQueue->unsafe_size();

    for (uint64_t i=0; i<sizeQueue; i++)
    {
        struct rangeorder tmpRng;
        rangeQueue->try_pop(tmpRng);
        rangesToMultiway[i]=tmpRng;
        seqs.push_back(std::make_pair<keyValPair*,keyValPair* >(sortedKeyValPairs+(rangesToMultiway[i].rangelower),sortedKeyValPairs+(rangesToMultiway[i].rangeupper)));
    }

    // seqs.push_back(std::make_pair<double*,double* >(*resultsFromBatches+(rangeorder[i].rangelower),*resultsFromBatches+(rangeorder[i].rangeupper)));
    __gnu_parallel::multiway_merge(seqs.begin(), seqs.end(), tmp, *N, compareKeyValPairs, __gnu_parallel::parallel_tag());



    
    //old with local variable
    //std::copy(tmp,tmp+(BATCHSIZE*(uint64_t)NUMBATCHES),resultsFromBatches);

    //new copy with the buffer
    // std::copy(*tmpBuffer,*tmpBuffer+(BATCHSIZE*(uint64_t)NUMBATCHES),*resultsFromBatches);

    
    //swap pointers -- avoid copying
    // double *a= *resultsFromBatches;
    // *resultsFromBatches = *tmpBuffer;

    //delete a;

	orderedPointIDKey = new unsigned int[*N];
	orderedPointInDistValue = new unsigned int[*N];

	// swap pointer, then copy to arrays
	delete[] sortedKeyValPairs;
	sortedKeyValPairs = tmp;

	#pragma omp parallel for num_threads(NCOPYTHREADS)
	for( unsigned long long int i=0; i < *N; i++ ) {
		orderedPointIDKey[i] = sortedKeyValPairs[i].key;
		orderedPointInDistValue[i] = sortedKeyValPairs[i].val;
	}

	delete[] sortedKeyValPairs;

    return;
}