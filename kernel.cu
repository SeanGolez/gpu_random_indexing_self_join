#include "kernel.h"
#include "structs.h"
#include <math.h>	
#include <algorithm>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include "params.h"


__device__ void swap(unsigned int* a, unsigned int* b) {
	unsigned int temp = *a;
	*a = *b;
	*b= temp;
}

__device__ void sortCell(unsigned int* list, DTYPE* database, int length, int tid){
	bool odd=false;
	for(int i=0; i<length; i++) {
		for(int j=(tid*2)+(int)odd; j<length-1; j+=32) {
			if(database[list[j]*GPUNUMDIM] > database[list[j+1]*GPUNUMDIM]) {
				swap(&list[j], &list[j+1]);
			}
		}
		odd = !odd;
	}
}

__device__ void seqSortCell(unsigned int* list, DTYPE* database, int length){
	int min;
	int minIdx;

	for(int i=0; i<length-1; i++ ) {
		min = database[list[i]*GPUNUMDIM];
		minIdx=i;
		for(int j=i; j<length; i++) {
			if(database[list[j]*GPUNUMDIM] < min) {
				min = database[list[j]*GPUNUMDIM];
				minIdx = j;
			}
		}
		swap(&list[i], &list[minIdx]);
	}
}


__global__ void kernelSortPointsInCells(DTYPE* database, struct grid * index, unsigned int* indexLookupArr, unsigned int nNonEmptyCells) {
        int tid = threadIdx.x + (blockIdx.x*BLOCKSIZE);
        int warpId = tid/32;
        int totalWarps = (gridDim.x*BLOCKSIZE)/32;

	int sortDim=0;
	if(GPUNUMDIM > NUMINDEXEDDIM)
		sortDim = NUMINDEXEDDIM;
	

        for(int i=warpId; i<nNonEmptyCells; i+=totalWarps) {
		if(index[i].indexmin < index[i].indexmax) {
  	              sortCell(indexLookupArr+index[i].indexmin, database+sortDim, (index[i].indexmax-index[i].indexmin)+1, threadIdx.x%32);
		}
        }

}




__device__ uint64_t getLinearID_nDimensionsGPU(unsigned int * indexes, unsigned int * dimLen, unsigned int nDimensions) {
    uint64_t offset = 0;
	uint64_t multiplier = 1;
	for (int i = 0; i<nDimensions; i++){
  	offset += (uint64_t)indexes[i] * multiplier;
  	multiplier *= dimLen[i];
	}

	return offset;
}

__device__ void getNDimIndexesFromLinearIdxGPU(unsigned int * indexes, unsigned int * dimLen, unsigned int nDimensions, uint64_t linearId) {
	// do the process to get linear id but backwards
	for (int i=0; i<nDimensions; i++){
  	indexes[i] = linearId % dimLen[i];
	linearId /= dimLen[i];
	}
}

// Kernel for binary search
// Needed to get the index of the searched value
__device__ int binarySearch(uint64_t* d_array, unsigned int size, uint64_t value) {
    int low = 0;
    int high = size - 1;

    while (low <= high) {
        int mid = low + (high - low) / 2;

		// Check if x is present at mid
        if (d_array[mid] == value) {
            return mid;
		}
        if (d_array[mid] < value) {
			// Move to the right half
            low = mid + 1; 
        } else {
			// Move to the left half
            high = mid - 1;
        }
    }

	// return cell not found
	return -1;
}


//unique key array on the GPU
__global__ void kernelUniqueKeys(int * pointIDKey, unsigned int * N, int * uniqueKey, int * uniqueKeyPosition, unsigned int * cnt)
{
	int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 

	if (tid>=*N){
		return;
	}	

	if (tid==0)
	{
		unsigned int idx=atomicAdd(cnt,int(1));
		uniqueKey[idx]=pointIDKey[0];
		uniqueKeyPosition[idx]=0;
		return;
	
	}
	
	//All other threads, compare to previous value to the array and add
	
	if (pointIDKey[tid-1]!=pointIDKey[tid])
	{
	unsigned int idx=atomicAdd(cnt,int(1));
	uniqueKey[idx]=pointIDKey[tid];
	uniqueKeyPosition[idx]=tid;
	}
	
}





//This version is the same as the batch estimator
//One query point per GPU thread


// unsigned int *debug1, unsigned int *debug2 – ignore, debug values
// unsigned int *N – total GPU threads for the kernel  	
// unsigned int * offset -  This is to offset into every nth data point, e.g., every 100th point calculates its neighbors 
// unsigned int *batchNum - The batch number being executed, used to calculate the point being processed
// DTYPE* database – The points in the database as 1 array
// DTYPE* epsilon – distance threshold
// struct grid * index – each non-empty grid cell is one of these, stores the indices into indexLookupArray that coincide with the data points in the database that are in the cell
// unsigned int * indexLookupArr – array of the size of database, has the indices of the datapoints in the database stored contiguously for each grid cell. each grid index cell references this 	
// struct gridCellLookup * gridCellLookupArr, - lookup array to the grid cells, needed to find if a grid cell exists (this is binary searched). Maps the location of the non-empty grid cells in grid * index to their linearized (1-D) array
// DTYPE* minArr – The minimum “edge” of the grid in each dimension
// unsigned int * nCells –The total number of cells in each dimension (if all were indexed), can compute the spatial extent, with minArr[0]+nCells[0]*epsilon, in the 1st dimension
// unsigned int * cnt – the result set size 	
// unsigned int * nNonEmptyCells – the number of non-empty cells in total, this is the size of the gridCellLookupArr
// int * pointIDKey, int * pointInDistVal - result set to be sorted as key/value pairs

__global__ void kernelNDGridIndexGlobal(unsigned int *debug1, unsigned int *debug2, unsigned int *N,  
	unsigned int * offset, unsigned int * indexGroupOffset, DTYPE* database, DTYPE * epsilon, struct grid * index, unsigned int * indexLookupArr, 
	struct gridCellLookup * gridCellLookupArrStart, struct gridCellLookup * gridCellLookupArrStartEnd, DTYPE* minArr, unsigned int * nCells, 
	unsigned int * orderedIndexPntIDs, unsigned int * cnt, int * pointIDKey, int * pointInDistVal, CTYPE* workCounts)
{

unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE);

if (tid>=*N){
	return;
}

/*
//If reordering the queries by the amount of work
#if QUERYREORDER==1
//test refactoring no reordering
// unsigned int pointIdx=tid*(*offset)+(*batchNum);
// unsigned int pointOffset=(GPUNUMDIM)*(tid*(*offset)+(*batchNum));

//the point id in the dataset
unsigned int pointIdx=orderedQueryPntIDs[tid*(*offset)+(*batchNum)]; 
//The offset into the database, taking into consideration the length of each dimension
unsigned int pointOffset=(GPUNUMDIM)*pointIdx; 


#endif

//If standard execution without reordering the queries by the amount of work
#if QUERYREORDER==0
//the point id in the dataset
unsigned int pointIdx=tid*(*offset)+(*batchNum);
//The offset into the database, taking into consideration the length of each dimension
unsigned int pointOffset=(GPUNUMDIM)*pointIdx;
#endif
*/

//the point id in the dataset
unsigned int pointIdx=orderedIndexPntIDs[tid*(*offset) + (*indexGroupOffset)]; 
//The offset into the database, taking into consideration the length of each dimension
unsigned int pointOffset=(GPUNUMDIM)*pointIdx;



//make a local copy of the point
//Store query point in registers
DTYPE point[GPUNUMDIM];
for (int i=0; i<GPUNUMDIM; i++){
	point[i]=database[pointOffset+i];	
}

//calculate the coords of the Cell for the point
//and the min/max ranges in each dimension
unsigned int nDCellIDs[NUMINDEXEDDIM];
unsigned int nDMinCellIDs[NUMINDEXEDDIM];
unsigned int nDMaxCellIDs[NUMINDEXEDDIM];

for (unsigned int i=0; i<NUMINDEXEDDIM; i++){
	nDCellIDs[i]=(point[i]-minArr[i])/(*epsilon);
	nDMinCellIDs[i]=max(0,nDCellIDs[i]-1); //boundary conditions (don't go beyond cell 0)
	nDMaxCellIDs[i]=min(nCells[i]-1,nDCellIDs[i]+1); //boundary conditions (don't go beyond the maximum number of cells)

}
	
        unsigned int indexes[NUMINDEXEDDIM];
        unsigned int loopRng[NUMINDEXEDDIM];

#if STAMP

	for(int i=0; i<NUMINDEXEDDIM; i++) {
		indexes[i] = nDCellIDs[i];
	}
	evaluateCell(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr, point, cnt, pointIDKey, pointInDistVal, pointIdx, false, nDCellIDs, workCounts);

    #include "stamploops.h"

#elif LINEARSTAMP
	for(int i=0; i<NUMINDEXEDDIM; i++) {
		indexes[i] = nDCellIDs[i];
	}
	evaluateCell(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr, point, cnt, pointIDKey, pointInDistVal, pointIdx, false, nDCellIDs, workCounts);

	for (loopRng[0]=rangeFilteredCellIdsMin[0]; loopRng[0]<=rangeFilteredCellIdsMax[0]; loopRng[0]++)
	for (loopRng[1]=rangeFilteredCellIdsMin[1]; loopRng[1]<=rangeFilteredCellIdsMax[1]; loopRng[1]++)
	#include "kernelloops.h"					
	{ //beginning of loop body
	
	for (int x=0; x<NUMINDEXEDDIM; x++){
	indexes[x]=loopRng[x];	
	}
	if(getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM) > getLinearID_nDimensionsGPU(nDCellIDs, nCells, NUMINDEXEDDIM)) {
		evaluateCell(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr, point, cnt, pointIDKey, pointInDistVal, pointIdx, true, nDCellIDs, workCounts);
	}	
	} //end loop body
#else

	for (loopRng[0]=nDMinCellIDs[0]; loopRng[0]<=nDMaxCellIDs[0]; loopRng[0]++)
	for (loopRng[1]=nDMinCellIDs[1]; loopRng[1]<=nDMaxCellIDs[1]; loopRng[1]++)
	#include "kernelloops.h"					
	{ //beginning of loop body
	
	for (int x=0; x<NUMINDEXEDDIM; x++){
	indexes[x]=loopRng[x];	
	}
		evaluateCell(nCells, indexes, database, epsilon, index, indexLookupArr, gridCellLookupArrStart, gridCellLookupArrStartEnd, point, cnt, pointIDKey, pointInDistVal, pointIdx, false, nDCellIDs, workCounts);
	
	} //end loop body
#endif

}

__forceinline__ __device__ void evalPoint(unsigned int* indexLookupArr, int k, DTYPE* database, DTYPE * epsilon, DTYPE* point, 
	unsigned int* cnt, int* pointIDKey, int* pointInDistVal, int pointIdx, bool differentCell) 
{
	
	unsigned int dataIdx=indexLookupArr[k];

	//If we use ILP
	#if ILP>0
	DTYPE runningDist[ILP];
	
	#pragma unroll
	for(int j=0; j<ILP; j++)
		runningDist[j]=0;


	for(int l=0; l<GPUNUMDIM; l+=ILP) {
		#pragma unroll
		for(int j=0; j<ILP && (l+j) < GPUNUMDIM; j++) {
			runningDist[j] += (database[dataIdx*GPUNUMDIM+l+j]-point[l+j])*(database[dataIdx*GPUNUMDIM+l+j]-point[l+j]);
		}
          	#if SHORTCIRCUIT==1
			#pragma unroll
			for(int j=1; j<ILP; j++) {
				runningDist[0] += runningDist[j];
				runningDist[j]=0;
			}
	        	if (sqrt(runningDist[0])>(*epsilon)) {
        	    	  return;
          		}
         	 #endif
	}

	#pragma unroll
	for(int j=1; j<ILP; j++) {
		runningDist[0] += runningDist[j];
	}

	#endif
	//end ILP

	//No ILP
	#if ILP==0
	DTYPE runningTotalDist=0;
    for (int l=0; l<GPUNUMDIM; l++){
      runningTotalDist+=(database[dataIdx*GPUNUMDIM+l]-point[l])*(database[dataIdx*GPUNUMDIM+l]-point[l]);
      #if SHORTCIRCUIT==1
      if (sqrt(runningTotalDist)>(*epsilon)) {
          return;
      }
      #endif
    }
	#endif
	//end no ILP

    	//distance calculation using either ILP or no ILP
    	#if ILP>0
        if (sqrt(runningDist[0])<=(*epsilon)){
        #endif
        #if ILP==0
        if (sqrt(runningTotalDist)<=(*epsilon)){	
        #endif	

		unsigned int idx=atomicAdd(cnt,int(1));
		pointIDKey[idx]=pointIdx;
		pointInDistVal[idx]=dataIdx;
		  
            if(differentCell) {
              unsigned int idx = atomicAdd(cnt,int(1));
			  pointIDKey[idx]=pointIdx;
              pointInDistVal[idx]=dataIdx;
           }
	}
}




__device__ void evaluateCell(unsigned int* nCells, unsigned int* indexes, DTYPE* database, 
	DTYPE * epsilon, struct grid * index, unsigned int * indexLookupArr, struct gridCellLookup * gridCellLookupArrStart, struct gridCellLookup * gridCellLookupArrStartEnd, 
	DTYPE* point, unsigned int* cnt,int* pointIDKey, int* pointInDistVal, int pointIdx, bool differentCell, unsigned int* nDCellIDs, CTYPE* workCounts)
{


#if COUNTMETRICS == 1
			atomicAdd(&workCounts[1],int(1));
#endif

	uint64_t calcLinearID=getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);

	//compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says 
	//a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)

	struct gridCellLookup tmp;
	tmp.gridLinearID=calcLinearID;

	if (thrust::binary_search(thrust::seq, gridCellLookupArrStart, gridCellLookupArrStartEnd, gridCellLookup(tmp))){

        //compute the neighbors for the adjacent non-empty cell
        struct gridCellLookup * resultBinSearch=thrust::lower_bound(thrust::seq, gridCellLookupArrStart, gridCellLookupArrStartEnd, gridCellLookup(tmp));
        unsigned int GridIndex=resultBinSearch->idx;
#if SORT==1
	int sortedDim;
#if GPUNUMDIM == NUMINDEXEDDIM
	sortedDim=0;
	int idx;
	bool mid=false;
	bool left=false;
	if(nDCellIDs[sortedDim] > indexes[sortedDim]){ 
		left = true; 
	}
	else if (nDCellIDs[sortedDim] < indexes[sortedDim]) {
		left =false; 
	}
	else mid = true;

        for(int k=index[GridIndex].indexmin; k<=index[GridIndex].indexmax; k++){
		idx = k;
		if(left) idx = index[GridIndex].indexmax - (k-index[GridIndex].indexmin);
		
		unsigned int dataIdx=indexLookupArr[idx];
		if(std::abs(database[dataIdx*GPUNUMDIM+sortedDim]-point[sortedDim]) > (*epsilon) && !mid) {
			k = index[GridIndex].indexmax+1;
		}
		else {
			evalPoint(indexLookupArr, idx, database, epsilon, point, cnt, pointIDKey, pointInDistVal, pointIdx, differentCell);
#if COUNTMETRICS == 1
			atomicAdd(&workCounts[0],1);
#endif
		}
        }
#else


	sortedDim = NUMINDEXEDDIM;
    // sortedDim = 43; //tmp
	int offset = index[GridIndex].indexmin;
	int length = (index[GridIndex].indexmax - index[GridIndex].indexmin);

	int searchIdx = (length)/2;
	bool lessThan;
	for(int step = (length+3)/4; step > 1; step = (step+1)/2) {
		lessThan = ((point[sortedDim] - database[(indexLookupArr[searchIdx+offset])*GPUNUMDIM+sortedDim]) >= (*epsilon));
		if(lessThan) { 
			searchIdx += step;
			if(searchIdx > length) searchIdx = length;
		}
		else {
			searchIdx -= step;
			if(searchIdx < 0) {
				searchIdx = 0;
				step = 0;
			}
		}
	}
	if(searchIdx > 0 && ((point[sortedDim] - database[indexLookupArr[(searchIdx+offset)-1]*GPUNUMDIM+sortedDim]) <= (*epsilon)))
		searchIdx--;
	if(searchIdx > 0 && ((point[sortedDim] - database[indexLookupArr[(searchIdx+offset)-1]*GPUNUMDIM+sortedDim]) <= (*epsilon)))
		searchIdx--;

	while(searchIdx <= length && ((database[indexLookupArr[searchIdx+offset]*GPUNUMDIM+sortedDim] - point[sortedDim]) < (*epsilon))) {
		evalPoint(indexLookupArr, searchIdx+offset, database, epsilon, point, cnt, pointIDKey, pointInDistVal, pointIdx, differentCell);
		searchIdx++;
#if COUNTMETRICS == 1
			atomicAdd(&workCounts[0],1);
#endif
	}
	
#endif

// Brute force method if SORTED != 1
#else
	for (int k=index[GridIndex].indexmin; k<=index[GridIndex].indexmax; k++){
		evalPoint(indexLookupArr, k, database, epsilon, point, cnt, pointIDKey, pointInDistVal, pointIdx, differentCell);
#if COUNTMETRICS == 1
			atomicAdd(&workCounts[0],1);
#endif
        }
#endif
	}

}



//Kernel brute forces to generate the neighbor table for each point in the database
//cnt with unsigned long long int
__global__ void kernelBruteForce(unsigned int *N, unsigned int *debug1, unsigned int *debug2, DTYPE* epsilon, unsigned long long int * cnt, DTYPE* database, int * pointIDKey, int * pointInDistVal) {

unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 

if (tid>=*N){
	return;
}


int dataOffset=tid*GPUNUMDIM;
DTYPE runningDist=0;
//compare my point to every other point
for (int i=0; i<(*N); i++)
{
	runningDist=0;
	for (int j=0; j<GPUNUMDIM; j++){
		runningDist+=(database[(i*GPUNUMDIM)+j]-database[dataOffset+j])*(database[(i*GPUNUMDIM)+j]-database[dataOffset+j]);
	}

	//if within epsilon:
	if ((sqrt(runningDist))<=(*epsilon)){
		//with long long unsigned int
		atomicAdd(cnt, (unsigned long long int)1);
		
	}
}


return;
}





//for descriptions of the parameters, see regular kernel that computes the result (not the batch estimator)
__global__ void kernelNDGridIndexBatchEstimator(unsigned int *debug1, unsigned int *debug2, unsigned int *N,  
	unsigned int * sampleOffset, const unsigned int DBSIZE, DTYPE* database, const DTYPE epsilon, unsigned int * whichIndexPoints, struct grid * allIndex, unsigned int * allIndexLookupArr, 
	struct gridCellLookup * allGridCellLookupArr, DTYPE* allMinArr, unsigned int * allNCells, gridCellLookup ** startGridPtrs, gridCellLookup ** stopGridPtrs, grid ** startIndexPtrs,
	unsigned int * cnt, unsigned int * allNNonEmptyCells, unsigned int * orderedQueryPntIDs)
{

unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 


if (tid>=*N){
	return;
}


//If reordering the queries by the amount of work
#if QUERYREORDER==1
//the point id in the dataset
unsigned int pointIdx=orderedQueryPntIDs[tid*(*sampleOffset)]; 
//The offset into the database, taking into consideration the length of each dimension
unsigned int pointID=(GPUNUMDIM)*pointIdx;
#endif

//If standard execution without reordering the queries by the amount of work
#if QUERYREORDER==0
unsigned int pointIdx = tid*(*sampleOffset);
unsigned int pointID=pointIdx*(GPUNUMDIM);
#endif

unsigned int whichIndex = whichIndexPoints[pointIdx];
unsigned int whichDatabase = 0;
if( whichIndex >= (NUMRANDINDEXES + 1)) {
	whichDatabase = whichIndex - NUMRANDINDEXES;
}

//make a local copy of the point
DTYPE point[GPUNUMDIM];
for (int i=0; i<GPUNUMDIM; i++){
	point[i]=database[(whichDatabase * DBSIZE * GPUNUMDIM) + pointID+i];
}

//calculate the coords of the Cell for the point
//and the min/max ranges in each dimension
unsigned int nDCellIDs[NUMINDEXEDDIM];
unsigned int nDMinCellIDs[NUMINDEXEDDIM];
unsigned int nDMaxCellIDs[NUMINDEXEDDIM];
for (int i=0; i<NUMINDEXEDDIM; i++){
	nDCellIDs[i]=(point[i]-allMinArr[i + (whichIndex * NUMINDEXEDDIM)])/(epsilon);
	nDMinCellIDs[i]=max(0,nDCellIDs[i]-1); //boundary conditions (don't go beyond cell 0)
	nDMaxCellIDs[i]=min(allNCells[i + (whichIndex * NUMINDEXEDDIM)]-1,nDCellIDs[i]+1); //boundary conditions (don't go beyond the maximum number of cells)

}

	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	for (loopRng[0]=nDMinCellIDs[0]; loopRng[0]<=nDMaxCellIDs[0]; loopRng[0]++)
	for (loopRng[1]=nDMinCellIDs[1]; loopRng[1]<=nDMaxCellIDs[1]; loopRng[1]++)
	#include "kernelloops.h"						
	{ //beginning of loop body
	

	#if COUNTMETRICS == 1
			atomicAdd(debug1,int(1));
	#endif	

	for (int x=0; x<NUMINDEXEDDIM; x++){
	indexes[x]=loopRng[x];	
	// if (tid==0)
	// 	printf("\ndim: %d, indexes: %d",x, indexes[x]);
	}
	
	unsigned int nCells[NUMINDEXEDDIM];
	for (int i=0; i<NUMINDEXEDDIM; i++){
		nCells[i] = allNCells[i+(whichIndex*NUMINDEXEDDIM)];
	}

	uint64_t calcLinearID=getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);

	//compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says 
	//a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)

	struct gridCellLookup tmp;
	tmp.gridLinearID=calcLinearID;


	gridCellLookup * startGridPtr = startGridPtrs[whichIndex];
	gridCellLookup * stopGridPtr = stopGridPtrs[whichIndex];
	grid * gridIndexPtr = startIndexPtrs[whichIndex];
	if (thrust::binary_search(thrust::seq, startGridPtr, stopGridPtr, gridCellLookup(tmp))){	
		struct gridCellLookup * resultBinSearch=thrust::lower_bound(thrust::seq, startGridPtr, stopGridPtr, gridCellLookup(tmp));
		unsigned int GridIndex=resultBinSearch->idx;
		
		for (int k=(gridIndexPtr+GridIndex)->indexmin; k<=(gridIndexPtr+GridIndex)->indexmax; k++){
				DTYPE runningTotalDist=0;
				unsigned int dataIdx=allIndexLookupArr[k+(whichIndex*(DBSIZE))];

				

				for (int l=0; l<GPUNUMDIM; l++){
				runningTotalDist+=(database[(whichDatabase * DBSIZE * GPUNUMDIM) + dataIdx*GPUNUMDIM+l]-point[l])*(database[(whichDatabase * DBSIZE * GPUNUMDIM) + dataIdx*GPUNUMDIM+l]-point[l]);
				}


				if (sqrt(runningTotalDist)<=(epsilon)){
					//Count number within epsilon
					unsigned int idx=atomicAdd(&cnt[whichIndex],int(1));
				}
			}



	}

	} //end loop body
}

__global__ void kernelInitEnumerateDB(unsigned int * databaseVal, const unsigned int N)
{


	unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 


	if (tid>=N){
		return;
	}

	databaseVal[tid]=tid;	

		
}

__global__ void kernelIndexComputeNonemptyCells(DTYPE* database, const unsigned int N, const unsigned int whichDatabase, DTYPE* epsilon, DTYPE* minArr, unsigned int * nCells, uint64_t * pointCellArr)
{


	unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 


	if (tid>=N){
		return;
	}

	// printf("\n%u",tid); 

	unsigned int pointID=tid*(GPUNUMDIM);

	unsigned int tmpNDCellIdx[NUMINDEXEDDIM];
	for (int j=0; j<NUMINDEXEDDIM; j++){
		tmpNDCellIdx[j]=((database[(N*GPUNUMDIM*whichDatabase) + pointID + j]-minArr[j])/(*epsilon));
	}
	uint64_t linearID=getLinearID_nDimensionsGPU(tmpNDCellIdx, nCells, NUMINDEXEDDIM);

	pointCellArr[tid]=linearID;	

		
}


__global__ void kernelIndexComputeAdjacentCells(uint64_t * celllDistCalcArr, uint64_t * uniqueCellArr, uint64_t * cellNumPointsArr, unsigned int * nCells, unsigned int * nNonEmptyCells, int *incrementors, unsigned int * nAdjCells)
{
	unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 

	unsigned int numDistCalcs = 0;
	int tempIdx;

	// make sure thread is within unique cell count
	if (tid>=*nNonEmptyCells){
		return;
	}

	// get value of cell linearID for this thread
	uint64_t linearID = uniqueCellArr[tid];

	unsigned int pointIdx[NUMINDEXEDDIM];

	// get index of grid point
	getNDimIndexesFromLinearIdxGPU(pointIdx, nCells, NUMINDEXEDDIM, linearID);

	// check that linear ID does not already have adjacent cells caluclated for it
	unsigned int tempAdjCellIdx[NUMINDEXEDDIM];

	// loop through each incrementor vector
	for (int i=0; i<(*nAdjCells); i++)
	{
		// loop thhrough each entry
		for (int j=0; j<NUMINDEXEDDIM; j++)
		{
			// get adjacent cell index
			tempAdjCellIdx[j] = pointIdx[j] + incrementors[(i * NUMINDEXEDDIM) + j];
		}
		// convert index back to cell id
		uint64_t adjLinearID=getLinearID_nDimensionsGPU(tempAdjCellIdx, nCells, NUMINDEXEDDIM);

		// test if cell is nonempty
		tempIdx = binarySearch(uniqueCellArr, *nNonEmptyCells, adjLinearID);
		if(tempIdx != -1)
		{
			// add number of points in the cell to num distance calcs
			numDistCalcs += cellNumPointsArr[tempIdx];
		}
	}

	// set number of dist calcs
	celllDistCalcArr[tid] = numDistCalcs;
}


__global__ void kernelMapPointToNumDistCalcs(uint64_t * pointDistCalcArr, DTYPE* database, const unsigned int N, const unsigned int whichDatabase, DTYPE* epsilon, DTYPE* minArr, unsigned int * nCells, uint64_t *cellDistCalcArr,  uint64_t * uniqueCellArr, unsigned int * nNonEmptyCells)
{
	unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 
	int tempIdx;

	if (tid>=N){
		return;
	}

	unsigned int pointID=tid*(GPUNUMDIM);

	unsigned int tmpNDCellIdx[NUMINDEXEDDIM];
	for (int j=0; j<NUMINDEXEDDIM; j++){
		tmpNDCellIdx[j]=((database[(N*GPUNUMDIM*whichDatabase) + pointID + j]-minArr[j])/(*epsilon));
	}
	uint64_t linearID=getLinearID_nDimensionsGPU(tmpNDCellIdx, nCells, NUMINDEXEDDIM);

	// get the index of linear ID in unique cell array
	tempIdx = binarySearch(uniqueCellArr, *nNonEmptyCells, linearID);
	if(tempIdx != -1)
	{
		// set value to number of distance calculations for that cell
		pointDistCalcArr[tid] = cellDistCalcArr[tempIdx];
	}
	else
	{
		// unexpected that point cell is not found
		printf("\nWARNING: The cell for point %d was not found\n", pointID);
	}
}

__global__ void kernelPairwiseDatabaseRotation( DTYPE * database, const unsigned int N, const unsigned int whichDatabase, DTYPE * theta, unsigned int * dimPair ) {
	unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE);

	if (tid>=N){
		return;
	}

	unsigned int pointID=tid*(GPUNUMDIM);

	for( unsigned int i=0; i<NUMPAIRROTATIONS; i++ )
	{
		DTYPE dim_0 = database[(N*GPUNUMDIM*whichDatabase) + pointID + dimPair[i * 2]];
		DTYPE dim_1 = database[(N*GPUNUMDIM*whichDatabase) + pointID + dimPair[i * 2 + 1]];

		DTYPE new_dim_0 = (dim_0 * cos(theta[i])) - (dim_1 * sin(theta[i]));
		DTYPE new_dim_1 = (dim_0 * sin(theta[i])) + (dim_1 * cos(theta[i]));

		database[(N*GPUNUMDIM*whichDatabase) + pointID + dimPair[i * 2]] = new_dim_0;
		database[(N*GPUNUMDIM*whichDatabase) + pointID + dimPair[i * 2 + 1]] = new_dim_1;
	}
}











__global__ void kernelNDGridIndexGlobalManagedMemory(unsigned int *debug1, unsigned int *debug2, const unsigned int N,  
	const unsigned int indexGroupOffset, DTYPE* database, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, 
	struct gridCellLookup * gridCellLookupArrStart, struct gridCellLookup * gridCellLookupArrEnd, DTYPE* minArr, unsigned int * nCells, 
	unsigned long long int * cnt, struct keyValPair * keyValPairs, unsigned int * orderedQueryPntIDs, CTYPE* workCounts)
{

unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE);
/*
if (threadIdx.x == 0) {
    printf("%d: Current sum: %llu\n", blockIdx.x, *cnt);
}
*/

if (tid>=N){
	return;
}

//If reordering the queries by the amount of work
#if QUERYREORDER==1
//test refactoring no reordering
// unsigned int pointIdx=tid*(*offset)+(*batchNum);
// unsigned int pointOffset=(GPUNUMDIM)*(tid*(*offset)+(*batchNum));

//the point id in the dataset
unsigned int pointIdx=orderedQueryPntIDs[tid + indexGroupOffset]; 
//The offset into the database, taking into consideration the length of each dimension
unsigned int pointOffset=(GPUNUMDIM)*pointIdx; 


#endif

//If standard execution without reordering the queries by the amount of work
#if QUERYREORDER==0
//the point id in the dataset
unsigned int pointIdx=tid + indexGroupOffset;
//The offset into the database, taking into consideration the length of each dimension
unsigned int pointOffset=(GPUNUMDIM)*pointIdx;
#endif



//make a local copy of the point
//Store query point in registers
DTYPE point[GPUNUMDIM];
for (int i=0; i<GPUNUMDIM; i++){
	point[i]=database[pointOffset+i];	
}


//calculate the coords of the Cell for the point
//and the min/max ranges in each dimension
unsigned int nDCellIDs[NUMINDEXEDDIM];
unsigned int nDMinCellIDs[NUMINDEXEDDIM];
unsigned int nDMaxCellIDs[NUMINDEXEDDIM];
for (int i=0; i<NUMINDEXEDDIM; i++){
	nDCellIDs[i]=(point[i]-minArr[i])/(*epsilon);
	nDMinCellIDs[i]=max(0,nDCellIDs[i]-1); //boundary conditions (don't go beyond cell 0)
	nDMaxCellIDs[i]=min(nCells[i]-1,nDCellIDs[i]+1); //boundary conditions (don't go beyond the maximum number of cells)

}



	///////////////////////////////////////
	//End taking intersection
	//////////////////////////////////////	
	
        unsigned int indexes[NUMINDEXEDDIM];
        unsigned int loopRng[NUMINDEXEDDIM];

#if STAMP

	for(int i=0; i<NUMINDEXEDDIM; i++) {
		indexes[i] = nDCellIDs[i];
	}
	evaluateCell(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr, point, cnt, pointIDKey, pointInDistVal, pointIdx, false, nDCellIDs, workCounts);

    #include "stamploops.h"

#elif LINEARSTAMP
	for(int i=0; i<NUMINDEXEDDIM; i++) {
		indexes[i] = nDCellIDs[i];
	}
	evaluateCell(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr, point, cnt, pointIDKey, pointInDistVal, pointIdx, false, nDCellIDs, workCounts);

	for (loopRng[0]=rangeFilteredCellIdsMin[0]; loopRng[0]<=rangeFilteredCellIdsMax[0]; loopRng[0]++)
	for (loopRng[1]=rangeFilteredCellIdsMin[1]; loopRng[1]<=rangeFilteredCellIdsMax[1]; loopRng[1]++)
	#include "kernelloops.h"					
	{ //beginning of loop body
	
	for (int x=0; x<NUMINDEXEDDIM; x++){
	indexes[x]=loopRng[x];	
	}
	if(getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM) > getLinearID_nDimensionsGPU(nDCellIDs, nCells, NUMINDEXEDDIM)) {
		evaluateCell(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr, point, cnt, pointIDKey, pointInDistVal, pointIdx, true, nDCellIDs, workCounts);
	}	
	} //end loop body
#else

	for (loopRng[0]=nDMinCellIDs[0]; loopRng[0]<=nDMaxCellIDs[0]; loopRng[0]++)
	for (loopRng[1]=nDMinCellIDs[1]; loopRng[1]<=nDMaxCellIDs[1]; loopRng[1]++)
	#include "kernelloops.h"					
	{ //beginning of loop body
	
	for (int x=0; x<NUMINDEXEDDIM; x++){
	indexes[x]=loopRng[x];	
	}
		evaluateCellManagedMemory(nCells, indexes, gridCellLookupArrStart, gridCellLookupArrEnd, database, epsilon, index, indexLookupArr, point, cnt, keyValPairs, pointIdx, false, nDCellIDs, workCounts);
	
	} //end loop body
#endif

}

__forceinline__ __device__ void evalPointManagedMemory(unsigned int* indexLookupArr, int k, DTYPE* database, DTYPE* epsilon, DTYPE* point, unsigned long long int* cnt, struct keyValPair * keyValPairs, int pointIdx, bool differentCell) {
	
	unsigned int dataIdx=indexLookupArr[k];

	//If we use ILP
	#if ILP>0
	DTYPE runningDist[ILP];
	
	#pragma unroll
	for(int j=0; j<ILP; j++)
		runningDist[j]=0;


	for(int l=0; l<GPUNUMDIM; l+=ILP) {
		#pragma unroll
		for(int j=0; j<ILP && (l+j) < GPUNUMDIM; j++) {
			runningDist[j] += (database[dataIdx*GPUNUMDIM+l+j]-point[l+j])*(database[dataIdx*GPUNUMDIM+l+j]-point[l+j]);
		}
          	#if SHORTCIRCUIT==1
			#pragma unroll
			for(int j=1; j<ILP; j++) {
				runningDist[0] += runningDist[j];
				runningDist[j]=0;
			}
	        	if (sqrt(runningDist[0])>(*epsilon)) {
        	    	  return;
          		}
         	 #endif
	}

	#pragma unroll
	for(int j=1; j<ILP; j++) {
		runningDist[0] += runningDist[j];
	}

	#endif
	//end ILP

	//No ILP
	#if ILP==0
	DTYPE runningTotalDist=0;
    for (int l=0; l<GPUNUMDIM; l++){
      runningTotalDist+=(database[dataIdx*GPUNUMDIM+l]-point[l])*(database[dataIdx*GPUNUMDIM+l]-point[l]);
      #if SHORTCIRCUIT==1
      if (sqrt(runningTotalDist)>(*epsilon)) {
          return;
      }
      #endif
    }
	#endif
	//end no ILP

    	//distance calculation using either ILP or no ILP
    	#if ILP>0
        if (sqrt(runningDist[0])<=(*epsilon)){
        #endif
        #if ILP==0
        if (sqrt(runningTotalDist)<=(*epsilon)){	
        #endif	
          unsigned long long int idx=atomicAdd(cnt,1ULL);
		  keyValPairs[idx].key=pointIdx;
		  keyValPairs[idx].val=dataIdx;

            if(differentCell) {
              unsigned long long int idx = atomicAdd(cnt,1ULL);
			  keyValPairs[idx].key=pointIdx;
			  keyValPairs[idx].val=dataIdx;
           }
	}
}



__device__	void evaluateCellManagedMemory(unsigned int* nCells, unsigned int* indexes, struct gridCellLookup * gridCellLookupArrStart, 
	struct gridCellLookup * gridCellLookupArrStartEnd, DTYPE* database, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, DTYPE* point, 
	unsigned long long int* cnt, struct keyValPair * keyValPairs, int pointIdx, bool differentCell, unsigned int* nDCellIDs, CTYPE* workCounts) {


#if COUNTMETRICS == 1
			atomicAdd(&workCounts[1],int(1));
#endif

        uint64_t calcLinearID=getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);
        //compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says 
        //a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)

        struct gridCellLookup tmp;
        tmp.gridLinearID=calcLinearID;
        //find if the cell is non-empty
        if (thrust::binary_search(thrust::seq, gridCellLookupArrStart, gridCellLookupArrStartEnd, gridCellLookup(tmp))){

        //compute the neighbors for the adjacent non-empty cell
        struct gridCellLookup * resultBinSearch=thrust::lower_bound(thrust::seq, gridCellLookupArrStart, gridCellLookupArrStartEnd, gridCellLookup(tmp));
        unsigned int GridIndex=resultBinSearch->idx;
#if SORT==1
	int sortedDim;
#if GPUNUMDIM == NUMINDEXEDDIM
	sortedDim=0;
	int idx;
	bool mid=false;
	bool left=false;
	if(nDCellIDs[sortedDim] > indexes[sortedDim]){ 
		left = true; 
	}
	else if (nDCellIDs[sortedDim] < indexes[sortedDim]) {
		left =false; 
	}
	else mid = true;

        for(int k=index[GridIndex].indexmin; k<=index[GridIndex].indexmax; k++){
		idx = k;
		if(left) idx = index[GridIndex].indexmax - (k-index[GridIndex].indexmin);
		
		unsigned int dataIdx=indexLookupArr[idx];
		if(std::abs(database[dataIdx*GPUNUMDIM+sortedDim]-point[sortedDim]) > (*epsilon) && !mid) {
			k = index[GridIndex].indexmax+1;
		}
		else {
			evalPoint(indexLookupArr, idx, database, epsilon, point, cnt, pointIDKey, pointInDistVal, pointIdx, differentCell);
#if COUNTMETRICS == 1
			atomicAdd(&workCounts[0],1);
#endif
		}
        }
#else


	sortedDim = NUMINDEXEDDIM;
    // sortedDim = 43; //tmp
	int offset = index[GridIndex].indexmin;
	int length = (index[GridIndex].indexmax - index[GridIndex].indexmin);

	int searchIdx = (length)/2;
	bool lessThan;
	for(int step = (length+3)/4; step > 1; step = (step+1)/2) {
		lessThan = ((point[sortedDim] - database[(indexLookupArr[searchIdx+offset])*GPUNUMDIM+sortedDim]) >= (*epsilon));
		if(lessThan) { 
			searchIdx += step;
			if(searchIdx > length) searchIdx = length;
		}
		else {
			searchIdx -= step;
			if(searchIdx < 0) {
				searchIdx = 0;
				step = 0;
			}
		}
	}
	if(searchIdx > 0 && ((point[sortedDim] - database[indexLookupArr[(searchIdx+offset)-1]*GPUNUMDIM+sortedDim]) <= (*epsilon)))
		searchIdx--;
	if(searchIdx > 0 && ((point[sortedDim] - database[indexLookupArr[(searchIdx+offset)-1]*GPUNUMDIM+sortedDim]) <= (*epsilon)))
		searchIdx--;

	while(searchIdx <= length && ((database[indexLookupArr[searchIdx+offset]*GPUNUMDIM+sortedDim] - point[sortedDim]) < (*epsilon))) {
		evalPoint(indexLookupArr, searchIdx+offset, database, epsilon, point, cnt, pointIDKey, pointInDistVal, pointIdx, differentCell);
		searchIdx++;
#if COUNTMETRICS == 1
			atomicAdd(&workCounts[0],1);
#endif
	}
	
#endif

// Brute force method if SORTED != 1
#else
	for (int k=index[GridIndex].indexmin; k<=index[GridIndex].indexmax; k++){
		evalPointManagedMemory(indexLookupArr, k, database, epsilon, point, cnt, keyValPairs, pointIdx, differentCell);
#if COUNTMETRICS == 1
			atomicAdd(&workCounts[0],1);
#endif
        }
#endif
	}

}