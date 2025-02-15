#include <pthread.h>
#include <cstdlib>
#include <stdio.h>
#include <random>
#include "omp.h"
#include <algorithm>
#include <string.h>
#include <fstream>
#include <iostream>
#include <string>
#include "GPU.h"
#include "kernel.h"

#ifndef PYTHON
#include "tree_index.h"
#endif

#include <math.h>
#include <queue>
#include <iomanip>
#include <set>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>

// for printing defines as strings
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// sort descending
bool compareByDimVariance(const dim_reorder_sort &a, const dim_reorder_sort &b)
{
	return a.variance > b.variance;
}

using namespace std;

// function prototypes
uint64_t getLinearID_nDimensions(unsigned int *indexes, unsigned int *dimLen, unsigned int nDimensions);
void getNDimIndexesFromLinearIdx(unsigned int *indexes, unsigned int *dimLen, unsigned int nDimensions, uint64_t linearId);
void populateNDGridIndexAndLookupArray(std::vector<std::vector<DTYPE>> *NDdataPoints, DTYPE epsilon, struct gridCellLookup **gridCellLookupArr, struct grid **index, unsigned int *indexLookupArr, DTYPE *minArr, unsigned int *nCells, uint64_t totalCells, unsigned int *nNonEmptyCells, unsigned int **gridCellNDMask, unsigned int *gridCellNDMaskOffsets, unsigned int *nNDMaskElems, std::unordered_map<uint64_t, std::vector<uint64_t>> &uniqueGridAdjacentCells, std::vector<std::vector<int>> &incrementorVects);
void generateNDGridDimensions(DTYPE * database, const unsigned int DBSIZE,  const unsigned int whichDatabase, DTYPE epsilon, DTYPE *minArr, DTYPE *maxArr, unsigned int *nCells, uint64_t *totalCells, DTYPE *indexOffset);
void importNDDataset(std::vector<std::vector<DTYPE>> *dataPoints, char *fname);
void ReorderByDimension(std::vector<std::vector<DTYPE>> *NDdataPoints);
void computeNumDistanceCalcs(std::vector<workArrayPnt> *totalPointsWork, unsigned int *nNonEmptyCells, gridCellLookup *gridCellLookupArr, grid *index, std::unordered_map<uint64_t, std::vector<uint64_t>> *uniqueGridAdjacentCells, std::vector<std::vector<DTYPE>> *NDdataPoints, DTYPE *minArr, unsigned int *nCells, DTYPE &epsilon);
void findAdjacentCellIDs(std::unordered_map<uint64_t, std::vector<uint64_t>> &uniqueGridAdjacentCells, std::vector<std::vector<int>> &incrementorVects, std::vector<uint64_t> &uniqueGridCellLinearIdsVect, unsigned int *nCells);
void printNeighborTable(unsigned int databaseSize, struct neighborTableLookup *neighborTable);

// sort ascending
bool compareByPointValue(const keyValNumPointsStruct &a, const keyValNumPointsStruct &b)
{
	return a.counts < b.counts;
}

// generate all combinations of sets (use for getting adjacent cells)
void generateCombinations(const std::vector<std::vector<int>> &sets, std::vector<int> &current, int depth, std::vector<std::vector<int>> &combinations)
{
	if (depth == sets.size())
	{
		combinations.push_back(current);
		return;
	}

	for (auto element : sets[depth])
	{
		current[depth] = element;
		generateCombinations(sets, current, depth + 1, combinations);
	}
}

// sort total cell work array descending
bool compareWorkArrayByNumDistanceCalcs(const workArrayPnt &a, const workArrayPnt &b)
{
	return a.numDistCalcs > b.numDistCalcs;
}

// sort index point array by work descending
bool compareIndexArrayByNumDistanceCalcs(const indexArrayPnt& a, const indexArrayPnt& b) {
	return a.numDistCalcs > b.numDistCalcs;
	}

#ifndef PYTHON // standard C version
int main(int argc, char *argv[])
{
	unsigned int NUMTOTALINDEXES = 1 + (NUMRANDINDEXES + NUMRANDROTATIONS);

	// check that the number of data dimensions is greater than or equal to the number of indexed dimensions
	assert(GPUNUMDIM >= NUMINDEXEDDIM);

	omp_set_max_active_levels(4);
	/////////////////////////
	// Get information from command line
	// 1) the dataset, 2) epsilon, 3) number of dimensions
	/////////////////////////

	// Read in parameters from file:
	// dataset filename and cluster instance file
	if (argc != 4)
	{
		cout << "\n\nIncorrect number of input parameters.  \nShould be dataset file, epsilon, number of dimensions\n";
		return 0;
	}

	// copy parameters from commandline:
	// char inputFname[]="data/test_data_removed_nan.txt";
	char inputFname[500];
	char inputEpsilon[500];
	char inputnumdim[500];

	strcpy(inputFname, argv[1]);
	strcpy(inputEpsilon, argv[2]);
	strcpy(inputnumdim, argv[3]);

	DTYPE epsilon = atof(inputEpsilon);
	unsigned int NDIM = atoi(inputnumdim);

	if (GPUNUMDIM != NDIM)
	{
		printf("\nERROR: The number of dimensions defined for the GPU is not the same as the number of dimensions\n \
		 passed into the computer program on the command line. GPUNUMDIM=%d, NDIM=%d Exiting!!!",
			   GPUNUMDIM, NDIM);
		return 0;
	}

	printf("\nDataset file: %s", inputFname);
	printf("\nEpsilon: %f", epsilon);
	printf("\nNumber of dimensions (NDIM): %d\n", NDIM);

	//////////////////////////////
	// import the dataset:
	/////////////////////////////

	std::vector<std::vector<DTYPE>> NDdataPoints;
	importNDDataset(&NDdataPoints, inputFname);

	// GPU with Grid index

	double entire_time_start = omp_get_wtime();

	char fname[] = "gpu_stats.txt";
	ofstream gpu_stats;
	gpu_stats.open(fname, ios::app);

	printf("\n*****************\nWarming up GPU:\n*****************\n");
	warmUpGPU();
	printf("\n*****************\n");

	double timeReorderByDimVariance = 0;
	#if REORDER == 1
		double reorder_start = omp_get_wtime();
		ReorderByDimension(&NDdataPoints);
		double reorder_end = omp_get_wtime();
		timeReorderByDimVariance = reorder_end - reorder_start;
		printf("\nTime to reorder: %f", timeReorderByDimVariance);
	#endif

	///////////////////////////////////
	//COPY THE DATABASE TO THE GPU
	///////////////////////////////////
	//CUDA error code:
	cudaError_t errCode;

	unsigned int DBSIZE;
	DBSIZE=NDdataPoints.size();

	DTYPE* database = (DTYPE*)malloc(sizeof(DTYPE)*(DBSIZE)*(GPUNUMDIM)*(NUMRANDROTATIONS+1));  
	DTYPE* dev_database;
	
	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_database, sizeof(DTYPE)*(DBSIZE)*(GPUNUMDIM)*(NUMRANDROTATIONS+1));		
	if(errCode != cudaSuccess) {
	cout << "\nError: database alloc -- error with code " << errCode << endl; cout.flush(); 
	}


	//copy the database from the ND vector to the array, add copy for each random rotation:
	for( int i=0; i<NUMRANDROTATIONS+1; i++ ){
		for (int j=0; j<(DBSIZE); j++){
			std::copy(NDdataPoints[j].begin(), NDdataPoints[j].end(), database+((DBSIZE)*GPUNUMDIM*i)+(j*(GPUNUMDIM)));
		}
	}


	//copy database to the device
	errCode=cudaMemcpy(dev_database, database, sizeof(DTYPE)*(DBSIZE)*(GPUNUMDIM)*(NUMRANDROTATIONS+1), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: database Got error with code " << errCode << endl; 
	}

	///////////////////////////////////
	//END COPY THE DATABASE TO THE GPU
	///////////////////////////////////

	double rotations_time_start = omp_get_wtime();

	// std::vector<std::vector<std::vector<DTYPE>>> allRotatedNDdataPoints;
	
	// generate random point rotations
	// NOTE: leave i=0 as the original database
	for( int i = 1; i < NUMRANDROTATIONS+1; i++ ) {
		// printf("\nRotation %d", i+1);
		// printf("\n~~~~~~~~~~~");

		rotateOnGPU(dev_database, DBSIZE, i);

		// cout << endl;
		// printf("\n");
	}

	double rotations_time_end = omp_get_wtime();

	printf("\nTime to get rotations: %f", (rotations_time_end-rotations_time_start));


	errCode=cudaMemcpy(database, dev_database, sizeof(DTYPE)*(DBSIZE)*(GPUNUMDIM)*(NUMRANDROTATIONS+1), cudaMemcpyDeviceToHost);	
	if(errCode != cudaSuccess) {
	cout << "\nError: database Got error with code " << errCode << endl; 
	}

	/*
	//printf("\n\n");
	double fill_vector_time_start = omp_get_wtime();
	for( int i=0; i<NUMRANDROTATIONS; i++ ){
		std::vector<std::vector<DTYPE>> tmpDBVector;
		for (int j=0; j<(DBSIZE); j++){
			std::vector<DTYPE> tmpPointVector;
			for( int k=0; k<GPUNUMDIM; k++ ){
				//printf("%f, ", database[(DBSIZE*GPUNUMDIM*i) + (j*GPUNUMDIM) + k]);
				tmpPointVector.emplace_back(database[(DBSIZE*GPUNUMDIM*i) + (j*GPUNUMDIM) + k]);
			}
			//printf("\n");
			tmpDBVector.emplace_back(tmpPointVector);
		}
		//printf("\n\n");
		allRotatedNDdataPoints.emplace_back(tmpDBVector);
	}
	double fill_vector_time_end = omp_get_wtime();
	printf("\nTime to fill vector: %f", (fill_vector_time_end-fill_vector_time_start));
	*/


	// inititalize arrays
	std::vector<std::vector<workArrayPnt>> allTotalPointsWork;
	DTYPE *allMinArr = new DTYPE[NUMINDEXEDDIM * NUMTOTALINDEXES];
	unsigned int *allNCells = new unsigned int[NUMINDEXEDDIM * NUMTOTALINDEXES];
	unsigned int *allNNonEmptyCells = new unsigned int[NUMTOTALINDEXES];

	std::vector<struct grid> allIndexVec;
	std::vector<struct gridCellLookup> allGridCellLookupArrVec;
	unsigned int *allIndexLookupArr = new unsigned int[NDdataPoints.size() * NUMTOTALINDEXES];

	// maps each non empty cell to its adjacent cells, including itself
	// get all incrementors to find adjacent cells (vectors to add to position to get each adjecent cell)
	std::vector<std::vector<int>> incrementorVects;
	std::vector<int> incrementors_set = {-1, 0, 1};
	std::vector<std::vector<int>> sets(NUMINDEXEDDIM, incrementors_set);
	std::vector<int> current(sets.size());
	generateCombinations(sets, current, 0, incrementorVects);

	// define random engine
    std::random_device rd;
	// initialize generator
    std::mt19937 gen(rd());
	// produce random float within [0, epsilon/2)
	std::uniform_real_distribution<> dis(DTYPE(0), epsilon/2);

	// keep track of random offsets
	std::vector<DTYPE> allOffsets;

	unsigned int idxCounter = 0;
	DTYPE indexOffsetPtr[NUMINDEXEDDIM];

	// TODO: this is currently making 2 of the default index
	// get num distance calcs for each point for each index
	for (int indexIdx = 0; indexIdx < NUMTOTALINDEXES; indexIdx++)
	{
		// display the offset
		printf("\n\n+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+");
		
		// use default rotation for indexes
		unsigned int rotationIdx = 0;
		if( indexIdx >= (NUMRANDINDEXES + 1)) {
			rotationIdx = indexIdx - NUMRANDINDEXES;
			printf("\nRotation %d", rotationIdx);
		}
		else if( indexIdx > 0 ) {
			printf("\nOffset %d", indexIdx);
		}
		else {
			printf("\nOriginal Index");
		}

		#if RANDOMOFFSETSAMEALLDIM == 1 || FIXEDOFFSETALLDIM == 1
		DTYPE indexOffset;
		DTYPE* indexOffsetPtr;
		// use no offset index for first iteration and offset for second
		if (indexIdx == 0)
		{
			indexOffset = 0;
		}
		else
		{
			#if RANDOMOFFSETSAMEALLDIM == 1
			// generate new number that is not the min and not previously generated
			do {
				indexOffset = dis(gen);
			} while( 
				indexOffset == 0 || std::binary_search(allOffsets.begin(), allOffsets.end(), indexOffset )
			);

			allOffsets.emplace_back(indexOffset);

			#elif FIXEDOFFSETALLDIM == 1
			// generate even spaces indexes
			indexOffset = indexIdx * ((epsilon/2) / (NUMRANDINDEXES));
			#endif

			indexOffsetPtr = &indexOffset;
		}
		#elif RANDOMOFFSETFOREACHDIM == 1
		// use no offset index for first iteration and offset for second
		// use no offset if using rotation
		if (indexIdx == 0 || indexIdx >= (NUMRANDINDEXES + 1))
		{
			for( int i=0; i<NUMINDEXEDDIM; i++)
			{
				indexOffsetPtr[i] = 0;
			}
		}
		else
		{
			// generate new number indexes for each dimension
			for( int i=0; i<NUMINDEXEDDIM; i++)
			{
				indexOffsetPtr[i] = dis(gen);
			}
		}
		#endif

		DTYPE *minArr = new DTYPE[NUMINDEXEDDIM];
		DTYPE *maxArr = new DTYPE[NUMINDEXEDDIM];
		unsigned int *nCells = new unsigned int[NUMINDEXEDDIM];
		uint64_t totalCells = 0;
		unsigned int nNonEmptyCells = 0;

		double tstart_index = omp_get_wtime();
		generateNDGridDimensions(database, DBSIZE, rotationIdx, epsilon, minArr, maxArr, nCells, &totalCells, indexOffsetPtr);
		printf("\nGrid: total cells (including empty) %lu", totalCells);

		// allocate memory for index now that we know the number of cells
		// the grid struct itself
		// the grid lookup array that accompanys the grid -- so we only send the non-empty cells
		struct grid *index;						  // allocate in the populateDNGridIndexAndLookupArray -- only index the non-empty cells
		struct gridCellLookup *gridCellLookupArr; // allocate in the populateDNGridIndexAndLookupArray -- list of non-empty cells

		// ids of the elements in the database that are found in each grid cell
		unsigned int *indexLookupArr = new unsigned int[DBSIZE];

		// number of distance calculations per point
		std::vector<workArrayPnt> totalPointsWork;

		// std::unordered_map<uint64_t, std::vector<uint64_t>> uniqueGridAdjacentCells;
		// populateNDGridIndexAndLookupArray(&NDdataPoints, epsilon, &gridCellLookupArr, &index, indexLookupArr, minArr, nCells, totalCells, &nNonEmptyCells, &gridCellNDMask, gridCellNDMaskOffsets, nNDMaskElems, uniqueGridAdjacentCells, incrementorVects);
		
		populateNDGridIndexAndLookupArrayGPU(dev_database, DBSIZE, rotationIdx, &epsilon, minArr, totalCells, nCells, &gridCellLookupArr, &index, indexLookupArr, &nNonEmptyCells, &incrementorVects, &totalPointsWork);
		double tend_index = omp_get_wtime();
		printf("\nTime to index (not counted in the time): %f", tend_index - tstart_index);

		// add work for each point vector to total
		allTotalPointsWork.push_back(totalPointsWork);

		// copy over all data to array positions for current random index
		for (int i = 0; i < NUMINDEXEDDIM; i++)
		{
			allMinArr[i + (idxCounter * NUMINDEXEDDIM)] = minArr[i];
			allNCells[i + (idxCounter * NUMINDEXEDDIM)] = nCells[i];
		}

		allNNonEmptyCells[idxCounter] = nNonEmptyCells;

		for (int i = 0; i < nNonEmptyCells; i++)
		{
			allIndexVec.push_back(index[i]);
			allGridCellLookupArrVec.push_back(gridCellLookupArr[i]);
		}

		for (int i = 0; i < DBSIZE; i++)
		{
			allIndexLookupArr[i + (idxCounter * DBSIZE)] = indexLookupArr[i];
		}

		printf("\n+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+\n");

		delete[] minArr;
		delete[] maxArr;
		delete[] nCells;
		delete[] indexLookupArr;
		minArr = NULL;
		maxArr = NULL;
		nCells = NULL;
		indexLookupArr = NULL;

		idxCounter++;
	}


	// create arrays with index for each random index
	struct grid *allIndex = new grid[allIndexVec.size()];
	struct gridCellLookup *allGridCellLookupArr = new gridCellLookup[allGridCellLookupArrVec.size()];
	std::copy(allIndexVec.begin(), allIndexVec.end(), allIndex);
	std::copy(allGridCellLookupArrVec.begin(), allGridCellLookupArrVec.end(), allGridCellLookupArr);

	// get which grid offeset index to use for each point index
	unsigned int * whichIndexPoints = new unsigned int[NDdataPoints.size()];

	// create array to sort points by index
	std::vector<indexArrayPnt> indexPoints;
	unsigned int * orderedIndexPntIDs=new unsigned int[NDdataPoints.size()];

	#if QUERYREORDER==0
	unsigned int * orderedQueryPntIDs=NULL;
	#endif
	#if QUERYREORDER==1
	// create array for arranging points by work (used for batch estimator)
	std::vector<indexArrayPnt> totalPointsWork;
	unsigned int * orderedQueryPntIDs=new unsigned int[NDdataPoints.size()];
	#endif

	// loop through each point idx
	for (int i = 0; i < NDdataPoints.size(); i++)
	{
		unsigned int whichIdx;
		unsigned int leastDistCalcs = UINT_MAX;
		// loop through each "random" index
		for (int j = 0; j < NUMTOTALINDEXES; j++)
		{
			if (allTotalPointsWork[j][i].numDistCalcs < leastDistCalcs)
			{
				leastDistCalcs = allTotalPointsWork[j][i].numDistCalcs;
				whichIdx = j;
			}
		}

		// add index data for this point
		indexArrayPnt tmp;
		tmp.pntIdx = i;
		tmp.whichIndex = whichIdx;
		tmp.numDistCalcs = leastDistCalcs;
		indexPoints.push_back(tmp);
		
		#if QUERYREORDER==1
			totalPointsWork.push_back(tmp);
		#endif

		whichIndexPoints[i] = whichIdx;
	}

	// sort by which index
	std::sort(indexPoints.begin(), indexPoints.end(), [](const indexArrayPnt& a, const indexArrayPnt& b) {
		return a.whichIndex < b.whichIndex;
	});


	// split into groups of each index
	std::vector<indexArrayPntGroups> indexGroups;
	// start count at first index
	unsigned int cnt = indexPoints.begin()->whichIndex;
	// initialize first element in array
	indexArrayPntGroups tmp;
	tmp.index = cnt;
	tmp.indexmin = 0;
	indexGroups.push_back(tmp);
	for(int i=1; i<indexPoints.size(); i++) {
		if( indexPoints[i].whichIndex != indexPoints[i-1].whichIndex) {
			indexGroups[cnt].indexmax = i;
			cnt++;
			tmp.index = cnt;
			tmp.indexmin = i;
			indexGroups.push_back(tmp);
		}
	}
	indexGroups[cnt].indexmax = indexPoints.size();

	#if QUERYREORDER==1
		// sort each group by work
		// create array for arranging points by work
		for( int i=0; i<indexGroups.size(); i++) {
			std::sort(indexPoints.begin()+indexGroups[i].indexmin, indexPoints.begin()+indexGroups[i].indexmax, compareIndexArrayByNumDistanceCalcs);
		}

		// sort total work
		std::sort(totalPointsWork.begin(), totalPointsWork.end(), compareIndexArrayByNumDistanceCalcs);

		// populate query reorder
		for (unsigned int i=0; i<totalPointsWork.size(); i++)
		{
			orderedQueryPntIDs[i]=totalPointsWork[i].pntIdx;
		}
	#endif

	// get divider to adjust batches to index size
	//DTYPE * batchDivider = (DTYPE *)malloc(sizeof(DTYPE) * NUMRANDINDEXES);

	for(int i=0; i<NUMTOTALINDEXES; i++) {
        unsigned int count = std::count_if(indexPoints.begin(), indexPoints.end(), [&i](const indexArrayPnt& p) {
            return p.whichIndex == i; // Compare whichIndex with i
        });
        
        // Print the result

		if( i == 0 ) {
			printf("Number of points for original index: \t%d\n", count);
		}
		else if( i < (NUMRANDINDEXES + 1)) {
			printf("Number of points for offset %d: \t\t%d\n", i, count);
		}
		else {
			printf("Number of points for rotation %d: \t%d\n", i - NUMRANDINDEXES , count);
		}

        

		/*
		batchDivider[i] = (DTYPE)count / (DTYPE)indexPoints.size();
		printf("batchDivider %d: %f\n", i, batchDivider[i]);
		*/
	}

	/*
	for(auto elem:indexGroups) {
		printf("%d: %d, %d\n", elem.index, elem.indexmin, elem.indexmax);
		
	}
	*/

	// fill ordered index point ids array
	for( int i=0; i<indexPoints.size(); i++ ) {
		orderedIndexPntIDs[i] = indexPoints[i].pntIdx;
	}

	// output size of each array in megabytes
	unsigned int totalNNonemptyCells = 0;
	for( int i=0; i<NUMTOTALINDEXES; i ++)
	{
		totalNNonemptyCells += allNNonEmptyCells[i];
	}

	printf("\nSize of allIndex: %lu MB", (sizeof(struct grid)*(totalNNonemptyCells))/(1024*1024));
	printf("\nSize of allIndexLookupArr: %lu MB", (sizeof(unsigned int)*(NDdataPoints.size())*(NUMTOTALINDEXES))/(1024*1024));
	printf("\nSize of allGridCellLookupArr: %lu MB", (sizeof(struct gridCellLookup)*(totalNNonemptyCells))/(1024*1024));
	printf("\nSize of allMinArr: %lu MB", (sizeof(DTYPE)*(NUMINDEXEDDIM)*(NUMTOTALINDEXES))/(1024*1024));
	printf("\nSize of allNCells: %lu MB", (sizeof(unsigned int)*(NUMINDEXEDDIM)*(NUMTOTALINDEXES))/(1024*1024));
	printf("\nSize of allNNonEmptyCells: %lu MB\n", (sizeof(unsigned int)*(NUMTOTALINDEXES))/(1024*1024));

	uint64_t totalNeighbors = 0;
	neighborTableLookup *neighborTable = new neighborTableLookup[NDdataPoints.size()];
	std::vector<struct neighborDataPtrs> pointersToNeighbors;

	CTYPE *workCounts = (CTYPE *)malloc(2 * sizeof(CTYPE));
	workCounts[0] = 0;
	workCounts[1] = 0;

	pointersToNeighbors.clear();

	double entire_time_end = omp_get_wtime();

	double tstart = omp_get_wtime();

	double kernelTimeWithoutBatchEstimator;
	kernelTimeWithoutBatchEstimator = distanceTableNDGridBatches(dev_database, DBSIZE, NUMTOTALINDEXES, &epsilon, allIndex, allGridCellLookupArr, allNNonEmptyCells, allMinArr, allNCells, allIndexLookupArr, neighborTable, &pointersToNeighbors, &totalNeighbors, workCounts, orderedIndexPntIDs, &indexGroups, orderedQueryPntIDs, whichIndexPoints);

	double tend = omp_get_wtime();

	printf("\nTime to get neighbors: %f\n", (tend - tstart));

	double totalTime = (tend - entire_time_start) + timeReorderByDimVariance;
	double totalTimeMinusGettingNeighbors = (entire_time_end - entire_time_start) + timeReorderByDimVariance;
	double totalTimeMinusBatchEstimator = (entire_time_end - entire_time_start) + kernelTimeWithoutBatchEstimator + timeReorderByDimVariance;
	printf("\nTotal time: %f\n", totalTime);
	printf("\nTotal time minus getting neighbors: %f\n", totalTimeMinusGettingNeighbors);
	printf("\nTotal time minus batch estimator: %f\n", totalTimeMinusBatchEstimator);

	gpu_stats << totalTimeMinusBatchEstimator << ", " << inputFname << ", " << epsilon << ", " << totalNeighbors << ", GPUNUMDIM/NUMINDEXEDDIM/NUMRANDINDEXES/NUMRANDROTATIONS/NUMPAIRROTATIONS/ILP/STAMP/SORT/REORDER/SHORTCIRCUIT/QUERYREORDER/DTYPE(float/double): " << GPUNUMDIM << ", " << NUMINDEXEDDIM << ", " << NUMRANDINDEXES << ", " << NUMRANDROTATIONS << ", " << NUMPAIRROTATIONS << ", " << ILP << ", " << STAMP << ", " << SORT << ", " << REORDER << ", " << SHORTCIRCUIT << ", " << QUERYREORDER << ", " << STR(DTYPE) << endl;
	gpu_stats.close();

	// remove after testing
	#if TESTSCRIPT == 1
	char test_fname[] = "py_test_stats.txt";
	gpu_stats.open(test_fname, ios::app);
	gpu_stats << inputFname << ',' << epsilon << ',' << NUMRANDINDEXES << ',' << NUMRANDROTATIONS << ',' << totalTime << ',' << totalTimeMinusBatchEstimator << ',' << workCounts[0] << ',' << workCounts[1] << ',' << totalNeighbors << ',' << RANDOMOFFSETSAMEALLDIM << ',' << FIXEDOFFSETALLDIM << ',' << RANDOMOFFSETFOREACHDIM << endl;
	gpu_stats.close();
	#endif

#if PRINTNEIGHBORTABLE == 1
	printNeighborTable(NDdataPoints.size(), neighborTable);
#endif

	delete[] allMinArr;
	delete[] allNCells;
	delete[] allNNonEmptyCells;
	delete[] allIndexLookupArr;
	delete[] allIndex;
	delete[] allGridCellLookupArr;
	delete[] orderedIndexPntIDs;
	delete[] whichIndexPoints;

	cudaFree(dev_database);

	// free(batchDivider);
}
#endif // end #if not Python (standard C version)

void printNeighborTable(unsigned int databaseSize, struct neighborTableLookup *neighborTable)
{

	char fname[] = "DSSJ_out.txt";
	ofstream DSSJ_out;
	DSSJ_out.open(fname, ios::out);

	printf("\n\nOutputting neighbors to: %s\n", fname);
	DSSJ_out << "#data point (line is the point id), neighbor point ids\n";

	for (int i = 0; i < databaseSize; i++)
	{
		// sort to have increasing point IDs
		std::sort(neighborTable[i].dataPtr + neighborTable[i].indexmin, neighborTable[i].dataPtr + neighborTable[i].indexmax + 1);
		for (int j = neighborTable[i].indexmin; j <= neighborTable[i].indexmax; j++)
		{
			DSSJ_out << neighborTable[i].dataPtr[j] << ", ";
		}
		DSSJ_out << "\n";
	}

	DSSJ_out.close();
}

struct cmpStruct
{
	cmpStruct(std::vector<std::vector<DTYPE>> points) { this->points = points; }
	bool operator()(int a, int b)
	{
		return points[a][0] < points[b][0];
	}

	std::vector<std::vector<DTYPE>> points;
};

void populateNDGridIndexAndLookupArray(std::vector<std::vector<DTYPE>> *NDdataPoints, DTYPE epsilon, struct gridCellLookup **gridCellLookupArr, struct grid **index, unsigned int *indexLookupArr, DTYPE *minArr, unsigned int *nCells, uint64_t totalCells, unsigned int *nNonEmptyCells, unsigned int **gridCellNDMask, unsigned int *gridCellNDMaskOffsets, unsigned int *nNDMaskElems, std::unordered_map<uint64_t, std::vector<uint64_t>> &uniqueGridAdjacentCells, std::vector<std::vector<int>> &incrementorVects)
{

	/////////////////////////////////
	// Populate grid lookup array
	// and corresponding indicies in the lookup array
	/////////////////////////////////
	printf("\n\n*****************************\nPopulating Grid Index and lookup array:\n*****************************\n");
	// printf("\nSize of dataset: %lu", NDdataPoints->size());

	///////////////////////////////
	// First, we need to figure out how many non-empty cells there will be
	// For memory allocation
	// Need to do a scan of the dataset and calculate this
	// Also need to keep track of the list of uniquie linear grid cell IDs for inserting into the grid
	///////////////////////////////
	std::set<uint64_t> uniqueGridCellLinearIds;
	std::vector<uint64_t> uniqueGridCellLinearIdsVect;

	for (int i = 0; i < NDdataPoints->size(); i++)
	{
		unsigned int tmpNDCellIdx[NUMINDEXEDDIM];
		for (int j = 0; j < NUMINDEXEDDIM; j++)
		{
			tmpNDCellIdx[j] = (((*NDdataPoints)[i][j] - minArr[j]) / epsilon);
		}
		uint64_t linearID = getLinearID_nDimensions(tmpNDCellIdx, nCells, NUMINDEXEDDIM);
		uniqueGridCellLinearIds.insert(linearID);
	}

	// printf("uniqueGridCellLinearIds: %d",uniqueGridCellLinearIds.size());

	// copy the set to the vector (sets can't do binary searches -- no random access)
	std::copy(uniqueGridCellLinearIds.begin(), uniqueGridCellLinearIds.end(), std::back_inserter(uniqueGridCellLinearIdsVect));

	///////////////////////////////////////////////

	///////////////////////////////
	// get the adjacent cells
	//////////////////////////////

	// find adjacent cells, fill map
	findAdjacentCellIDs(uniqueGridAdjacentCells, incrementorVects, uniqueGridCellLinearIdsVect, nCells);

	///////////////////////////////////////////////

	std::vector<uint64_t> *gridElemIDs;
	gridElemIDs = new std::vector<uint64_t>[uniqueGridCellLinearIds.size()];

	// Create ND array mask:
	// This mask determines which cells in each dimension has points in them.
	std::set<unsigned int> NDArrMask[NUMINDEXEDDIM];

	vector<uint64_t>::iterator lower;

	for (int i = 0; i < NDdataPoints->size(); i++)
	{
		unsigned int tmpNDCellID[NUMINDEXEDDIM];
		for (int j = 0; j < NUMINDEXEDDIM; j++)
		{
			tmpNDCellID[j] = (((*NDdataPoints)[i][j] - minArr[j]) / epsilon);

			// add value to the ND array mask
			NDArrMask[j].insert(tmpNDCellID[j]);
		}

		// get the linear id of the cell
		uint64_t linearID = getLinearID_nDimensions(tmpNDCellID, nCells, NUMINDEXEDDIM);
		// printf("\nlinear id: %d",linearID);
		if (linearID > totalCells)
		{

			printf("\n\nERROR Linear ID is: %lu, total cells is only: %lu\n\n", linearID, totalCells);
		}

		// find the index in gridElemIds that corresponds to this grid cell linear id

		lower = std::lower_bound(uniqueGridCellLinearIdsVect.begin(), uniqueGridCellLinearIdsVect.end(), linearID);
		uint64_t gridIdx = lower - uniqueGridCellLinearIdsVect.begin();
		gridElemIDs[gridIdx].push_back(i);
	}

	///////////////////////////////
	// Here we fill a temporary index with points, and then copy the non-empty cells to the actual index
	///////////////////////////////

	struct grid *tmpIndex = new grid[uniqueGridCellLinearIdsVect.size()];

	int cnt = 0;

	// populate temp index and lookup array

	for (int i = 0; i < uniqueGridCellLinearIdsVect.size(); i++)
	{
		tmpIndex[i].indexmin = cnt;
		for (int j = 0; j < gridElemIDs[i].size(); j++)
		{
			if (j > ((NDdataPoints->size() - 1)))
			{
				printf("\n\n***ERROR Value of a data point is larger than the dataset! %d\n\n", j);
				return;
			}
			indexLookupArr[cnt] = gridElemIDs[i][j];
			cnt++;
		}
		tmpIndex[i].indexmax = cnt - 1;
	}

	// printf("\nExiting grid populate method early!");
	// return;

	printf("\nFull cells: %d (%f, fraction full)", (unsigned int)uniqueGridCellLinearIdsVect.size(), uniqueGridCellLinearIdsVect.size() * 1.0 / double(totalCells));
	printf("\nEmpty cells: %ld (%f, fraction empty)", totalCells - (unsigned int)uniqueGridCellLinearIdsVect.size(), (totalCells - uniqueGridCellLinearIdsVect.size() * 1.0) / double(totalCells));

	*nNonEmptyCells = uniqueGridCellLinearIdsVect.size();

	printf("\nSize of index that would be sent to GPU (GiB) -- (if full index sent), excluding the data lookup arr: %f", (double)sizeof(struct grid) * (totalCells) / (1024.0 * 1024.0 * 1024.0));
	printf("\nSize of compressed index to be sent to GPU (GiB) , excluding the data and grid lookup arr: %f", (double)sizeof(struct grid) * (uniqueGridCellLinearIdsVect.size() * 1.0) / (1024.0 * 1024.0 * 1024.0));

	/////////////////////////////////////////
	// copy the tmp index into the actual index that only has the non-empty cells

	// allocate memory for the index that will be sent to the GPU
	*index = new grid[uniqueGridCellLinearIdsVect.size()];
	*gridCellLookupArr = new struct gridCellLookup[uniqueGridCellLinearIdsVect.size()];

	cmpStruct theStruct(*NDdataPoints);

	for (int i = 0; i < uniqueGridCellLinearIdsVect.size(); i++)
	{
		(*index)[i].indexmin = tmpIndex[i].indexmin;
		(*index)[i].indexmax = tmpIndex[i].indexmax;
		(*gridCellLookupArr)[i].idx = i;
		(*gridCellLookupArr)[i].gridLinearID = uniqueGridCellLinearIdsVect[i];
	}

	printf("\nWhen copying from entire index to compressed index: number of non-empty cells: %lu", uniqueGridCellLinearIdsVect.size());

	// copy NDArrMask from set to an array

	// find the total size and allocate the array

	unsigned int cntNDOffsets = 0;
	unsigned int cntNonEmptyNDMask = 0;
	for (int i = 0; i < NUMINDEXEDDIM; i++)
	{
		cntNonEmptyNDMask += NDArrMask[i].size();
	}
	*gridCellNDMask = new unsigned int[cntNonEmptyNDMask];

	*nNDMaskElems = cntNonEmptyNDMask;

	// copy the offsets to the array
	for (int i = 0; i < NUMINDEXEDDIM; i++)
	{
		// Min
		gridCellNDMaskOffsets[(i * 2)] = cntNDOffsets;
		for (std::set<unsigned int>::iterator it = NDArrMask[i].begin(); it != NDArrMask[i].end(); ++it)
		{
			(*gridCellNDMask)[cntNDOffsets] = *it;
			cntNDOffsets++;
		}
		// max
		gridCellNDMaskOffsets[(i * 2) + 1] = cntNDOffsets - 1;
	}

	delete[] tmpIndex;

} // end function populate grid index and lookup array

// determines the linearized ID for a point in n-dimensions
// indexes: the indexes in the ND array: e.g., arr[4][5][6]
// dimLen: the length of each array e.g., arr[10][10][10]
// nDimensions: the number of dimensions

uint64_t getLinearID_nDimensions(unsigned int *indexes, unsigned int *dimLen, unsigned int nDimensions)
{
	// int i;
	// uint64_t offset = 0;
	// for( i = 0; i < nDimensions; i++ ) {
	//     offset += (uint64_t)pow(dimLen[i],i) * (uint64_t)indexes[nDimensions - (i + 1)];
	// }
	// return offset;

	uint64_t index = 0;
	uint64_t multiplier = 1;
	for (int i = 0; i < nDimensions; i++)
	{
		index += (uint64_t)indexes[i] * multiplier;
		multiplier *= dimLen[i];
	}

	return index;
}

// determines a point given the linearized ID
// concept taken from
// https://stackoverflow.com/a/10904309
void getNDimIndexesFromLinearIdx(unsigned int *indexes, unsigned int *dimLen, unsigned int nDimensions, uint64_t linearId)
{
	// do the process to get linear id but backwards
	for (int i = 0; i < nDimensions; i++)
	{
		indexes[i] = linearId % dimLen[i];
		linearId /= dimLen[i];
	}
}

// min arr- the minimum value of the points in each dimensions - epsilon
// we can use this as an offset to calculate where points are located in the grid
// max arr- the maximum value of the points in each dimensions + epsilon
// returns the time component of sorting the dimensions when SORT=1
void generateNDGridDimensions(DTYPE * database, const unsigned int DBSIZE,  const unsigned int whichDatabase, DTYPE epsilon, DTYPE *minArr, DTYPE *maxArr, unsigned int *nCells, uint64_t *totalCells, DTYPE *indexOffset)
{

	printf("\n\n*****************************\nGenerating grid dimensions.\n*****************************\n");

	printf("\nNumber of dimensions data: %d, Number of dimensions indexed: %d", GPUNUMDIM, NUMINDEXEDDIM);

	// make the min/max values for each grid dimension the first data element
	for (int j = 0; j < NUMINDEXEDDIM; j++)
	{
		minArr[j] = database[(DBSIZE*GPUNUMDIM*whichDatabase) + j];
		maxArr[j] = database[(DBSIZE*GPUNUMDIM*whichDatabase) + j];
	}

	for (int i = 1; i < DBSIZE; i++)
	{
		for (int j = 0; j < NUMINDEXEDDIM; j++)
		{
			DTYPE pointDimVal = database[(DBSIZE*GPUNUMDIM*whichDatabase) + (i * GPUNUMDIM) + j];
			if (pointDimVal < minArr[j])
			{
				minArr[j] = pointDimVal;
			}
			if (pointDimVal > maxArr[j])
			{
				maxArr[j] = pointDimVal;
			}
		}
	}

	printf("\n");
	for (int j = 0; j < NUMINDEXEDDIM; j++)
	{
		printf("Data Dim: %d, min/max: %f,%f\n", j, minArr[j], maxArr[j]);
	}

	// add buffer around each dim so no weirdness later with putting data into cells
	for (int j = 0; j < NUMINDEXEDDIM; j++)
	{
		minArr[j] -= epsilon;

		maxArr[j] += epsilon;
	}

	// change min array by offset
	for (int j = 0; j < NUMINDEXEDDIM; j++)
	{
		#if RANDOMOFFSETSAMEALLDIM == 1 || FIXEDOFFSETALLDIM == 1
		minArr[j] -= (*indexOffset);
		#elif RANDOMOFFSETFOREACHDIM == 1
		minArr[j] -= indexOffset[j];
		#endif
	}

	for (int j = 0; j < NUMINDEXEDDIM; j++)
	{
		printf("Appended by epsilon Dim: %d, min/max: %f,%f\n", j, minArr[j], maxArr[j]);
	}

	// calculate the number of cells:
	for (int j = 0; j < NUMINDEXEDDIM; j++)
	{
		nCells[j] = ceil((maxArr[j] - minArr[j]) / epsilon);
		printf("Number of cells dim: %d: %d\n", j, nCells[j]);
	}

	// calc total cells: num cells in each dim multiplied
	uint64_t tmpTotalCells = nCells[0];
	for (int j = 1; j < NUMINDEXEDDIM; j++)
	{
		tmpTotalCells *= nCells[j];
	}

	*totalCells = tmpTotalCells;
}

// reorders the input data by variance of each dimension
void ReorderByDimension(std::vector<std::vector<DTYPE>> *NDdataPoints)
{

	double tstart_sort = omp_get_wtime();
	DTYPE sums[GPUNUMDIM];
	DTYPE average[GPUNUMDIM];
	struct dim_reorder_sort dim_variance[GPUNUMDIM];
	for (int i = 0; i < GPUNUMDIM; i++)
	{
		sums[i] = 0;
		average[i] = 0;
	}

	DTYPE greatest_variance = 0;
	int greatest_variance_dim = 0;

	int sample = 100;
	DTYPE inv_sample = 1.0 / (sample * 1.0);
	printf("\nCalculating variance based on on the following fraction of pts: %f", inv_sample);
	double tvariancestart = omp_get_wtime();
	// calculate the variance in each dimension
	for (int i = 0; i < GPUNUMDIM; i++)
	{
		// first calculate the average in the dimension:
		// only use every 10th point
		for (int j = 0; j < (*NDdataPoints).size(); j += sample)
		{
			sums[i] += (*NDdataPoints)[j][i];
		}

		average[i] = (sums[i]) / ((*NDdataPoints).size() * inv_sample);
		// printf("\nAverage in dim: %d, %f",i,average[i]);

		// Next calculate the std. deviation
		sums[i] = 0; // reuse this for other sums
		for (int j = 0; j < (*NDdataPoints).size(); j += sample)
		{
			sums[i] += (((*NDdataPoints)[j][i]) - average[i]) * (((*NDdataPoints)[j][i]) - average[i]);
		}

		dim_variance[i].variance = sums[i] / ((*NDdataPoints).size() * inv_sample);
		dim_variance[i].dim = i;

		// printf("\nDim:%d, variance: %f",dim_variance[i].dim,dim_variance[i].variance);

		if (greatest_variance < dim_variance[i].variance)
		{
			greatest_variance = dim_variance[i].variance;
			greatest_variance_dim = i;
		}
	}

	// double tvarianceend=omp_get_wtime();
	// printf("\nTime to compute variance only: %f",tvarianceend - tvariancestart);
	// sort based on variance in dimension:

	// double tstartsortreorder=omp_get_wtime();
	std::sort(dim_variance, dim_variance + GPUNUMDIM, compareByDimVariance);

	for (int i = 0; i < GPUNUMDIM; i++)
	{
		printf("\nReodering dimension by: dim: %d, variance: %f", dim_variance[i].dim, dim_variance[i].variance);
	}

	printf("\nDimension with greatest variance: %d", greatest_variance_dim);

	// copy the database
	//  double * tmp_database= (double *)malloc(sizeof(double)*(*NDdataPoints).size()*(GPUNUMDIM));
	//  std::copy(database, database+((*DBSIZE)*(GPUNUMDIM)),tmp_database);
	std::vector<std::vector<DTYPE>> tmp_database;

	// copy data into temp database
	tmp_database = (*NDdataPoints);

#pragma omp parallel for num_threads(5) shared(NDdataPoints, tmp_database)
	for (int j = 0; j < GPUNUMDIM; j++)
	{

		int originDim = dim_variance[j].dim;
		for (int i = 0; i < (*NDdataPoints).size(); i++)
		{
			(*NDdataPoints)[i][j] = tmp_database[i][originDim];
		}
	}

	double tend_sort = omp_get_wtime();
	// double tendsortreorder=omp_get_wtime();
	// printf("\nTime to sort/reorder only: %f",tendsortreorder-tstartsortreorder);
	double timecomponent = tend_sort - tstart_sort;
	printf("\nTime to reorder cols by variance (this gets added to the time because its an optimization): %f", timecomponent);
}

// computes the number of distance calculations for each point
void computeNumDistanceCalcs(std::vector<workArrayPnt> *totalPointsWork, unsigned int *nNonEmptyCells, gridCellLookup *gridCellLookupArr, grid *index, std::unordered_map<uint64_t, std::vector<uint64_t>> *uniqueGridAdjacentCells, std::vector<std::vector<DTYPE>> *NDdataPoints, DTYPE *minArr, unsigned int *nCells, DTYPE &epsilon)
{
	// number of points in each cell
	std::unordered_map<std::uint64_t, unsigned int> cellPts;

	// number of distance calculations per cell
	std::unordered_map<std::uint64_t, unsigned int> totalCellsWork;

	// loop over each non-empty cell and find the points contained within
	// record the number of points in the cell
	for (int i = 0; i < *nNonEmptyCells; i++)
	{
		uint64_t cellID = gridCellLookupArr[i].gridLinearID;
		unsigned int numPtsInCell = (index[i].indexmax - index[i].indexmin) + 1;

		cellPts[cellID] = numPtsInCell;
	}

	// loop through each cell, get number of distance calculations
	for (const auto &mainCell : cellPts)
	{
		unsigned int totalWork = 0;
		uint64_t linearCellID = mainCell.first;

		// loop through all adjacent cells
		for (const auto &adjCell : (*uniqueGridAdjacentCells)[linearCellID])
		{
			totalWork += cellPts[adjCell];
		}
		totalCellsWork[linearCellID] = totalWork;
	}

	// loop through each data point, find which cell it is in and map number of distance calculations
	for (int i = 0; i < NDdataPoints->size(); i++)
	{
		unsigned int tmpNDCellIdx[NUMINDEXEDDIM];
		for (int j = 0; j < NUMINDEXEDDIM; j++)
		{
			tmpNDCellIdx[j] = (((*NDdataPoints)[i][j] - minArr[j]) / epsilon);
		}
		uint64_t linearID = getLinearID_nDimensions(tmpNDCellIdx, nCells, NUMINDEXEDDIM);
		workArrayPnt tmp;
		tmp.pntIdx = i;
		tmp.numDistCalcs = totalCellsWork[linearID];
		totalPointsWork->push_back(tmp);
	}

	std::sort(totalPointsWork->begin(), totalPointsWork->end(), compareWorkArrayByNumDistanceCalcs);

	return;
}

// finds the adjacent cells of cell (including the cell itself)
void findAdjacentCellIDs(std::unordered_map<uint64_t, std::vector<uint64_t>> &uniqueGridAdjacentCells, std::vector<std::vector<int>> &incrementorVects, std::vector<uint64_t> &uniqueGridCellLinearIdsVect, unsigned int *nCells)
{
	unsigned int pointIdx[NUMINDEXEDDIM];

	// loop through each nonempty linear grid ID
	for (uint64_t linearID : uniqueGridCellLinearIdsVect)
	{
		// get index of grid point
		getNDimIndexesFromLinearIdx(pointIdx, nCells, NUMINDEXEDDIM, linearID);

		// check that linear ID does not already have adjacent cells caluclated for it
		unsigned int adjCellIdx[NUMINDEXEDDIM];
		std::vector<uint64_t> uniqueGridAdjacentCellsSingleCell;

		for (std::vector<int> incrementorVect : incrementorVects)
		{
			// "vector" addition
			for (int j = 0; j < NUMINDEXEDDIM; j++)
			{
				int entr_sum = pointIdx[j] + incrementorVect[j];
				// check if cell is in bounds
				adjCellIdx[j] = entr_sum;
			}

			// convert index back to cell id
			uint64_t adjLinearID = getLinearID_nDimensions(adjCellIdx, nCells, NUMINDEXEDDIM);

			// check if cell is nonempty
			if (std::binary_search(uniqueGridCellLinearIdsVect.begin(), uniqueGridCellLinearIdsVect.end(), adjLinearID))
			{
				// add to adjacent cells
				uniqueGridAdjacentCellsSingleCell.push_back(adjLinearID);
			}
		}

		uniqueGridAdjacentCells[linearID] = uniqueGridAdjacentCellsSingleCell;
	}
}