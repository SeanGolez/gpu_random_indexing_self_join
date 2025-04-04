#include "structs.h"
#include "params.h"



void makeDistanceTableGPUBruteForce(std::vector<std::vector <DTYPE> > * NDdataPoints, DTYPE* epsilon, struct table * neighborTable, unsigned long long int * totalNeighbors);

void distanceTableNDGridBatches(DTYPE * dev_database, const unsigned int DBSIZE, unsigned int NUMTOTALINDEXES, DTYPE * epsilon, struct grid * allIndex, 
	struct gridCellLookup * allGridCellLookupArr, unsigned int * allNNonEmptyCells, DTYPE* allMinArr, unsigned int * allNCells, 
	unsigned int * allIndexLookupArr, struct neighborTableLookup * neighborTable, std::vector<struct neighborDataPtrs> * pointersToNeighbors, 
	uint64_t * totalNeighbors, CTYPE* workCounts, unsigned int * orderedIndexPntIDs, std::vector<indexArrayPntGroups> * indexGroups, unsigned int * orderedQueryPntIDs,
	unsigned int * whichIndexPoints);


unsigned long long callGPUBatchEst(unsigned int DBSIZE, DTYPE* dev_database, unsigned int NUMTOTALINDEXES, DTYPE epsilon, unsigned int * dev_whichIndexPoints, struct grid * dev_grid, 
	unsigned int * dev_indexLookupArr, struct gridCellLookup * dev_gridCellLookupArr, DTYPE* dev_minArr, unsigned int * dev_nCells, unsigned int * dev_nNonEmptyCells, 
	unsigned int * dev_orderedQueryPntIDs, gridCellLookup ** dev_startGridPtrs, gridCellLookup ** dev_stopGridPtrs, grid ** dev_startIndexPtrs,
	unsigned int * retNumBatches, unsigned int * retGPUBufferSize);

void constructNeighborTableKeyValueWithPtrs(keyValPair * keyDistPairs, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned long long int * cnt);
void constructNeighborTableKeyValueWithPtrs(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned int * cnt);

void warmUpGPU();

void populateNDGridIndexAndLookupArrayGPU(DTYPE * dev_database, const unsigned int DBSIZE,  const unsigned int whichDatabase, DTYPE *epsilon, DTYPE* minArr,  uint64_t totalCells, unsigned int * nCells, struct gridCellLookup ** gridCellLookupArr, struct grid ** index, unsigned int * indexLookupArr, unsigned int *nNonEmptyCells, std::vector<std::vector<int>> *incrementorVects, std::vector<workArrayPnt> *totalPointsWork);

//for the brute force version without batches
void constructNeighborTableKeyValue(int * pointIDKey, int * pointInDistValue, struct table * neighborTable, unsigned int * cnt);




//Unicomp requires multiple updates to the neighbortable for a given point
//This allows updating the neighbortable for the same point
void constructNeighborTableKeyValueWithPtrsWithMultipleUpdates(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned int * cnt, pthread_mutex_t * pointLocks);

//Unicomp requires multiple updates to the neighbortable for a given point
//This allows updating the neighbortable for the same point
//WITHOUT VECTORS FOR DATA
void constructNeighborTableKeyValueWithPtrsWithMultipleUpdatesMultipleDataArrays(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned int * cnt, int * uniqueKeys, int * uniqueKeyPosition, unsigned int numUniqueKeys);

//Unicomp requires multiple updates to the neighbortable for a given point
//This allows updating the neighbortable for the same point
//Without locks
void constructNeighborTableKeyValueWithPtrsBatchMaskArray(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned int * cnt, int batchNum);

//Sort the queries by their workload based on the number of points in the cell
//From hybrid KNN paper in GPGPU'19 
void computeWorkDifficulty(unsigned int * outputOrderedQueryPntIDs, struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, unsigned int * indexLookupArr, struct grid * index);


void rotateOnGPU(DTYPE * dev_database, const unsigned int DBSIZE, const unsigned int whichDatabase);