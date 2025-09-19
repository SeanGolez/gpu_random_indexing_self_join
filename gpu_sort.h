#include "structs.h"
#include "params.h"
#include "tbb/tbb.h"
#include "tbb/concurrent_queue.h"

struct rangeorder{
uint64_t rangelower;
uint64_t rangeupper;
};

void gpuRadixSortAndCpuMerge(unsigned int * unorderedPointIDKey, unsigned int * unorderedPointInDistValue, unsigned int * orderedPointIDKey, unsigned int * orderedPointInDistValue, unsigned long long int * N );

void mergeConsumerWithRanges(unsigned int * orderedPointIDKey, unsigned int * orderedPointInDistValue, uint64_t lower1, uint64_t upper1,uint64_t lower2, uint64_t upper2);

void workStealSortingABatch(unsigned int * unorderedPointIDKey, unsigned int * unorderedPointInDistValue, unsigned int * orderedPointIDKey, unsigned int * orderedPointInDistValue, uint64_t batchsize);

bool getMergeRangesForMergingPairsForMultiway(bool * batchCompleteArr, tbb::concurrent_queue<rangeorder> * rangeQueue, unsigned long long int *N, int NUMBATCHES,  uint64_t * lower1, uint64_t * upper1, uint64_t * lower2, uint64_t * upper2);

void multiwayMergeBatchesAfterPipeline(unsigned long long * N, tbb::concurrent_queue<rangeorder> * rangeQueue, unsigned int * orderedPointIDKey, unsigned int * orderedPointInDistValue);