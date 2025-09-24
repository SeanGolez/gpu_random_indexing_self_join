//Notes on optimizations:

//Optimizations STAMP and LINEARSTAMP in the following papers:
//1) Gowanlock, Michael, and Karsin, Ben. "Accelerating the similarity self-join using the GPU." 
//Journal of parallel and distributed computing 133 (2019): 107-123.
//2) Gowanlock, Michael, and Karsin, Ben. "GPU accelerated self-join for the distance similarity metric." 
//2018 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW). IEEE, 2018.

//Optimizations SORT, REORDER, SHORTCIRCUIT in the following paper:
//1) Gowanlock, Michael, and Karsin, Ben. "GPU-Accelerated Similarity Self-Join for Multi-Dimensional Data." 
//Proceedings of the 15th International Workshop on Data Management on New Hardware. 2019.

//Optimizations ILP, QUERYREORDER
//In a paper under reivew (to be updated with reference upon acceptance)

//Kernel block size
#define BLOCKSIZE 256
 
//Number of dimensions of the data (n)
#define GPUNUMDIM 3

//Number of indexed dimensions (k)
#define NUMINDEXEDDIM 3

//data type of the input dataset (float or double)
#define DTYPE float


///////////////////////
//Utility
//used for outputting the neighbortable at the end
#define PRINTNEIGHBORTABLE 0
///////////////////////


///////////////////////
//Optimizations:

//unidirectional comparison (unicomp)
#define STAMP 0

//Another version of unicomp in JPDC2019 paper (worse performance than unicomp)
#define LINEARSTAMP 0 

//Sortidu
#define SORT 0

//Reorder the data by dimensionality
#define REORDER 1

//For ILP in distance calculations
#define ILP 8 //0-default no ILP
			  //The number is the number of registers/cached elements


//Short circuit the distance calculation
#define SHORTCIRCUIT 1

//Reorder the query points by work
#define QUERYREORDER 0 //note: Keep this off for probe-and-sort

//End optimizations
///////////////////////

///////////////////////
//Flags used in performance evaluations for papers
//Do not use when timing algorithm
#define SEARCHFILTTIME 0

//used to see how many point comparisons and grid cell searches
//For performance evaluation purposes, and not when timing the algorithm
#define COUNTMETRICS 0

//Data type for the above
#define CTYPE unsigned long long
///////////////////////
 
///////////////////////
//Batching scheme

//Result set buffer size, one buffer of this size per GPU stream
#define GPUBUFFERSIZE 100000000 //Default 100000000

//number of concurrent gpu streams
#define GPUSTREAMS 3 //Default 3

//Minimum number of batches (used to mitigate against pinned memory dominating response time for low epsilon)
//See performance model in JPDC2019 paper
#define MINBATCHES 3 //Default 3

//Fraction of dataset sampled to estimate result set size
//Can be increased if estimate of the total result set is inaccurate. 
#define SAMPLERATE 0.015 //Default 0.015
						 
//end batching scheme					
///////////////////////

// Number of bytes to allocate for key value pairs in GiB
#define KEYVALUEMEM 210

//probe-and-sort
#define PROBEANDSORT 1 //0 sort everything in unified memory at the end
					   //1 do probe-and-sort
					   //-1 no sorting at all (as a baseline)

#define SLEEPSEC 2 //Number of seconds to wait until probing the count produced on the GPU

#define PAGESIZE 4 //Unified Memory page size (in KiB)

#define FRACTIONTHREADSCOMPLETE 0.05 //this is used as a threshold to determine when to sort on the CPU
//end probe-and-sort


//////////////////////////
//parallel CPU parameters for OpenMP
#define NCOPYTHREADS 2 //number of threads for parallel scan operations: e.g., copies on the CPU and reductions
					//The number of threads should be obtained from stream benchmarks or similar.

//end parallel CPU parameters for OpenMP
//////////////////////////