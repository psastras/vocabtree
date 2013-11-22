#pragma once

/// This header is automatically generated.  Do not modify.


#define ENABLE_MPI @ENABLE_MPI@
#define ENABLE_FASTCLUSTER @ENABLE_FASTCLUSTER@
#define ENABLE_MULTITHREADING @ENABLE_MULTITHREAD@

// needed for sublime clang to work
#if ENABLE_MULTITHREADING & ENABLE_OPENMP
	#ifdef __clang__
		#if __has_include(<omp.h>)
			#define ENABLE_OPENMP @ENABLE_OPENMP@
		#else
			#define ENABLE_OPENMP 0
		#endif
	#else
		#define ENABLE_OPENMP @ENABLE_OPENMP@
	#endif
#else
	#define ENABLE_OPENMP 0
#endif
