#include <cuda_runtime.h>
#include <cufft.h>
#include "managed_allocator.h"
#include <thrust/generate.h>
#include <thrust/transform.h>

template<class T>
using managed_device_vector = thrust::device_vector<T, managed_allocator<T, cudaMemAttachGlobal>>;

template<class T>
using managed_host_vector = thrust::device_vector<T, managed_allocator<T, cudaMemAttachHost>>;

int main( int argc, char** argv) {

   int num_vals = 1000;
   int window_size = 40;
   int num_sums = num_vals - window_size;
   managed_device_vector<float2> mdv( num_vals );
   managed_host_vector<float2> mhv( num_sums );

   return 0;
}


