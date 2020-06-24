#include <cuda_runtime.h>

#include "managed_allocator.h"

template<class T>
using managed_device_vector = thrust::device_vector<T, managed_allocator<T, cudaMemAttachGlobal>>;

template<class T>
using managed_host_vector = thrust::device_vector<T, managed_allocator<T, cudaMemAttachHost>>;

int main( int argc, char** argv) {

   int num_vals = 1000;
   int window_size = 40;
   int num_sums = num_vals - window_size;
   managed_device_vector<float2> mdv;
   managed_host_vector<float2> mhv;

   mdv.reserve( num_vals );
   mhv.reserve( num_sums );

   return 0;
}


