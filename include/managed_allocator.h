#ifndef __MANAGED_ALLOCATOR_H__
#define __MANAGED_ALLOCATOR_H__

#include "utils.h"
#include <cuda_runtime.h>

#include <thrust/device_malloc_allocator.h>
#include <thrust/device_vector.h>

#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

// CMFlag is one of:
// cudaMemAttachGlobal
// cudaMemAttachHost

template<class T, unsigned int CMFlag>
class managed_allocator : public thrust::device_malloc_allocator<T>
{
  public:
    using value_type = T;

    typedef thrust::device_ptr<T>  pointer;
    inline pointer allocate(size_t n)
    {
      value_type* result = nullptr;
  
      cudaError_t error = cudaMallocManaged(&result, n*sizeof(T), CMFlag);
  
      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "managed_allocator::allocate(): cudaMallocManaged");
      }
  
      return thrust::device_pointer_cast(result);
    }
  
    inline void deallocate(pointer ptr, size_t)
    {
      cudaError_t error = cudaFree(thrust::raw_pointer_cast(ptr));
  
      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "managed_allocator::deallocate(): cudaFree");
      }
    }
};



#endif // end of #ifndef __MANAGED_ALLOCATOR_H__
