#include <cuda_runtime.h>
#include <cufft.h>

#include "utils.h"
#include "cuda_utils.h"
#include "managed_allocator.h"

#include <thrust/generate.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <algorithm>
#include <random>

// create a nickname for vectors which use a managed_allocator
template<class T>
using managed_vector = std::vector<T, managed_allocator<T>>;

__global__ void sliding_window( 
      cufftComplex* __restrict__ sums, 
      const cufftComplex* __restrict__ vals, 
      const int window_size, 
      const int num_sums 
) {

   int global_index = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = gridDim.x * blockDim.x;
   for ( int index = global_index; index < num_sums; index+=stride ) {
      for ( int w_index = 0; w_index < window_size; w_index++ ) {
         sums[index] = cuCaddf( sums[index], vals[index + w_index] );
      }
   }
}


cufftComplex gen_rand_cufftComplex() {
   static std::default_random_engine r_engine;
   static std::uniform_real_distribution<float> udist{-50.0, 50.0}; // range 0 - 50

   return cufftComplex{(float)udist(r_engine),(float)udist(r_engine)};
}

template <class T>
void print_man_vec( const managed_vector<T>& vals, const char* prefix = "" ) {
   std::cout << prefix;
   std::for_each(vals.begin(), vals.end(), [](const cufftComplex &n){ std::cout << "{ " << n.x << ", " << n.y << " }\n"; });
}


int main( int argc, char** argv) {
   try {
      int num_vals = 1 << 21;
      int window_size = 1 << 7;
      int num_sums = num_vals - window_size;
      
      int threads_per_block = 1024;
      int num_blocks = (num_vals + threads_per_block - 1) / threads_per_block;
      cudaError_t cerror = cudaSuccess;
      bool debug = false;
         
      managed_vector<cufftComplex> vals( num_vals );
      managed_vector<cufftComplex> exp_sums( num_sums );
      managed_vector<cufftComplex> sums( num_sums );

      std::generate( vals.begin(), vals.end(), gen_rand_cufftComplex );
      
      if ( debug )
         print_man_vec<cufftComplex>( vals, "Vals:\n" );

      std::fill( sums.begin(), sums.end(), cufftComplex{0,0} );
      
      sliding_window<<<num_blocks, threads_per_block>>>( sums.data(), vals.data(), 
            window_size, num_sums );

      try_cuda_func_throw( cerror, cudaDeviceSynchronize() );

      if ( debug )
         print_man_vec<cufftComplex>( sums, "Sumss:\n" );

      
      return 0;
   } catch( std::exception& ex ) {
      std::cout << __func__ << "(): ERROR: " << ex.what() << "\n";
   }
}


