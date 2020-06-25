#include <cuda_runtime.h>
#include <cufft.h>

#include "managed_allocator.h"

#include <thrust/generate.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <algorithm>
#include <random>

// create a nickname for vectors which use a managed_allocator
template<class T>
using managed_vector = std::vector<T, managed_allocator<T>>;

__global__ void sliding_window( cufftComplex* __restrict__ sums, const cufftComplex* __restrict__ vals, const int window_size, const int num_sums ) {

   int global_index = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = gridDim.x * blockDim.x;
   for ( int index = global_index; index < num_sums; index+=stride ) {
      for ( int w_index = 0; w_index < window_size; w_index++ ) {
         // sums[index].x += vals[index + w_index].x;
         // sums[index].y += vals[index + w_index].y;
         sums[index] = cuCaddf( sums[index], vals[index + w_index] );
      }
   }
}


cufftComplex gen_rand_cufftComplex() {
   static std::default_random_engine r_engine;
   static std::uniform_real_distribution<float> udist{0.0, 50.0}; // range 0 - 50

   /*cufftComplex result{(float)udist(r_engine),(float)udist(r_engine)};*/
   return cufftComplex{(float)udist(r_engine),(float)udist(r_engine)};
}


inline void cout_cufftComplex( const cufftComplex& val ) {
   std::cout << "{ " << val.x << ", " << val.y << " }\n";
}


int main( int argc, char** argv) {
   try {
      // There are 5 SMs on my laptop's GeForce GTX 1050 (Pascal microarch)
      // max of 2048 threads per SM
      int num_vals = 1024;
      int window_size = 64;
      int num_sums = num_vals - window_size;
      
      int threads_per_block = 256;
      int num_blocks = (num_vals + threads_per_block - 1) / threads_per_block;
      
      managed_vector<cufftComplex> vals( num_vals );
      managed_vector<cufftComplex> exp_sums( num_sums );
      managed_vector<cufftComplex> sums( num_sums );

      for( size_t index = 0; index != vals.size(); ++index ) {
         vals.at( index ).x = (float)index;   
         vals.at( index ).y = (float)index;   
      } 
      
      for( size_t index = 0; index != sums.size(); ++index ) {
         exp_sums.at( index ).x = 0;   
         exp_sums.at( index ).y = 0;   
         sums.at( index ).x = 0;   
         sums.at( index ).y = 0;   
      } 

      std::cout << "Vals:\n"; 
      for( size_t index = 0; index != vals.size(); ++index ) {
         cout_cufftComplex( vals[index] );  
      } 
      std::cout << "\n";

      sliding_window<<<num_blocks, threads_per_block>>>( sums.data(), vals.data(), 
            window_size, num_sums );

      cudaDeviceSynchronize();

      std::cout << "Sums:\n"; 
      for( size_t index = 0; index != sums.size(); ++index ) {
         cout_cufftComplex( sums[index] );  
      } 
      std::cout << "\n";
   
      return 0;
   } catch( std::exception& ex ) {
      std::cout << __func__ << "(): ERROR: " << ex.what() << "\n";
   }
}


