#include <cuda_runtime.h>
#include <cufft.h>
#include "managed_allocator.h"
#include <thrust/generate.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <algorithm>
#include <random>

template<class T>
using managed_device_vector = thrust::device_vector<T, managed_allocator<T, cudaMemAttachGlobal>>;


template<class T>
using managed_host_vector = thrust::device_vector<T, managed_allocator<T, cudaMemAttachHost>>;


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

   int num_vals = 1000;
   int window_size = 40;
   int num_sums = num_vals - window_size;
   
   managed_device_vector<cufftComplex> vals( num_vals );
   managed_device_vector<cufftComplex> sums( num_sums );

   for( size_t index = 0; index != vals.size(); ++index ) {
      vals[index] = gen_rand_cufftComplex();
   } 

   std::cout << "Vals:\n"; 
   for( size_t index = 0; index != vals.size(); ++index ) {
      cout_cufftComplex( vals[index] );  
   } 
   std::cout << "\n";

   return 0;
}


