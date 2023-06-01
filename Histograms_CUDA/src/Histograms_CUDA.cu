#include "Histograms_CUDA.h"

///////////////////////////////////////////////////////
//  ____                                 __   ____   //
// |  _ \ _____      _____ _ __    ___  / _| |___ \  //
// | |_) / _ \ \ /\ / / _ \ '__|  / _ \| |_    __) | //
// |  __/ (_) \ V  V /  __/ |    | (_) |  _|  / __/  //
// |_|   \___/ \_/\_/ \___|_|     \___/|_|   |_____| //
//                                                   //
//   ____          _            _   _                //
//  |  _ \ ___  __| |_   _  ___| |_(_) ___  _ __     //
//  | |_) / _ \/ _` | | | |/ __| __| |/ _ \| '_ \    //
//  |  _ <  __/ (_| | |_| | (__| |_| | (_) | | | |   //
//  |_| \_\___|\__,_|\__,_|\___|\__|_|\___/|_| |_|   //
//                                                   //
///////////////////////////////////////////////////////
                                                      
__global__ void reduction_kernel_float(uint64_t N, float* in)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N)
	{
		in[i] += in[i+N];
	}
}

__global__ void reduction_kernel_double(uint64_t N, double* in)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N)
	{
		in[i] += in[i+N];
	}
}

__global__ void reduction_kernel_uint8(uint64_t N, uint8_t* in)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N)
	{
		in[i] += in[i+N];
	}
}

__global__ void reduction_kernel_uint16(uint64_t N, uint16_t* in)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N)
	{
		in[i] += in[i+N];
	}
}

__global__ void reduction_kernel_uint32(uint64_t N, uint32_t* in)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N)
	{
		in[i] += in[i+N];
	}
}

__global__ void reduction_kernel_uint64(uint64_t N, uint64_t* in)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N)
	{
		in[i] += in[i+N];
	}
}

__global__ void reduction_kernel_int8(uint64_t N, int8_t* in)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N)
	{
		in[i] += in[i+N];
	}
}

__global__ void reduction_kernel_int16(uint64_t N, int16_t* in)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N)
	{
		in[i] += in[i+N];
	}
}

__global__ void reduction_kernel_int32(uint64_t N, int32_t* in)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N)
	{
		in[i] += in[i+N];
	}
}

__global__ void reduction_kernel_int64(uint64_t N, int64_t* in)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N)
	{
		in[i] += in[i+N];
	}
}


template<class DataType> 
void reduction(uint64_t N, DataType* in, uint64_t size){}

template<>
void reduction<float>(uint64_t N, float* in, uint64_t size)
{
	if (N/2 >= size)
	{
		reduction_kernel_float<<<N/1024+1,512>>>(N/2,in);
		reduction<float>(N/2,in,size);
	}
}

template<>
void reduction<double>(uint64_t N, double* in, uint64_t size)
{
	if (N/2 >= size)
	{
		reduction_kernel_double<<<N/1024+1,512>>>(N/2,in);
		reduction<double>(N/2,in,size);
	}
}

template<>
void reduction<uint8_t>(uint64_t N, uint8_t* in, uint64_t size)
{
	if (N/2 >= size)
	{
		reduction_kernel_uint8<<<N/1024+1,512>>>(N/2,in);
		reduction<uint8_t>(N/2,in,size);
	}
}

template<>
void reduction<uint16_t>(uint64_t N, uint16_t* in, uint64_t size)
{
	if (N/2 >= size)
	{
		reduction_kernel_uint16<<<N/1024+1,512>>>(N/2,in);
		reduction<uint16_t>(N/2,in,size);
	}
}

template<>
void reduction<uint32_t>(uint64_t N, uint32_t* in, uint64_t size)
{
	if (N/2 >= size)
	{
		reduction_kernel_uint32<<<N/1024+1,512>>>(N/2,in);
		reduction<uint32_t>(N/2,in,size);
	}
}

template<>
void reduction<uint64_t>(uint64_t N, uint64_t* in, uint64_t size)
{
	if (N/2 >= size)
	{
		reduction_kernel_uint64<<<N/1024+1,512>>>(N/2,in);
		reduction<uint64_t>(N/2,in,size);
	}
}

template<>
void reduction<int8_t>(uint64_t N, int8_t* in, uint64_t size)
{
	if (N/2 >= size)
	{
		reduction_kernel_int8<<<N/1024+1,512>>>(N/2,in);
		reduction<int8_t>(N/2,in,size);
	}
}

template<>
void reduction<int16_t>(uint64_t N, int16_t* in, uint64_t size)
{
	if (N/2 >= size)
	{
		reduction_kernel_int16<<<N/1024+1,512>>>(N/2,in);
		reduction<int16_t>(N/2,in,size);
	}
}

template<>
void reduction<int32_t>(uint64_t N, int32_t* in, uint64_t size)
{
	if (N/2 >= size)
	{
		reduction_kernel_int32<<<N/1024+1,512>>>(N/2,in);
		reduction<int32_t>(N/2,in,size);
	}
}

template<>
void reduction<int64_t>(uint64_t N, int64_t* in, uint64_t size)
{
	if (N/2 >= size)
	{
		reduction_kernel_int64<<<N/1024+1,512>>>(N/2,in);
		reduction<int64_t>(N/2,in,size);
	}
}

///////////////////////////////////////////////////////
//   ____          _            _   _                //
//  |  _ \ ___  __| |_   _  ___| |_(_) ___  _ __     //
//  | |_) / _ \/ _` | | | |/ __| __| |/ _ \| '_ \    //
//  |  _ <  __/ (_| | |_| | (__| |_| | (_) | | | |   //
//  |_| \_\___|\__,_|\__,_|\___|\__|_|\___/|_| |_|   //
//                                                   //
///////////////////////////////////////////////////////

__global__ void reduction_general_kernel_float(uint64_t N, float* in, uint64_t size)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	int64_t howmany = N/size;
	if(i<size)
	{
		for(int64_t j=1;j<howmany;j++)
		{
			in[i] += in[i+j*size];
		}
	}
}

__global__ void reduction_general_kernel_double(uint64_t N, double* in, uint64_t size)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	int64_t howmany = N/size;
	if(i<size)
	{
		for(int64_t j=1;j<howmany;j++)
		{
			in[i] += in[i+j*size];
		}
	}
}

__global__ void reduction_general_kernel_uint8(uint64_t N, uint8_t* in, uint64_t size)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	int64_t howmany = N/size;
	if(i<size)
	{
		for(int64_t j=1;j<howmany;j++)
		{
			in[i] += in[i+j*size];
		}
	}
}

__global__ void reduction_general_kernel_uint16(uint64_t N, uint16_t* in, uint64_t size)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	int64_t howmany = N/size;
	if(i<size)
	{
		for(int64_t j=1;j<howmany;j++)
		{
			in[i] += in[i+j*size];
		}
	}
}

__global__ void reduction_general_kernel_uint32(uint64_t N, uint32_t* in, uint64_t size)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	int64_t howmany = N/size;
	if(i<size)
	{
		for(int64_t j=1;j<howmany;j++)
		{
			in[i] += in[i+j*size];
		}
	}
}

__global__ void reduction_general_kernel_uint64(uint64_t N, uint64_t* in, uint64_t size)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	int64_t howmany = N/size;
	if(i<size)
	{
		for(int64_t j=1;j<howmany;j++)
		{
			in[i] += in[i+j*size];
		}
	}
}

__global__ void reduction_general_kernel_int8(uint64_t N, int8_t* in, uint64_t size)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	int64_t howmany = N/size;
	if(i<size)
	{
		for(int64_t j=1;j<howmany;j++)
		{
			in[i] += in[i+j*size];
		}
	}
}

__global__ void reduction_general_kernel_int16(uint64_t N, int16_t* in, uint64_t size)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	int64_t howmany = N/size;
	if(i<size)
	{
		for(int64_t j=1;j<howmany;j++)
		{
			in[i] += in[i+j*size];
		}
	}
}

__global__ void reduction_general_kernel_int32(uint64_t N, int32_t* in, uint64_t size)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	int64_t howmany = N/size;
	if(i<size)
	{
		for(int64_t j=1;j<howmany;j++)
		{
			in[i] += in[i+j*size];
		}
	}
}

__global__ void reduction_general_kernel_int64(uint64_t N, int64_t* in, uint64_t size)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	int64_t howmany = N/size;
	if(i<size)
	{
		for(int64_t j=1;j<howmany;j++)
		{
			in[i] += in[i+j*size];
		}
	}
}

template<class DataType> 
void reduction_general(uint64_t N, DataType* in, uint64_t size){}

template<>
void reduction_general<float>(uint64_t N, float* in, uint64_t size)
{
	int power = std::log2(N/size);
	uint64_t n = size*1<<power;
	if(n < N)
	{
		int64_t diff = N-n+size;
		reduction_general_kernel_float<<<diff/512+1,512>>>(diff,in+n-size,size);
	}
	reduction<float>(n,in,size);
}

template<>
void reduction_general<double>(uint64_t N, double* in, uint64_t size)
{
	int power = std::log2(N/size);
	uint64_t n = size*1<<power;
	if(n < N)
	{
		int64_t diff = N-n+size;
		reduction_general_kernel_double<<<diff/512+1,512>>>(diff,in+n-size,size);
	}
	reduction<double>(n,in,size);
}

template<>
void reduction_general<uint8_t>(uint64_t N, uint8_t* in, uint64_t size)
{
	int power = std::log2(N/size);
	uint64_t n = size*1<<power;
	if(n < N)
	{
		int64_t diff = N-n+size;
		reduction_general_kernel_uint8<<<diff/512+1,512>>>(diff,in+n-size,size);
	}
	reduction<uint8_t>(n,in,size);
}

template<>
void reduction_general<uint16_t>(uint64_t N, uint16_t* in, uint64_t size)
{
	int power = std::log2(N/size);
	uint64_t n = size*1<<power;
	if(n < N)
	{
		int64_t diff = N-n+size;
		reduction_general_kernel_uint16<<<diff/512+1,512>>>(diff,in+n-size,size);
	}
	reduction<uint16_t>(n,in,size);
}

template<>
void reduction_general<uint32_t>(uint64_t N, uint32_t* in, uint64_t size)
{
	int power = std::log2(N/size);
	uint64_t n = size*1<<power;
	if(n < N)
	{
		int64_t diff = N-n+size;
		reduction_general_kernel_uint32<<<diff/512+1,512>>>(diff,in+n-size,size);
	}
	reduction<uint32_t>(n,in,size);
}

template<>
void reduction_general<uint64_t>(uint64_t N, uint64_t* in, uint64_t size)
{
	int power = std::log2(N/size);
	uint64_t n = size*1<<power;
	if(n < N)
	{
		int64_t diff = N-n+size;
		reduction_general_kernel_uint64<<<diff/512+1,512>>>(diff,in+n-size,size);
	}
	reduction<uint64_t>(n,in,size);
}

template<>
void reduction_general<int8_t>(uint64_t N, int8_t* in, uint64_t size)
{
	int power = std::log2(N/size);
	uint64_t n = size*1<<power;
	if(n < N)
	{
		int64_t diff = N-n+size;
		reduction_general_kernel_int8<<<diff/512+1,512>>>(diff,in+n-size,size);
	}
	reduction<int8_t>(n,in,size);
}

template<>
void reduction_general<int16_t>(uint64_t N, int16_t* in, uint64_t size)
{
	int power = std::log2(N/size);
	uint64_t n = size*1<<power;
	if(n < N)
	{
		int64_t diff = N-n+size;
		reduction_general_kernel_int16<<<diff/512+1,512>>>(diff,in+n-size,size);
	}
	reduction<int16_t>(n,in,size);
}

template<>
void reduction_general<int32_t>(uint64_t N, int32_t* in, uint64_t size)
{
	int power = std::log2(N/size);
	uint64_t n = size*1<<power;
	if(n < N)
	{
		int64_t diff = N-n+size;
		reduction_general_kernel_int32<<<diff/512+1,512>>>(diff,in+n-size,size);
	}
	reduction<int32_t>(n,in,size);
}

template<>
void reduction_general<int64_t>(uint64_t N, int64_t* in, uint64_t size)
{
	int power = std::log2(N/size);
	uint64_t n = size*1<<power;
	if(n < N)
	{
		int64_t diff = N-n+size;
		reduction_general_kernel_int64<<<diff/512+1,512>>>(diff,in+n-size,size);
	}
	reduction<int64_t>(n,in,size);
}

////////////////////////////////////////////////////////////////
//  _     _   _   _ _     _                                   //
// / | __| | | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___   //
// | |/ _` | | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \  //
// | | (_| | |  _  | \__ \ || (_) | (_| | | | (_| | | | | | | //
// |_|\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_| //
//                                 |___/                      //
////////////////////////////////////////////////////////////////

__global__ void digitizer_histogram_1d_uint8(uint64_t N, uint8_t* in, uint32_t* hist)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N){atomicAdd(&hist[in[i]],1);}
}

__global__ void digitizer_histogram_1d_uint16(uint64_t N, uint16_t* in, uint32_t* hist)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N){atomicAdd(&hist[in[i]],1);}
}

__global__ void digitizer_histogram_1d_uint32(uint64_t N, uint32_t* in, uint32_t* hist)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N){atomicAdd(&hist[in[i]],1);}
}

__global__ void digitizer_histogram_1d_uint64(uint64_t N, uint64_t* in, uint32_t* hist)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N){atomicAdd(&hist[in[i]],1);}
}

__global__ void digitizer_histogram_1d_int8(uint64_t N, int8_t* in, uint32_t* hist)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N){atomicAdd(&hist[in[i]-INT8_MIN],1);}
}

__global__ void digitizer_histogram_1d_int16(uint64_t N, int16_t* in, uint32_t* hist)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N){atomicAdd(&hist[in[i]-INT16_MIN],1);}
}

__global__ void digitizer_histogram_1d_int32(uint64_t N, int32_t* in, uint32_t* hist)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N){atomicAdd(&hist[in[i]-INT32_MIN],1);}
}

__global__ void digitizer_histogram_1d_int64(uint64_t N, int64_t* in, uint32_t* hist)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N){atomicAdd(&hist[in[i]-INT64_MIN],1);}
}

template<class DataType>
void digitizer_histogram_1d(uint64_t N, DataType* in, uint32_t* hist, cudaStream_t stream){}

template<>
void digitizer_histogram_1d<uint8_t>(uint64_t N, uint8_t* in, uint32_t* hist, cudaStream_t stream)
{
	digitizer_histogram_1d_uint8<<<(N/512+1),512,0,stream>>>(N,in,hist);
}

template<>
void digitizer_histogram_1d<uint16_t>(uint64_t N, uint16_t* in, uint32_t* hist, cudaStream_t stream)
{
	digitizer_histogram_1d_uint16<<<(N/512+1),512,0,stream>>>(N,in,hist);
}

template<>
void digitizer_histogram_1d<uint32_t>(uint64_t N, uint32_t* in, uint32_t* hist, cudaStream_t stream)
{
	digitizer_histogram_1d_uint32<<<(N/512+1),512,0,stream>>>(N,in,hist);
}

template<>
void digitizer_histogram_1d<uint64_t>(uint64_t N, uint64_t* in, uint32_t* hist, cudaStream_t stream)
{
	digitizer_histogram_1d_uint64<<<(N/512+1),512,0,stream>>>(N,in,hist);
}

template<>
void digitizer_histogram_1d<int8_t>(uint64_t N, int8_t* in, uint32_t* hist, cudaStream_t stream)
{
	digitizer_histogram_1d_int8<<<(N/512+1),512,0,stream>>>(N,in,hist);
}

template<>
void digitizer_histogram_1d<int16_t>(uint64_t N, int16_t* in, uint32_t* hist, cudaStream_t stream)
{
	digitizer_histogram_1d_int16<<<(N/512+1),512,0,stream>>>(N,in,hist);
}

template<>
void digitizer_histogram_1d<int32_t>(uint64_t N, int32_t* in, uint32_t* hist, cudaStream_t stream)
{
	digitizer_histogram_1d_int32<<<(N/512+1),512,0,stream>>>(N,in,hist);
}

template<>
void digitizer_histogram_1d<int64_t>(uint64_t N, int64_t* in, uint32_t* hist, cudaStream_t stream)
{
	digitizer_histogram_1d_int64<<<(N/512+1),512,0,stream>>>(N,in,hist);
}

///////////////////////////////////////////////////////////////////
//    _     _   _   _ _     _                                    //
//   / | __| | | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___    //
//   | |/ _` | | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \   //
//   | | (_| | |  _  | \__ \ || (_) | (_| | | | (_| | | | | | |  //
//   |_|\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_|  //
//	                                 |___/                       //
//                         _     _           _                   //
//               ___ _   _| |__ | |__  _   _| |_ ___             //
//              / __| | | | '_ \| '_ \| | | | __/ _ \            //
//              \__ \ |_| | |_) | |_) | |_| | ||  __/            //
//              |___/\__,_|_.__/|_.__/ \__, |\__\___|            //
//                                     |___/                     //
///////////////////////////////////////////////////////////////////

__global__ void digitizer_histogram_subbyte_1d_uint8(uint64_t N, uint8_t* in, 
				uint32_t* hist, uint8_t shift)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N){atomicAdd(&hist[in[i]>>shift],1);}
}

template<class DataType>
void digitizer_histogram_subbyte_1d(uint64_t N, DataType* in, uint32_t* hist, 
				uint8_t nbits, cudaStream_t stream){}

template<>
void digitizer_histogram_subbyte_1d<uint8_t>(uint64_t N, uint8_t* in, uint32_t* hist, 
				uint8_t nbits, cudaStream_t stream)
{
	digitizer_histogram_subbyte_1d_uint8<<<(N/512+1),512,0,stream>>>(N,in,hist,8-nbits);
}

///////////////////////////////////////////////////////////////////
//    _     _   _   _ _     _                                    //
//   / | __| | | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___    //
//   | |/ _` | | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \   //
//   | | (_| | |  _  | \__ \ || (_) | (_| | | | (_| | | | | | |  //
//   |_|\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_|  //
//	                                 |___/                       //
//                             _                                 //
//	                       ___| |_ ___ _ __                      //
//	                      / __| __/ _ \ '_ \                     //
//	                      \__ \ ||  __/ |_) |                    //
//	                      |___/\__\___| .__/                     //
//	                                  |_|                        //
///////////////////////////////////////////////////////////////////

__global__ void digitizer_histogram_step_1d_uint8(uint64_t N, uint8_t* in, 
				uint32_t* hist, uint8_t shift)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<(N-1))
	{
		atomicAdd(&hist[in[i]>>shift],1);
		atomicAdd(&(hist+(1<<(8-shift)))[(in[i]>>shift)<<(8-shift) ^ (in[i+1]>>shift)],1);
	}
}

template<class DataType>
void digitizer_histogram_step_1d(uint64_t N, DataType* in, uint32_t* hist, 
				uint8_t nbits, cudaStream_t stream){}

template<>
void digitizer_histogram_step_1d<uint8_t>(uint64_t N, uint8_t* in, uint32_t* hist, 
				uint8_t nbits, cudaStream_t stream)
{
	digitizer_histogram_step_1d_uint8<<<(N/512+1),512,0,stream>>>(N,in,hist,8-nbits);
}

///////////////////////////////////////////////////////////////////
//  ____     _   _   _ _     _                                   //
// |___ \ __| | | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___   //
//   __) / _` | | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \  //
//  / __/ (_| | |  _  | \__ \ || (_) | (_| | | | (_| | | | | | | //
// |_____\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_| //
//					                  |___/                      //
///////////////////////////////////////////////////////////////////

__global__ void digitizer_histogram_2d_uint8(uint64_t N, uint8_t* in_x, uint8_t* in_y, 
				uint32_t* hist)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N){atomicAdd(&hist[in_y[i] ^ in_x[i]<<8],1);}
}

__global__ void digitizer_histogram_2d_uint16(uint64_t N, uint16_t* in_x, uint16_t* in_y, 
				uint32_t* hist)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N){atomicAdd(&hist[in_y[i] ^ in_x[i]<<16],1);}
}

__global__ void digitizer_histogram_2d_int8(uint64_t N, int8_t* in_x, int8_t* in_y, 
				uint32_t* hist)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N){atomicAdd(&hist[(in_y[i]-INT8_MIN) ^ ((in_x[i]-INT8_MIN)<<8)],1);}
}

__global__ void digitizer_histogram_2d_int16(uint64_t N, int16_t* in_x, int16_t* in_y, 
				uint32_t* hist)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N){atomicAdd(&hist[(in_y[i]-INT16_MIN) ^ ((in_x[i]-INT16_MIN)<<16)],1);}
}

template<class DataType>
void digitizer_histogram_2d(uint64_t N, DataType* in_x, DataType* in_y, uint32_t* hist, 
				cudaStream_t stream){}

template<>
void digitizer_histogram_2d<uint8_t>(uint64_t N, uint8_t* in_x, uint8_t* in_y, uint32_t* hist, 
				cudaStream_t stream)
{
	digitizer_histogram_2d_uint8<<<(N/512+1),512,0,stream>>>(N,in_x,in_y,hist);
}

template<>
void digitizer_histogram_2d<uint16_t>(uint64_t N, uint16_t* in_x, uint16_t* in_y, uint32_t* hist, 
				cudaStream_t stream)
{
	digitizer_histogram_2d_uint16<<<(N/512+1),512,0,stream>>>(N,in_x,in_y,hist);
}

template<>
void digitizer_histogram_2d<int8_t>(uint64_t N, int8_t* in_x, int8_t* in_y, uint32_t* hist, 
				cudaStream_t stream)
{
	digitizer_histogram_2d_int8<<<(N/512+1),512,0,stream>>>(N,in_x,in_y,hist);
}

template<>
void digitizer_histogram_2d<int16_t>(uint64_t N, int16_t* in_x, int16_t* in_y, uint32_t* hist, cudaStream_t stream)
{
	digitizer_histogram_2d_int16<<<(N/512+1),512,0,stream>>>(N,in_x,in_y,hist);
}

///////////////////////////////////////////////////////////////////
//  ____     _   _   _ _     _                                   //
// |___ \ __| | | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___   //
//   __) / _` | | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \  //
//  / __/ (_| | |  _  | \__ \ || (_) | (_| | | | (_| | | | | | | //
// |_____\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_| //
//					                  |___/                      //
//                         _     _           _                   // 
//               ___ _   _| |__ | |__  _   _| |_ ___             //
//              / __| | | | '_ \| '_ \| | | | __/ _ \            //
//              \__ \ |_| | |_) | |_) | |_| | ||  __/            //
//              |___/\__,_|_.__/|_.__/ \__, |\__\___|            //
//                                     |___/                     //
///////////////////////////////////////////////////////////////////

__global__ void digitizer_histogram_subbyte_2d_uint8(uint64_t N, uint8_t* in_x, uint8_t* in_y, 
				uint32_t* hist, uint8_t shift)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N){atomicAdd(&hist[(in_y[i]>>shift) ^ ((in_x[i]>>shift)<<(8-shift))],1);}
}

template<class DataType>
void digitizer_histogram_subbyte_2d(uint64_t N, DataType* in_x, DataType* in_y, uint32_t* hist, 
				uint8_t nbits, cudaStream_t stream){}

template<>
void digitizer_histogram_subbyte_2d<uint8_t>(uint64_t N, uint8_t* in_x, uint8_t* in_y, 
				uint32_t* hist, uint8_t nbits, cudaStream_t stream)
{
	digitizer_histogram_subbyte_2d_uint8<<<(N/512+1),512,0,stream>>>(N,in_x,in_y,hist,8-nbits);
}

///////////////////////////////////////////////////////////////////
//  ____     _   _   _ _     _                                   //
// |___ \ __| | | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___   //
//   __) / _` | | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \  //
//  / __/ (_| | |  _  | \__ \ || (_) | (_| | | | (_| | | | | | | //
// |_____\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_| //
//					                  |___/                      //
//                             _                                 //
//	                       ___| |_ ___ _ __                      //
//	                      / __| __/ _ \ '_ \                     //
//	                      \__ \ ||  __/ |_) |                    //
//	                      |___/\__\___| .__/                     //
//	                                  |_|                        //
///////////////////////////////////////////////////////////////////

__global__ void digitizer_histogram_step_2d_uint8(uint64_t N, uint8_t* in_x, uint8_t* in_y, 
				uint32_t* hist, uint8_t shift, uint8_t nbits)
{
	uint64_t i = threadIdx.x+blockIdx.x*blockDim.x;
	uint64_t bin_x, bin_y, bin_x2, bin_y2;
	if(i<(N-1))
	{
		bin_x = in_x[i]>>shift;
		bin_y = in_y[i]>>shift;
		bin_x2 = in_x[i+1]>>shift;
		bin_y2 = in_y[i+1]>>shift;
		atomicAdd(&hist[(bin_x<<nbits) ^ bin_y],1);
		atomicAdd(
		&(hist+(1<<nbits<<nbits))
		[(bin_x<<nbits<<nbits<<nbits) ^ (bin_y<<nbits<<nbits) ^ (bin_x2<<nbits) ^ (bin_y2)]
		,1);
	}
}

template<class DataType>
void digitizer_histogram_step_2d(uint64_t N, DataType* in_x, DataType* in_y, uint32_t* hist, 
				uint8_t nbits, cudaStream_t stream){}

template<>
void digitizer_histogram_step_2d<uint8_t>(uint64_t N, uint8_t* in_x, uint8_t* in_y, 
				uint32_t* hist, 
				uint8_t nbits, cudaStream_t stream)
{
	digitizer_histogram_step_2d_uint8<<<(N/512+1),512,0,stream>>>(N,in_x,in_y,hist,8-nbits,nbits);
}
