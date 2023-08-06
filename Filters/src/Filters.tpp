#include <cstring>
void manage_thread_affinity()
{
	//Shamelessly stolen from Jean-Olivier's code at
	//https://github.com/JeanOlivier/Histograms-OTF/blob/master/histograms.c
    #ifdef _WIN32_WINNT
        int nbgroups = GetActiveProcessorGroupCount();
        int *threads_per_groups = (int *) malloc(nbgroups*sizeof(int));
        for (int i=0; i<nbgroups; i++)
        {
            threads_per_groups[i] = GetActiveProcessorCount(i);
        }

        // Fetching thread number and assigning it to cores
        int tid = omp_get_thread_num(); // Internal omp thread number (0 -- OMP_NUM_THREADS)
        HANDLE thandle = GetCurrentThread();
        bool result;

        // We change group for each thread
        short unsigned int set_group = tid%nbgroups; 
		
		// Nb of threads in group for affinity mask.
        int nbthreads = threads_per_groups[set_group]; 
        
		// nbcores amount of 1 in binary
        GROUP_AFFINITY group = {((uint64_t)1<<nbthreads)-1, set_group};
		
		// Actually setting the affinity
        result = SetThreadGroupAffinity(thandle, &group, NULL); 
        if(!result) std::fprintf(stderr, "Failed setting output for tid=%i\n", tid);
        free(threads_per_groups);
    #else
        //We let openmp and the OS manage the threads themselves
    #endif
}

template <class DataType>
DataType getThreads()
{
	DataType threads;
    #ifdef _WIN32_WINNT
		DataType nbgroups = GetActiveProcessorGroupCount();
	    threads = std::min((DataType) 64,(DataType) (omp_get_max_threads()*nbgroups));
	#else
    #ifdef _WIN32_WINNT
		uint64_t nbgroups = GetActiveProcessorGroupCount();
	    threads = std::min((uint64_t) 64,omp_get_max_threads()*nbgroups);
	#else
		threads = omp_get_max_threads();
	#endif
		threads = (DataType) omp_get_max_threads();
	#endif
	return threads;
}

template<class DataTypeIn, class DataTypeOut>
DataTypeOut dotProductAVX(uint64_t N, DataTypeIn* in1, DataTypeIn* in2){}

template<>
double dotProductAVX<double,double>(uint64_t N, double* in1, double* in2)
{
	__m256d ymm0,ymm1;
	double out = 0;
	double* res = (double*)&ymm0;
	uint64_t N2 = N/4;
	uint64_t k = 0;
	for(uint64_t j=0;j<N2;j++)
	{
		k = 4*j;
		ymm0 = _mm256_loadu_pd(in1+k);
		ymm1 = _mm256_loadu_pd(in2+k);
		ymm0 = _mm256_mul_pd(ymm0,ymm1);
		ymm0 = _mm256_add_pd(ymm0,_mm256_permute2f128_pd(ymm0,ymm0,1));
		ymm0 = _mm256_hadd_pd(ymm0,ymm0);
		out += res[0];
	}
	out += std::inner_product(in1+4*N2,in1+N,in2,0.0);
	return out;
}

template<>
float dotProductAVX<float,float>(uint64_t N, float* in1, float* in2)
{
	__m256 ymm0,ymm1;
	float out = 0;
	float* res = (float*)&ymm0;
	uint64_t N2 = N/8;
	uint64_t k = 0;
	for(uint64_t j=0;j<N2;j++)
	{
		k = 8*j;
		ymm0 = _mm256_loadu_ps(in1+k);
		ymm1 = _mm256_loadu_ps(in2+k);
		ymm0 = _mm256_mul_ps(ymm0,ymm1);
		ymm0 = _mm256_add_ps(ymm0,_mm256_permute2f128_ps(ymm0,ymm0,1));
		ymm0 = _mm256_hadd_ps(ymm0,ymm0);
		ymm0 = _mm256_hadd_ps(ymm0,ymm0);
		out += res[0];
	}
	out += std::inner_product(in1+8*N2,in1+N,in2,0.0);
	return out;
}

template<class DataTypeIn, class DataTypeOut>
void filterEdgeLeftAVX(uint64_t N, DataTypeIn* in1, DataTypeIn* in2, DataTypeOut* out){}

template<>
void filterEdgeLeftAVX<double,double>(uint64_t N, double* in1, double* in2, double* out)
{
	__m256d ymm0,ymm1,ymm2;
	uint64_t N2,k;
	double f;
	for(uint64_t j=0;j<(N-1);j++)
	{
		f = in2[N-1-j];
		N2 = (N-1-j)/4;
		ymm0 = _mm256_broadcast_sd(&f);
		#pragma omp parallel for
		for(uint64_t i=0;i<N2;i++)
		{
			k = 4*i;
			ymm1 = _mm256_loadu_pd(in1+k);
			ymm2 = _mm256_loadu_pd(out+j+k);
			ymm2 = _mm256_add_pd(ymm2,_mm256_mul_pd(ymm0,ymm1));
			_mm256_storeu_pd(out+j+k,ymm2);
		}
		for(uint64_t i=4*N2;i<(N-1-j);i++)
		{
			out[j+i] += f*in1[i];
		}
	}
}
/*
template<>
void filterEdgeLeftAVX<double,double>(uint64_t N, double* in1, double* in2, double* out)
{
	__m256d ymm0,ymm1,ymm2;
	uint64_t N2,k;
	double* res = (double*)&ymm2;
	N2 = (N-1)/4;
	for(uint64_t j=0;j<N2;j++)
	{
		k = 4*j;
		ymm2 = _mm256_setzero_pd();
		for(uint64_t i=0;i<(k+4);i++)
		{
			ymm0 = _mm256_loadu_pd(in1); 
			ymm1 = _mm256_broadcast_sd(in2+N-1-i);
			ymm2 = _mm256_add_pd(ymm2,_mm256_mul_pd(ymm0,ymm1));
			out[i+k] = res[0];
			res[0] = 0.0;
			ymm2 = _mm256_permute4x64_pd(ymm2,0b00111001);
		}
	}
	for(uint64_t j=(4*N2);j<(N-1);j++)
	{
		out[j] = std::inner_product(in2+N-j,in2+N,in1+j,0.0);
	}
}*/

template<>
void filterEdgeLeftAVX<float,float>(uint64_t N, float* in1, float* in2, float* out)
{
	__m256 ymm0,ymm1,ymm2;
	uint64_t N2,k;
	float f;
	for(uint64_t j=0;j<(N-1);j++)
	{
		f = in2[N-1-j];
		N2 = (N-1-j)/8;
		ymm0 = _mm256_broadcast_ss(&f);
		#pragma omp parallel for
		for(uint64_t i=0;i<N2;i++)
		{
			k = 8*i;
			ymm1 = _mm256_loadu_ps(in1+k);
			ymm2 = _mm256_loadu_ps(out+j+k);
			ymm2 = _mm256_add_ps(ymm2,_mm256_mul_ps(ymm0,ymm1));
			_mm256_storeu_ps(out+j+k,ymm2);
		}
		for(uint64_t i=8*N2;i<(N-1-j);i++)
		{
			out[j+i] += f*in1[i];
		}
	}
}

template<class DataTypeIn, class DataTypeOut>
void filterEdgeRightAVX(uint64_t N, DataTypeIn* in1, DataTypeIn* in2, DataTypeOut* out){}

template<>
void filterEdgeRightAVX<double,double>(uint64_t N, double* in1, double* in2, double* out)
{
	__m256d ymm0,ymm1,ymm2;
	uint64_t N2,k;
	double f;
	for(uint64_t j=0;j<N;j++)
	{
		f = in2[N-1-j];
		ymm0 = _mm256_broadcast_sd(&f);
		N2 = (N-j)/4;
		#pragma omp parallel for
		for(uint64_t i=0;i<N2;i++)
		{
			k = 4*i;
			
			ymm1 = _mm256_loadu_pd(in1-k-3);
			ymm2 = _mm256_loadu_pd(out-j-k-3);
			ymm2 = _mm256_add_pd(ymm2,_mm256_mul_pd(ymm0,ymm1));
			_mm256_storeu_pd(out-j-k-3,ymm2);	
		}
		for(uint64_t i=4*N2;i<(N-j);i++)
		{
			(out-j-i)[0] += f*(in1-i)[0];
		}
	}
}

template<>
void filterEdgeRightAVX<float,float>(uint64_t N, float* in1, float* in2, float* out)
{
	__m256 ymm0,ymm1,ymm2;
	uint64_t N2,k;
	float f;
	for(uint64_t j=0;j<N;j++)
	{
		f = in2[N-1-j];
		ymm0 = _mm256_broadcast_ss(&f);
		N2 = (N-j)/8;
		#pragma omp parallel for
		for(uint64_t i=0;i<N2;i++)
		{
			k = 8*i;
			ymm1 = _mm256_loadu_ps(out-j-k-7);
			ymm2 = _mm256_loadu_ps(in1-k-7);
			ymm1 = _mm256_add_ps(ymm1,_mm256_mul_ps(ymm0,ymm2));
			_mm256_storeu_ps(out-j-k-7,ymm1);
		}
		for(uint64_t i=8*N2;i<(N-j);i++)
		{
			(out-j-i)[0] += f*(in1-i)[0];
		}
	}
}

template<class DataTypeIn, class DataTypeOut>
void filterAVX(uint64_t N, uint64_t Nfilter, DataTypeIn* in1, DataTypeIn* in2, DataTypeOut* out){}

template<>
void filterAVX<double,double>(uint64_t N, uint64_t Nfilter, double* data, double* filter, double* out)
{
	__m256d ymm0,ymm1,ymm2,ymm3,ymm4,ymm5,ymm6,ymm7,ymm8,ymm9,ymm10,ymm11,ymm12,ymm13,ymm14,ymm15;
	double *res1,*res2,*res3,*res4,*res5,*res6,*res7,*res8,*res9,*res10,*res11,*res12,*res13,*res14;
	uint64_t N2 = N/56;
	uint64_t k = 0;
	for(uint64_t j=0;j<N2;j++)
	{
		k = 56*j;
		res1 = out+k; 		// ymm2
		res2 = out+k+4;		// ymm3
		res3 = out+k+8;		// ymm4
		res4 = out+k+12;	// ymm5
		res5 = out+k+16;	// ymm6
		res6 = out+k+20;	// ymm7
		res7 = out+k+24;	// ymm8
		res8 = out+k+28;	// ymm9
		res9 = out+k+32;	// ymm10
		res10 = out+k+36;	// ymm11
		res11 = out+k+40;	// ymm12
		res12 = out+k+44;	// ymm13
		res13 = out+k+48;	// ymm14
		res14 = out+k+52;	// ymm15
		ymm2 = _mm256_setzero_pd();
		ymm3 = _mm256_setzero_pd();
		ymm4 = _mm256_setzero_pd();
		ymm5 = _mm256_setzero_pd();
		ymm6 = _mm256_setzero_pd();
		ymm7 = _mm256_setzero_pd();
		ymm8 = _mm256_setzero_pd();
		ymm9 = _mm256_setzero_pd();
		ymm10 = _mm256_setzero_pd();
		ymm11 = _mm256_setzero_pd();
		ymm12 = _mm256_setzero_pd();
		ymm13 = _mm256_setzero_pd();
		ymm14 = _mm256_setzero_pd();
		ymm15 = _mm256_setzero_pd();
		for(uint64_t i=0;i<Nfilter;i++)
		{
			ymm0 = _mm256_loadu_pd(data+k+i); 
			ymm1 = _mm256_broadcast_sd(filter+i);
			ymm2 = _mm256_add_pd(ymm2,_mm256_mul_pd(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_pd(data+k+i+4); 
			ymm3 = _mm256_add_pd(ymm3,_mm256_mul_pd(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_pd(data+k+i+8); 
			ymm4 = _mm256_add_pd(ymm4,_mm256_mul_pd(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_pd(data+k+i+12); 
			ymm5 = _mm256_add_pd(ymm5,_mm256_mul_pd(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_pd(data+k+i+16); 
			ymm6 = _mm256_add_pd(ymm6,_mm256_mul_pd(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_pd(data+k+i+20); 
			ymm7 = _mm256_add_pd(ymm7,_mm256_mul_pd(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_pd(data+k+i+24); 
			ymm8 = _mm256_add_pd(ymm8,_mm256_mul_pd(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_pd(data+k+i+28); 
			ymm9 = _mm256_add_pd(ymm9,_mm256_mul_pd(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_pd(data+k+i+32); 
			ymm10 = _mm256_add_pd(ymm10,_mm256_mul_pd(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_pd(data+k+i+36); 
			ymm11 = _mm256_add_pd(ymm11,_mm256_mul_pd(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_pd(data+k+i+40); 
			ymm12 = _mm256_add_pd(ymm12,_mm256_mul_pd(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_pd(data+k+i+44); 
			ymm13 = _mm256_add_pd(ymm13,_mm256_mul_pd(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_pd(data+k+i+48); 
			ymm14 = _mm256_add_pd(ymm14,_mm256_mul_pd(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_pd(data+k+i+52); 
			ymm15 = _mm256_add_pd(ymm15,_mm256_mul_pd(ymm0,ymm1));
		}
		_mm256_storeu_pd(res1,ymm2);
		_mm256_storeu_pd(res2,ymm3);
		_mm256_storeu_pd(res3,ymm4);
		_mm256_storeu_pd(res4,ymm5);
		_mm256_storeu_pd(res5,ymm6);
		_mm256_storeu_pd(res6,ymm7);
		_mm256_storeu_pd(res7,ymm8);
		_mm256_storeu_pd(res8,ymm9);
		_mm256_storeu_pd(res9,ymm10);
		_mm256_storeu_pd(res10,ymm11);
		_mm256_storeu_pd(res11,ymm12);
		_mm256_storeu_pd(res12,ymm13);
		_mm256_storeu_pd(res13,ymm14);
		_mm256_storeu_pd(res14,ymm15);
	}
	for(uint64_t j=(56*N2);j<N;j++)
	{
		out[j] = std::inner_product(filter,filter+Nfilter,data+j,0.0);
	}
}

template<>
void filterAVX<float,float>(uint64_t N, uint64_t Nfilter, float* data, float* filter, float* out)
{
	__m256 ymm0,ymm1,ymm2,ymm3,ymm4,ymm5,ymm6,ymm7,ymm8,ymm9,ymm10,ymm11,ymm12,ymm13,ymm14,ymm15;
	float *res1,*res2,*res3,*res4,*res5,*res6,*res7,*res8,*res9,*res10,*res11,*res12,*res13,*res14;
	uint64_t N2 = N/112;
	uint64_t k = 0;
	for(uint64_t j=0;j<N2;j++)
	{
		k = 112*j;
		res1 = out+k; 		// ymm2
		res2 = out+k+8;		// ymm3
		res3 = out+k+16;	// ymm4
		res4 = out+k+24;	// ymm5
		res5 = out+k+32;	// ymm6
		res6 = out+k+40;	// ymm7
		res7 = out+k+48;	// ymm8
		res8 = out+k+56;	// ymm9
		res9 = out+k+64;	// ymm10
		res10 = out+k+72;	// ymm11
		res11 = out+k+80;	// ymm12
		res12 = out+k+88;	// ymm13
		res13 = out+k+96;	// ymm14
		res14 = out+k+104;	// ymm15
		ymm2 = _mm256_setzero_ps();
		ymm3 = _mm256_setzero_ps();
		ymm4 = _mm256_setzero_ps();
		ymm5 = _mm256_setzero_ps();
		ymm6 = _mm256_setzero_ps();
		ymm7 = _mm256_setzero_ps();
		ymm8 = _mm256_setzero_ps();
		ymm9 = _mm256_setzero_ps();
		ymm10 = _mm256_setzero_ps();
		ymm11 = _mm256_setzero_ps();
		ymm12 = _mm256_setzero_ps();
		ymm13 = _mm256_setzero_ps();
		ymm14 = _mm256_setzero_ps();
		ymm15 = _mm256_setzero_ps();
		for(uint64_t i=0;i<Nfilter;i++)
		{
			ymm0 = _mm256_loadu_ps(data+k+i); 
			ymm1 = _mm256_broadcast_ss(filter+i);
			ymm2 = _mm256_add_ps(ymm2,_mm256_mul_ps(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_ps(data+k+i+8); 
			ymm3 = _mm256_add_ps(ymm3,_mm256_mul_ps(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_ps(data+k+i+16); 
			ymm4 = _mm256_add_ps(ymm4,_mm256_mul_ps(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_ps(data+k+i+24); 
			ymm5 = _mm256_add_ps(ymm5,_mm256_mul_ps(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_ps(data+k+i+32); 
			ymm6 = _mm256_add_ps(ymm6,_mm256_mul_ps(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_ps(data+k+i+40); 
			ymm7 = _mm256_add_ps(ymm7,_mm256_mul_ps(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_ps(data+k+i+48); 
			ymm8 = _mm256_add_ps(ymm8,_mm256_mul_ps(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_ps(data+k+i+56); 
			ymm9 = _mm256_add_ps(ymm9,_mm256_mul_ps(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_ps(data+k+i+64); 
			ymm10 = _mm256_add_ps(ymm10,_mm256_mul_ps(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_ps(data+k+i+72); 
			ymm11 = _mm256_add_ps(ymm11,_mm256_mul_ps(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_ps(data+k+i+80); 
			ymm12 = _mm256_add_ps(ymm12,_mm256_mul_ps(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_ps(data+k+i+88); 
			ymm13 = _mm256_add_ps(ymm13,_mm256_mul_ps(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_ps(data+k+i+96); 
			ymm14 = _mm256_add_ps(ymm14,_mm256_mul_ps(ymm0,ymm1));
			
			ymm0 = _mm256_loadu_ps(data+k+i+104); 
			ymm15 = _mm256_add_ps(ymm15,_mm256_mul_ps(ymm0,ymm1));
		}
		_mm256_storeu_ps(res1,ymm2);
		_mm256_storeu_ps(res2,ymm3);
		_mm256_storeu_ps(res3,ymm4);
		_mm256_storeu_ps(res4,ymm5);
		_mm256_storeu_ps(res5,ymm6);
		_mm256_storeu_ps(res6,ymm7);
		_mm256_storeu_ps(res7,ymm8);
		_mm256_storeu_ps(res8,ymm9);
		_mm256_storeu_ps(res9,ymm10);
		_mm256_storeu_ps(res10,ymm11);
		_mm256_storeu_ps(res11,ymm12);
		_mm256_storeu_ps(res12,ymm13);
		_mm256_storeu_ps(res13,ymm14);
		_mm256_storeu_ps(res14,ymm15);
	}
	for(uint64_t j=(112*N2);j<N;j++)
	{
		out[j] = std::inner_product(filter,filter+Nfilter,data+j,0.0);
	}
}

template<class DataType>
void generateBoxcar(uint64_t order, DataType* out)
{
	for(uint64_t i=0;i<order;i++){out[i] = 1;}
}

template<>
void generateBoxcar<float>(uint64_t order, float* out)
{
	for(uint64_t i=0;i<order;i++){out[i] = 1.0/order;}
}

template<>
void generateBoxcar<double>(uint64_t order, double* out)
{
	for(uint64_t i=0;i<order;i++){out[i] = 1.0/order;}
}

template<class DataTypeIn, class DataTypeOut>
void applyFilter(uint64_t Ndata, uint64_t Nfilter, DataTypeIn* data, 
				DataTypeOut* out, DataTypeIn* filter)
{
	uint64_t threads = getThreads<uint64_t>();
	uint64_t shift = Nfilter/2;
	uint64_t Nthreads = (Ndata-2*shift)/threads;
	DataTypeIn zero = (DataTypeIn) 0;
	
	if(Nthreads < 1<<14){threads = 1; Nthreads = Ndata-2*shift;}
	
	#pragma omp parallel for num_threads(threads)
	for(uint64_t i=0;i<threads;i++)
	{
		manage_thread_affinity();
		for(uint64_t j=(i*Nthreads+shift);j<((i+1)*Nthreads+shift);j++)
		{
			out[j] = std::inner_product(filter,filter+Nfilter,data+j-shift,zero);
		}
	}
	
	for(uint64_t i=0;i<shift;i++)
	{
		out[i] = std::inner_product(filter+(shift-i),filter+Nfilter,data,zero);
	}
	for(uint64_t i=(shift+threads*Nthreads);i<Ndata;i++)
	{
		out[i] = std::inner_product(filter,filter+Ndata-i+shift,data+i-shift,zero);
	}
}

template<class DataTypeIn, class DataTypeOut>
void applyFilterAVX(uint64_t Ndata, uint64_t Nfilter, DataTypeIn* data, 
				DataTypeOut* out, DataTypeIn* filter)
{
	uint64_t threads,Nthreads;
	if(Ndata < 1<<16){threads = 1;Nthreads = Ndata-Nfilter;}
	else{threads = getThreads<uint64_t>();Nthreads = (Ndata-Nfilter)/threads;}

	uint64_t* Nthreads_arr = (uint64_t*) malloc(threads*sizeof(uint64_t));
	for(uint64_t i=0;i<threads;i++){Nthreads_arr[i]=Nthreads;}
	Nthreads_arr[threads-1] += Ndata-Nfilter-threads*Nthreads;

	#pragma omp parallel for num_threads(threads)
	for(uint64_t i=0;i<threads;i++)
	{
		manage_thread_affinity();
		DataTypeIn* sData = data+i*Nthreads;
		DataTypeOut* sOut = out+i*Nthreads+Nfilter-1;
		filterAVX<DataTypeIn,DataTypeOut>(Nthreads_arr[i],Nfilter,sData,filter,sOut);
	}
	
	filterEdgeLeftAVX<DataTypeIn,DataTypeOut>(Nfilter,data,filter,out);
	filterEdgeRightAVX<DataTypeIn,DataTypeOut>(Nfilter,data+Ndata-1,filter,out+Ndata+Nfilter-2);
	
	free(Nthreads_arr);	
}
