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
void filterEdgeLeftAVX(uint64_t N, DataTypeIn* in1, DataTypeOut* in2, DataTypeOut* out){}

template<>
void filterEdgeLeftAVX<double,double>(uint64_t N, double* in1, double* in2, double* out)
{
	uint64_t N2 = (N-1)/4;
	#pragma omp parallel for schedule(dynamic,1)
	for(uint64_t j=0;j<N2;j++)
	{
		__m256d ymm0,ymm1,ymm2;
		uint64_t k;
		double* res = (double*)&ymm2;
		k = 4*j;
		ymm2 = _mm256_setzero_pd();
		for(uint64_t i=0;i<k;i++)
		{
			ymm0 = _mm256_broadcast_sd(in1+i);
			ymm1 = _mm256_loadu_pd(in2+N-4-k+i);
			ymm1 = _mm256_permute4x64_pd(ymm1,0b00011011);
			ymm2 = _mm256_add_pd(ymm2,_mm256_mul_pd(ymm0,ymm1));
		}
		ymm0 = _mm256_loadu_pd(in1+k); 
		for(uint64_t i=k;i<(k+4);i++)
		{
			ymm1 = _mm256_broadcast_sd(in2+N-1-i);
			ymm2 = _mm256_add_pd(ymm2,_mm256_mul_pd(ymm0,ymm1));
			out[i] = res[0];
			ymm2 = _mm256_permute4x64_pd(ymm2,0b00111001);
		}
	}
	for(uint64_t j=(4*N2);j<(N-1);j++)
	{
		out[j] = std::inner_product(in2+N-j-1,in2+N,in1,0.0);
	}
}

template<>
void filterEdgeLeftAVX<float,float>(uint64_t N, float* in1, float* in2, float* out)
{
	uint64_t N2 = (N-1)/8;
	#pragma omp parallel for schedule(dynamic,1)
	for(uint64_t j=0;j<N2;j++)
	{
		__m256 ymm0,ymm1,ymm2;
		__m256i ymm3,ymm4;
		ymm3 = _mm256_set_epi32(0,1,2,3,4,5,6,7);
		ymm4 = _mm256_set_epi32(0,7,6,5,4,3,2,1);
		uint64_t k;
		float* res = (float*)&ymm2;
		k = 8*j;
		ymm2 = _mm256_setzero_ps();
		for(uint64_t i=0;i<k;i++)
		{
			ymm0 = _mm256_broadcast_ss(in1+i);
			ymm1 = _mm256_loadu_ps(in2+N-8-k+i);
			ymm1 = _mm256_permutevar8x32_ps(ymm1,ymm3);
			ymm2 = _mm256_add_ps(ymm2,_mm256_mul_ps(ymm0,ymm1));
		}
		ymm0 = _mm256_loadu_ps(in1+k); 
		for(uint64_t i=k;i<(k+8);i++)
		{
			ymm1 = _mm256_broadcast_ss(in2+N-1-i);
			ymm2 = _mm256_add_ps(ymm2,_mm256_mul_ps(ymm0,ymm1));
			out[i] = res[0];
			ymm2 = _mm256_permutevar8x32_ps(ymm2,ymm4);
		}
	}
	for(uint64_t j=(8*N2);j<(N-1);j++)
	{
		out[j] = std::inner_product(in2+N-j-1,in2+N,in1,0.0);
	}
}

template<class DataTypeIn, class DataTypeOut>
void filterEdgeRightAVX(uint64_t N, DataTypeIn* in1, DataTypeOut* in2, DataTypeOut* out){}

template<>
void filterEdgeRightAVX<double,double>(uint64_t N, double* in1, double* in2, double* out)
{
	uint64_t N2 = N/4;
	#pragma omp parallel for schedule(dynamic,1)
	for(uint64_t j=0;j<N2;j++)
	{
		__m256d ymm0,ymm1,ymm2;
		uint64_t k;
		double* res = (double*)&ymm2;
		k = 4*j;
		ymm2 = _mm256_setzero_pd();
		for(uint64_t i=0;i<k;i++)
		{
			ymm0 = _mm256_broadcast_sd(in1-i);
			ymm1 = _mm256_loadu_pd(in2+N-4-i);
			ymm2 = _mm256_add_pd(ymm2,_mm256_mul_pd(ymm0,ymm1));
		}
		ymm0 = _mm256_loadu_pd(in1-3-k); 
		ymm0 = _mm256_permute4x64_pd(ymm0,0b00011011);
		for(uint64_t i=k;i<(k+4);i++)
		{
			ymm1 = _mm256_broadcast_sd(in2+i-k);
			ymm2 = _mm256_add_pd(ymm2,_mm256_mul_pd(ymm0,ymm1));
			(out-i)[0] = res[0];
			ymm2 = _mm256_permute4x64_pd(ymm2,0b00111001);
		}
	}
	for(uint64_t j=(4*N2);j<N;j++)
	{
		(out-j)[0] = std::inner_product(in2,in2+j+1,in1-j,0.0);
	}
}

template<>
void filterEdgeRightAVX<float,float>(uint64_t N, float* in1, float* in2, float* out)
{
	uint64_t N2 = N/8;

	#pragma omp parallel for schedule(dynamic,1)
	for(uint64_t j=0;j<N2;j++)
	{
		__m256 ymm0,ymm1,ymm2;
		__m256i ymm3,ymm4;
		ymm3 = _mm256_set_epi32(0,1,2,3,4,5,6,7);
		ymm4 = _mm256_set_epi32(0,7,6,5,4,3,2,1);
		uint64_t k;
		float* res = (float*)&ymm2;
		k = 8*j;
		ymm2 = _mm256_setzero_ps();
		for(uint64_t i=0;i<k;i++)
		{
			ymm0 = _mm256_broadcast_ss(in1-i);
			ymm1 = _mm256_loadu_ps(in2+N-8-i);
			ymm2 = _mm256_add_ps(ymm2,_mm256_mul_ps(ymm0,ymm1));
		}
		ymm0 = _mm256_loadu_ps(in1-7-k); 
		ymm0 = _mm256_permutevar8x32_ps(ymm0,ymm3);
		for(uint64_t i=k;i<(k+8);i++)
		{
			ymm1 = _mm256_broadcast_ss(in2+i-k);
			ymm2 = _mm256_add_ps(ymm2,_mm256_mul_ps(ymm0,ymm1));
			(out-i)[0] = res[0];
			ymm2 = _mm256_permutevar8x32_ps(ymm2,ymm4);
		}
	}
	for(uint64_t j=(8*N2);j<N;j++)
	{
		(out-j)[0] = std::inner_product(in2,in2+j+1,in1-j,0.0);
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

template<>
void filterAVX<uint8_t,uint32_t>(uint64_t N, uint64_t Nfilter, uint8_t* data, uint8_t* filter, uint32_t* out)
{
	__m256i ymm0,ymm1,ymm2,ymm3,ymm4,ymm5,ymm6,ymm7,ymm8,ymm9,ymm10,ymm11,ymm15;
	uint32_t *res1,*res2,*res3,*res4,*res5,*res6,*res7,*res8;
	uint64_t N2 = N/64;
	uint64_t k = 0;
	ymm15 = _mm256_set1_epi16(0);
	for(uint64_t j=0;j<N2;j++)
	{
		k = 64*j;
		res1 = out+k;
		res2 = out+k+8;
		res3 = out+k+16;
		res4 = out+k+24;
		res5 = out+k+32;
		res6 = out+k+40;
		res7 = out+k+48;
		res8 = out+k+56;
		ymm4 = _mm256_set1_epi16(0);
		ymm5 = _mm256_set1_epi16(0);
		ymm6 = _mm256_set1_epi16(0);
		ymm7 = _mm256_set1_epi16(0);
		ymm8 = _mm256_set1_epi16(0);
		ymm9 = _mm256_set1_epi16(0);
		ymm10 = _mm256_set1_epi16(0);
		ymm11 = _mm256_set1_epi16(0);
		for(uint64_t i=0;i<Nfilter;i++)
		{
			ymm0 = _mm256_set1_epi16(filter[i]);
			
			ymm1 = _mm256_loadu_si256((const __m256i*)(data+k+i));
			ymm2 = _mm256_unpacklo_epi8(ymm1,ymm15);	
			ymm3 = _mm256_mullo_epi16(ymm0,ymm2);
			ymm2 = _mm256_mulhi_epi16(ymm0,ymm2);
			ymm4 = _mm256_add_epi32(ymm4,_mm256_unpacklo_epi16(ymm3,ymm2));
			ymm5 = _mm256_add_epi32(ymm5,_mm256_unpackhi_epi16(ymm3,ymm2));
			ymm2 = _mm256_unpackhi_epi8(ymm1,ymm15);	
			ymm3 = _mm256_mullo_epi16(ymm0,ymm2);
			ymm2 = _mm256_mulhi_epi16(ymm0,ymm2);
			ymm6 = _mm256_add_epi32(ymm6,_mm256_unpacklo_epi16(ymm3,ymm2));
			ymm7 = _mm256_add_epi32(ymm7,_mm256_unpackhi_epi16(ymm3,ymm2));

			ymm1 = _mm256_loadu_si256((const __m256i*)(data+k+i+32));
			ymm2 = _mm256_unpacklo_epi8(ymm1,ymm15);	
			ymm3 = _mm256_mullo_epi16(ymm0,ymm2);
			ymm2 = _mm256_mulhi_epi16(ymm0,ymm2);
			ymm8 = _mm256_add_epi32(ymm8,_mm256_unpacklo_epi16(ymm3,ymm2));
			ymm9 = _mm256_add_epi32(ymm9,_mm256_unpackhi_epi16(ymm3,ymm2));
			ymm2 = _mm256_unpackhi_epi8(ymm1,ymm15);	
			ymm3 = _mm256_mullo_epi16(ymm0,ymm2);
			ymm2 = _mm256_mulhi_epi16(ymm0,ymm2);
			ymm10 = _mm256_add_epi32(ymm10,_mm256_unpacklo_epi16(ymm3,ymm2));
			ymm11 = _mm256_add_epi32(ymm11,_mm256_unpackhi_epi16(ymm3,ymm2));
		}
		ymm1 = _mm256_permute2x128_si256(ymm4, ymm5,0b00100000);
		ymm4 = _mm256_permute2x128_si256(ymm4, ymm5,0b00110001);
		ymm5 = _mm256_permute2x128_si256(ymm6, ymm7,0b00100000);
		ymm6 = _mm256_permute2x128_si256(ymm6, ymm7,0b00110001);
		ymm3 = _mm256_permute2x128_si256(ymm8, ymm9,0b00100000);
		ymm8 = _mm256_permute2x128_si256(ymm8, ymm9,0b00110001);
		ymm9 = _mm256_permute2x128_si256(ymm10, ymm11,0b00100000);
		ymm10 = _mm256_permute2x128_si256(ymm10, ymm11,0b00110001);
		_mm256_storeu_si256((__m256i*)res1,ymm1);
		_mm256_storeu_si256((__m256i*)res2,ymm4);
		_mm256_storeu_si256((__m256i*)res3,ymm5);
		_mm256_storeu_si256((__m256i*)res4,ymm6);
		_mm256_storeu_si256((__m256i*)res5,ymm3);
		_mm256_storeu_si256((__m256i*)res6,ymm8);
		_mm256_storeu_si256((__m256i*)res7,ymm9);
		_mm256_storeu_si256((__m256i*)res8,ymm10);
	}
	for(uint64_t j=(64*N2);j<N;j++)
	{
		out[j] = std::inner_product(filter,filter+Nfilter,data+j,0);
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

/*
template<class DataType>
void overlap_discard(uint64_t Ndata, uint64_t Nfilter, DataType* data, 
				DataType* out, DataType* filter){}

template<>
void overlap_discard(uint64_t Ndata, uint64_t Nfilter, double* data, double* out, double* filter)
{
	// Import FFTW wisdom
	fftw_import_wisdom_from_filename(&wisdom_path[0]);

	// Define parameters
	uint64_t Nfft = std::min(Ndata,std::max(2048,1<<((uint64_t)(3+std::log2(Nfilter))));
	uint64_t step = Nfft-Nfilter+1;
	
	// Initialize the filter
	double* filter_freq = (double*) malloc((Nfft+2)*sizeof(double));
	std::memcpy(filter,filter_freq,Nfilter*sizeof(double));

	// Compute the filter's FFT
	fftw_plan plan;
	plan = fftw_plan_dft_r2c_1d(
					Nfft, 
					reinterpret_cast<double*>(filter_freq), 
					reinterpret_cast<fftw_complex*>(filter_freq), 
					FFTW_EXHAUSTIVE);
	fftw_execute(plan); fftw_destroy_plan(plan);

	// Compute the first edge
	std::memset(0,out,(Nfft+2)*sizeof(double));
	std::memcpy(data,out,(Nfft/2)*sizeof(double));
	
	fftw_plan plan;
	plan = fftw_plan_dft_r2c_1d(
					Nfft, 
					out, 
					reinterpret_cast<fftw_complex*>(out), 
					FFTW_EXHAUSTIVE);
	fftw_execute(plan); fftw_destroy_plan(plan);

	for(uint64_t i=0;i<(Nfft+2);i++)
	{
		out[i] *= filter_freq[i];
	}

	fftw_plan plan;
	plan = fftw_plan_dft_c2r_1d(
					Nfft, 
					reinterpret_cast<fftw_complex*>(out), 
					out, 
					FFTW_EXHAUSTIVE);
	fftw_execute(plan); fftw_destroy_plan(plan);

	// Free the temporary buffer
	free(filter_freq);

	// Export and forget the wisdom
    fftw_export_wisdom_to_filename(&wisdom_path[0]);	
	fftw_forget_wisdom();
}*/
