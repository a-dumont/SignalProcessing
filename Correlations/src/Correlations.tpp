///////////////////////////////////////////////////////////////////
//                       _    ____                               //
//                      / \  / ___|___  _ __ _ __                //
//                     / _ \| |   / _ \| '__| '__|               //
//                    / ___ \ |__| (_) | |  | |                  //
//                   /_/   \_\____\___/|_|  |_|                  //
///////////////////////////////////////////////////////////////////
template<class DataType>
void aCorrCircularFreqAVX(uint64_t N, DataType* in, DataType* out){}

template<>
void aCorrCircularFreqAVX<float>(uint64_t N, float* in, float* out)
{
	__m256 ymm0;
	uint64_t howmany = N/8;
	for(uint64_t i=0;i<howmany;i++)
	{
		ymm0 = _mm256_loadu_ps(in+8*i);
		ymm0 = _mm256_mul_ps(ymm0,ymm0);
		_mm256_storeu_ps(out+8*i,ymm0);
	}
	for(uint64_t i=(8*howmany);i<N;i++){out[i] = in[i]*in[i];}
}

template<>
void aCorrCircularFreqAVX<double>(uint64_t N, double* in, double* out)
{
	__m256d ymm0;
	uint64_t howmany = N/4;
	for(uint64_t i=0;i<howmany;i++)
	{
		ymm0 = _mm256_loadu_pd(in+4*i);
		ymm0 = _mm256_mul_pd(ymm0,ymm0);
		_mm256_storeu_pd(out+4*i,ymm0);
	}
	for(uint64_t j=(4*howmany);j<N;j++){out[j] = in[j]*in[j];}
}
///////////////////////////////////////////////////////////////////
//                      __  ______                               //
//                      \ \/ / ___|___  _ __ _ __                //
//                       \  / |   / _ \| '__| '__|               //
//                       /  \ |__| (_) | |  | |                  //
//                      /_/\_\____\___/|_|  |_|                  //
///////////////////////////////////////////////////////////////////
template<class DataType>
void xCorrCircularFreqAVX(uint64_t N, DataType* in1, DataType* in2, DataType* out){}

template<>
void xCorrCircularFreqAVX<float>(uint64_t N, float* in1, float* in2, float* out)
{
	__m256 ymm0, ymm1, ymm2, ymm3, ymm4;
	uint64_t howmany = N/8;
	ymm4 = _mm256_set_ps(-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0);
	for(uint64_t i=0;i<howmany;i++)
	{
		ymm0 = _mm256_loadu_ps(in1+8*i);
		ymm1 = _mm256_loadu_ps(in2+8*i);
		ymm2 = _mm256_mul_ps(ymm0,ymm1);
		ymm1 = _mm256_mul_ps(ymm1,ymm4);
		ymm1 = _mm256_permute_ps(ymm1,0b10110001);
		ymm3 = _mm256_mul_ps(ymm0,ymm1);
		ymm0 = _mm256_hadd_ps(ymm2,ymm3);
		ymm0 = _mm256_permute_ps(ymm0,0b11011000);
		_mm256_storeu_ps(out+8*i,ymm0);
	}
	for(uint64_t i=(8*howmany);i<N;i+=2)
	{
		out[i] = in1[i]*in2[i]+in1[i+1]*in2[i+1];
		out[i+1] = in1[i+1]*in2[i]-in1[i]*in2[i+1];
	}
}

template<>
void xCorrCircularFreqAVX<double>(uint64_t N, double* in1, double* in2, double* out)
{
	__m256d ymm0, ymm1, ymm2, ymm3, ymm4;
	uint64_t howmany = N/4;
	ymm4 = _mm256_set_pd(-1.0,1.0,-1.0,1.0);
	for(uint64_t i=0;i<howmany;i++)
	{
		ymm0 = _mm256_loadu_pd(in1+4*i);
		ymm1 = _mm256_loadu_pd(in2+4*i);
		ymm2 = _mm256_mul_pd(ymm0,ymm1);
		ymm1 = _mm256_mul_pd(ymm1,ymm4);
		ymm1 = _mm256_permute_pd(ymm1,0b00000101);
		ymm3 = _mm256_mul_pd(ymm0,ymm1);
		ymm0 = _mm256_hadd_pd(ymm2,ymm3);
		_mm256_storeu_pd(out+4*i,ymm0);
	}
	for(uint64_t i=(4*howmany);i<N;i+=2)
	{
		out[i] = in1[i]*in2[i]+in1[i+1]*in2[i+1];
		out[i+1] = in1[i+1]*in2[i]-in1[i]*in2[i+1];
	}
}

///////////////////////////////////////////////////////////////////
//                       _____ ____                              //
//                      |  ___/ ___|___  _ __ _ __               //
//                      | |_ | |   / _ \| '__| '__|              //
//                      |  _|| |__| (_) | |  | |                 //
//                      |_|   \____\___/|_|  |_|                 //
///////////////////////////////////////////////////////////////////
template<class DataType>
void fCorrCircularFreqAVX(uint64_t N, DataType* in1, DataType* in2, 
				DataType* out1, DataType* out2, DataType* out3){}

template<>
void fCorrCircularFreqAVX<float>(uint64_t N, float* in1, float* in2, 
				float* out1, float* out2, float* out3)
{
	float temp1, temp2, temp3, temp4;
	__m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6;
	uint64_t howmany = N/8;
	ymm4 = _mm256_set_ps(-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0);
	for(uint64_t i=0;i<howmany;i++)
	{
		ymm0 = _mm256_loadu_ps(in1+8*i);
		ymm1 = _mm256_loadu_ps(in2+8*i);
		ymm5 = _mm256_mul_ps(ymm0,ymm0);
		ymm6 = _mm256_mul_ps(ymm1,ymm1);
		_mm256_storeu_ps(out1+8*i,ymm5);
		_mm256_storeu_ps(out2+8*i,ymm6);
		ymm2 = _mm256_mul_ps(ymm0,ymm1);
		ymm1 = _mm256_mul_ps(ymm1,ymm4);
		ymm1 = _mm256_permute_ps(ymm1,0b10110001);
		ymm3 = _mm256_mul_ps(ymm0,ymm1);
		ymm0 = _mm256_hadd_ps(ymm2,ymm3);
		ymm0 = _mm256_permute_ps(ymm0,0b11011000);
		_mm256_storeu_ps(out3+8*i,ymm0);
	}
	for(uint64_t i=(8*howmany);i<N;i+=2)
	{
		temp1 = in1[i]; temp2 = in1[i+1]; temp3 = in2[i]; temp4 = in2[i+1];
		out1[i] = temp1*temp1;
		out1[i+1] = temp2*temp2;
		out2[i] = temp3*temp3;
		out2[i+1] = temp4*temp4;
		out3[i] = temp1*temp3+temp2*temp4;
		out3[i+1] = temp2*temp3-temp1*temp4;
	}
}

template<>
void fCorrCircularFreqAVX<double>(uint64_t N, double* in1, double* in2, 
				double* out1, double* out2, double* out3)
{
	double temp1, temp2, temp3, temp4;
	__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6;
	uint64_t howmany = N/4;
	ymm4 = _mm256_set_pd(-1.0,1.0,-1.0,1.0);
	for(uint64_t i=0;i<howmany;i++)
	{
		ymm0 = _mm256_loadu_pd(in1+4*i);
		ymm1 = _mm256_loadu_pd(in2+4*i);
		ymm5 = _mm256_mul_pd(ymm0,ymm0);
		ymm6 = _mm256_mul_pd(ymm1,ymm1);
		_mm256_storeu_pd(out1+4*i,ymm5);
		_mm256_storeu_pd(out2+4*i,ymm6);
		ymm2 = _mm256_mul_pd(ymm0,ymm1);
		ymm1 = _mm256_mul_pd(ymm1,ymm4);
		ymm1 = _mm256_permute_pd(ymm1,0b00000101);
		ymm3 = _mm256_mul_pd(ymm0,ymm1);
		ymm0 = _mm256_hadd_pd(ymm2,ymm3);
		_mm256_storeu_pd(out3+4*i,ymm0);
	}
	for(uint64_t i=(4*howmany);i<N;i+=2)
	{
		temp1 = in1[i]; temp2 = in1[i+1]; temp3 = in2[i]; temp4 = in2[i+1];
		out1[i] = temp1*temp1;
		out1[i+1] = temp2*temp2;
		out2[i] = temp3*temp3;
		out2[i+1] = temp4*temp4;
		out3[i] = temp1*temp3+temp2*temp4;
		out3[i+1] = temp2*temp3-temp1*temp4;
	}
}

///////////////////////////////////////////////////////////////////
//               ___ _____ _   _ _____ ____  ____                //
//				/ _ \_   _| | | | ____|  _ \/ ___|               //
//			   | | | || | | |_| |  _| | |_) \___ \               //
//			   | |_| || | |  _  | |___|  _ < ___) |              //
//				\___/ |_| |_| |_|_____|_| \_\____/               //
///////////////////////////////////////////////////////////////////
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


template<class DataType>
void rfftBlock(int N, int size, DataType* in, std::complex<DataType>* out){}

template<>
void rfftBlock<double>(int N, int size, double* in, std::complex<double>* out)
{

	int rank = 1;
	int length[] = {size};
	int howmany = N/size;
	int idist = size;
	int odist = size/2+1;
	int stride = 1;

	fftw_plan plan = fftw_plan_many_dft_r2c(
					rank,
					length,
					howmany,
					in,
					NULL,
					stride,
					idist,
					reinterpret_cast<fftw_complex*>(out),
					NULL,
					stride,
					odist,
					FFTW_ESTIMATE);

	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<>
void rfftBlock<float>(int N, int size, float* in, std::complex<float>* out)
{

	int rank = 1;
	int length[] = {size};
	int howmany = N/size;
	int idist = size;
	int odist = size/2+1;
	int stride = 1;

	fftwf_plan plan = fftwf_plan_many_dft_r2c(
					rank,
					length,
					howmany,
					in,
					NULL,
					stride,
					idist,
					reinterpret_cast<fftwf_complex*>(out),
					NULL,
					stride,
					odist,
					FFTW_ESTIMATE);

	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
}

template<class DataTypeIn, class DataTypeOut>
void convertAVX(uint64_t N, DataTypeIn* in, DataTypeOut* out, DataTypeOut conv, DataTypeIn offset)
{}

template<>
void convertAVX<uint8_t, float>(uint64_t N, uint8_t* in, float* out, float conv, uint8_t offset)
{
	__m128i xmm0;
	__m256i ymm0, ymm1, ymm12; 
	__m256 ymm2, ymm3, ymm13;
	float *out0, *out1;

	uint64_t howmany = N/16;
	uint64_t j=0;
	int32_t off = (int32_t) offset;

	ymm12 = _mm256_set_epi32(off,off,off,off,off,off,off,off);
	ymm13 = _mm256_broadcast_ss(&conv);
	
	for(uint64_t i=0;i<howmany;i++)
	{
		j = 16*i;
		out0 = out+j;
		out1 = out+j+8;

		// 16 uint8 to 16 int32
    	xmm0 = _mm_loadu_si128((const __m128i*) (in+j));
    	ymm0 = _mm256_cvtepu8_epi32(xmm0);
		ymm0 = _mm256_sub_epi32(ymm0,ymm12);
		ymm2 = _mm256_cvtepi32_ps(ymm0);
		ymm2 = _mm256_mul_ps(ymm2,ymm13);
    	xmm0 = _mm_bsrli_si128(xmm0,8);
    	ymm1 = _mm256_cvtepu8_epi32(xmm0);
		ymm1 = _mm256_sub_epi32(ymm1,ymm12);
		ymm3 = _mm256_cvtepi32_ps(ymm1);
		ymm3 = _mm256_mul_ps(ymm3,ymm13);

		// Store result
    	_mm256_storeu_ps(out0,ymm2);
    	_mm256_storeu_ps(out1,ymm3);
	}
	for(uint64_t i=(16*howmany);i<N;i++){out[i] = conv*(in[i]-offset);}
}

template<>
void convertAVX<int16_t, float>(uint64_t N, int16_t* in, float* out, float conv, int16_t offset)
{
	__m128i xmm0;
	__m256i ymm0, ymm12; 
	__m256 ymm1, ymm13;
	float *out0;

	uint64_t howmany = N/8;
	uint64_t j=0;
	int32_t off = (int32_t) offset;

	ymm12 = _mm256_set_epi32(off,off,off,off,off,off,off,off);
	ymm13 = _mm256_broadcast_ss(&conv);
	
	for(uint64_t i=0;i<howmany;i++)
	{
		j = 8*i;
		out0 = out+j;

		// 16 uint8 to 16 int32
    	xmm0 = _mm_loadu_si128((const __m128i*) (in+j));
    	ymm0 = _mm256_cvtepi16_epi32(xmm0);
		ymm0 = _mm256_sub_epi32(ymm0,ymm12);
		ymm1 = _mm256_cvtepi32_ps(ymm0);
		ymm1 = _mm256_mul_ps(ymm1,ymm13);

		// Store result
    	_mm256_storeu_ps(out0,ymm1);
	}
	for(uint64_t i=(8*howmany);i<N;i++){out[i] = conv*(in[i]-offset);}
}

template<>
void convertAVX<uint8_t, double>(uint64_t N, uint8_t* in, double* out,double conv,uint8_t offset)
{
	__m128i xmm0;
	__m256i ymm0, ymm1, ymm12;
	__m256d ymm2, ymm3, ymm4, ymm5, ymm13;
	double *out0, *out1, *out2, *out3;

	uint64_t howmany = N/16;
	uint64_t j=0;
	int32_t off = (int32_t) offset;

	ymm12 = _mm256_set_epi32(off,off,off,off,off,off,off,off);
	ymm13 = _mm256_broadcast_sd(&conv);
	
	for(uint64_t i=0;i<howmany;i++)
	{
		j = 16*i;
		out0 = out+j;
		out1 = out+j+4;
		out2 = out+j+8;
		out3 = out+j+12;

		// 16 uint8 to 16 double
    	xmm0 = _mm_loadu_si128((const __m128i*) (in+j));
    	ymm0 = _mm256_cvtepu8_epi32(xmm0);
		ymm0 = _mm256_sub_epi32(ymm0,ymm12);
		ymm2 = _mm256_cvtepi32_pd(_mm256_extractf128_si256(ymm0, 0));
		ymm3 = _mm256_cvtepi32_pd(_mm256_extractf128_si256(ymm0, 1));
		ymm2 = _mm256_mul_pd(ymm2,ymm13);
		ymm3 = _mm256_mul_pd(ymm3,ymm13);
		
		xmm0 = _mm_bsrli_si128(xmm0,8);
    	ymm1 = _mm256_cvtepu8_epi32(xmm0);
		ymm1 = _mm256_sub_epi32(ymm1,ymm12);
		ymm4 = _mm256_cvtepi32_pd(_mm256_extractf128_si256(ymm1, 0));
		ymm5 = _mm256_cvtepi32_pd(_mm256_extractf128_si256(ymm1, 1));
		ymm4 = _mm256_mul_pd(ymm4,ymm13);
		ymm5 = _mm256_mul_pd(ymm5,ymm13);

		// Store result
    	_mm256_storeu_pd(out0,ymm2);
    	_mm256_storeu_pd(out1,ymm3);
    	_mm256_storeu_pd(out2,ymm4);
    	_mm256_storeu_pd(out3,ymm5);
	}
	for(uint64_t i=(16*howmany);i<N;i++){out[i] = conv*(in[i]-offset);}
}

template<>
void convertAVX<int16_t, double>(uint64_t N, int16_t* in, double* out, double conv, int16_t offset)
{
	__m128i xmm0;
	__m256i ymm0, ymm12; 
	__m256d ymm1, ymm2, ymm13;
	double *out0, *out1;

	uint64_t howmany = N/8;
	uint64_t j=0;
	int32_t off = (int32_t) offset;

	ymm12 = _mm256_set_epi32(off,off,off,off,off,off,off,off);
	ymm13 = _mm256_broadcast_sd(&conv);
	
	for(uint64_t i=0;i<howmany;i++)
	{
		j = 8*i;
		out0 = out+j;
		out1 = out+j+4;

		// 16 uint8 to 16 int32
    	xmm0 = _mm_loadu_si128((const __m128i*) (in+j));
    	ymm0 = _mm256_cvtepi16_epi32(xmm0);
		ymm0 = _mm256_sub_epi32(ymm0,ymm12);
		ymm1 = _mm256_cvtepi32_pd(_mm256_extractf128_si256(ymm0,0));
		ymm1 = _mm256_mul_pd(ymm1,ymm13);
		ymm2 = _mm256_cvtepi32_pd(_mm256_extractf128_si256(ymm0,1));
		ymm2 = _mm256_mul_pd(ymm2,ymm13);

		// Store result
    	_mm256_storeu_pd(out0,ymm1);
    	_mm256_storeu_pd(out1,ymm2);
	}
	for(uint64_t i=(8*howmany);i<N;i++){out[i] = conv*(in[i]-offset);}
}

template<class DataTypeIn, class DataTypeOut>
void convertAVX_pad(uint64_t N, uint64_t Npad, 
				DataTypeIn* in, DataTypeOut* out, DataTypeOut conv, DataTypeIn offset)
{}

template<>
void convertAVX_pad<uint8_t, float>(uint64_t N, uint64_t Npad, 
				uint8_t* in, float* out, float conv, uint8_t offset)
{
	__m128i xmm0;
	__m256i ymm0, ymm1, ymm12; 
	__m256 ymm2, ymm3, ymm13;
	float *out0, *out1;

	uint64_t howmany = N/16;
	uint64_t j=0;
	uint64_t k=0;
	int32_t off = (int32_t) offset;

	ymm12 = _mm256_set_epi32(off,off,off,off,off,off,off,off);
	ymm13 = _mm256_broadcast_ss(&conv);
	
	for(uint64_t i=0;i<howmany;i++)
	{
		j = 16*i;
		k = 2*(j/Npad);
		out0 = out+j+k;
		out1 = out+j+8+k;

		// 16 uint8 to 16 int32
    	xmm0 = _mm_loadu_si128((const __m128i*) (in+j));
    	ymm0 = _mm256_cvtepu8_epi32(xmm0);
		ymm0 = _mm256_sub_epi32(ymm0,ymm12);
		ymm2 = _mm256_cvtepi32_ps(ymm0);
		ymm2 = _mm256_mul_ps(ymm2,ymm13);
    	xmm0 = _mm_bsrli_si128(xmm0,8);
    	ymm1 = _mm256_cvtepu8_epi32(xmm0);
		ymm1 = _mm256_sub_epi32(ymm1,ymm12);
		ymm3 = _mm256_cvtepi32_ps(ymm1);
		ymm3 = _mm256_mul_ps(ymm3,ymm13);

		// Store result
    	_mm256_storeu_ps(out0,ymm2);
    	_mm256_storeu_ps(out1,ymm3);
	}
	for(uint64_t i=(16*howmany);i<N;i++){out[i] = conv*(in[i]-offset);}
}

template<>
void convertAVX_pad<int16_t, float>(uint64_t N, uint64_t Npad, 
				int16_t* in, float* out, float conv, int16_t offset)
{
	__m128i xmm0;
	__m256i ymm0, ymm12; 
	__m256 ymm1, ymm13;
	float *out0;

	uint64_t howmany = N/8;
	uint64_t j=0;
	uint64_t k=0;
	int32_t off = (int32_t) offset;

	ymm12 = _mm256_set_epi32(off,off,off,off,off,off,off,off);
	ymm13 = _mm256_broadcast_ss(&conv);
	
	for(uint64_t i=0;i<howmany;i++)
	{
		j = 8*i;
		k = 2*(j/Npad);
		out0 = out+j+k;

		// 16 uint8 to 16 int32
    	xmm0 = _mm_loadu_si128((const __m128i*) (in+j));
    	ymm0 = _mm256_cvtepi16_epi32(xmm0);
		ymm0 = _mm256_sub_epi32(ymm0,ymm12);
		ymm1 = _mm256_cvtepi32_ps(ymm0);
		ymm1 = _mm256_mul_ps(ymm1,ymm13);

		// Store result
    	_mm256_storeu_ps(out0,ymm1);
	}
	for(uint64_t i=(8*howmany);i<N;i++){out[i] = conv*(in[i]-offset);}
}

template<>
void convertAVX_pad<uint8_t, double>(uint64_t N, uint64_t Npad, 
				uint8_t* in, double* out,double conv,uint8_t offset)
{
	__m128i xmm0;
	__m256i ymm0, ymm1, ymm12;
	__m256d ymm2, ymm3, ymm4, ymm5, ymm13;
	double *out0, *out1, *out2, *out3;

	uint64_t howmany = N/16;
	uint64_t j=0;
	uint64_t k=0;
	int32_t off = (int32_t) offset;

	ymm12 = _mm256_set_epi32(off,off,off,off,off,off,off,off);
	ymm13 = _mm256_broadcast_sd(&conv);
	
	for(uint64_t i=0;i<howmany;i++)
	{
		j = 16*i;
		k = 2*(j/Npad);
		out0 = out+j+k;
		out1 = out+j+4+k;
		out2 = out+j+8+k;
		out3 = out+j+12+k;

		// 16 uint8 to 16 double
    	xmm0 = _mm_loadu_si128((const __m128i*) (in+j));
    	ymm0 = _mm256_cvtepu8_epi32(xmm0);
		ymm0 = _mm256_sub_epi32(ymm0,ymm12);
		ymm2 = _mm256_cvtepi32_pd(_mm256_extractf128_si256(ymm0, 0));
		ymm3 = _mm256_cvtepi32_pd(_mm256_extractf128_si256(ymm0, 1));
		ymm2 = _mm256_mul_pd(ymm2,ymm13);
		ymm3 = _mm256_mul_pd(ymm3,ymm13);
		
		xmm0 = _mm_bsrli_si128(xmm0,8);
    	ymm1 = _mm256_cvtepu8_epi32(xmm0);
		ymm1 = _mm256_sub_epi32(ymm1,ymm12);
		ymm4 = _mm256_cvtepi32_pd(_mm256_extractf128_si256(ymm1, 0));
		ymm5 = _mm256_cvtepi32_pd(_mm256_extractf128_si256(ymm1, 1));
		ymm4 = _mm256_mul_pd(ymm4,ymm13);
		ymm5 = _mm256_mul_pd(ymm5,ymm13);

		// Store result
    	_mm256_storeu_pd(out0,ymm2);
    	_mm256_storeu_pd(out1,ymm3);
    	_mm256_storeu_pd(out2,ymm4);
    	_mm256_storeu_pd(out3,ymm5);
	}
	for(uint64_t i=(16*howmany);i<N;i++){out[i] = conv*(in[i]-offset);}
}

template<>
void convertAVX_pad<int16_t, double>(uint64_t N, uint64_t Npad, 
				int16_t* in, double* out, double conv, int16_t offset)
{
	__m128i xmm0;
	__m256i ymm0, ymm12; 
	__m256d ymm1, ymm2, ymm13;
	double *out0, *out1;

	uint64_t howmany = N/8;
	uint64_t j=0;
	uint64_t k=0;
	int32_t off = (int32_t) offset;

	ymm12 = _mm256_set_epi32(off,off,off,off,off,off,off,off);
	ymm13 = _mm256_broadcast_sd(&conv);
	
	for(uint64_t i=0;i<howmany;i++)
	{
		j = 8*i;
		k = 2*(j/Npad);
		out0 = out+j+k;
		out1 = out+j+4+k;

		// 16 uint8 to 16 int32
    	xmm0 = _mm_loadu_si128((const __m128i*) (in+j));
    	ymm0 = _mm256_cvtepi16_epi32(xmm0);
		ymm0 = _mm256_sub_epi32(ymm0,ymm12);
		ymm1 = _mm256_cvtepi32_pd(_mm256_extractf128_si256(ymm0,0));
		ymm1 = _mm256_mul_pd(ymm1,ymm13);
		ymm2 = _mm256_cvtepi32_pd(_mm256_extractf128_si256(ymm0,1));
		ymm2 = _mm256_mul_pd(ymm2,ymm13);

		// Store result
    	_mm256_storeu_pd(out0,ymm1);
    	_mm256_storeu_pd(out1,ymm2);
	}
	for(uint64_t i=(8*howmany);i<N;i++){out[i] = conv*(in[i]-offset);}
}

template<class DataType>
void reduceAVX(uint64_t N, DataType* in, DataType* out){}

template<>
void reduceAVX<float>(uint64_t N, float* in, float* out)
{
	__m256 ymm0,ymm1,ymm2,ymm3,ymm4,ymm5,ymm6,ymm7,ymm8,ymm9,ymm10,ymm11,ymm12,ymm13,ymm14,ymm15;
	float* fymm0 = (float*) &ymm0;

	float* data=in; 
	float* result=out;

	uint64_t powers[10];
	for(int i=9;i>=0;i--){powers[i]=N>>(7*i);N^=(N>>(7*i)<<(7*i));}
	
	result[0] = std::accumulate(data,data+powers[0],0.0);
	
	uint64_t offset = powers[0];
	data = in+offset;

	for(uint8_t i=1;i<10;i++)
	{
		for(uint8_t j=0;j<i;j++)
		{
			for(uint64_t k=0;k<powers[i]*(1<<(7*(i-j)));k+=128)
			{
				ymm0 = _mm256_loadu_ps(data+k);
				ymm1 = _mm256_loadu_ps(data+k+8);
				ymm0 = _mm256_add_ps(ymm0,ymm1);
				
				ymm2 = _mm256_loadu_ps(data+k+16);
				ymm3 = _mm256_loadu_ps(data+k+24);
				ymm2 = _mm256_add_ps(ymm2,ymm3);

				ymm4 = _mm256_loadu_ps(data+k+32);
				ymm5 = _mm256_loadu_ps(data+k+40);
				ymm4 = _mm256_add_ps(ymm4,ymm5);
				
				ymm6 = _mm256_loadu_ps(data+k+48);
				ymm7 = _mm256_loadu_ps(data+k+56);
				ymm6 = _mm256_add_ps(ymm6,ymm7);
							
				ymm8 = _mm256_loadu_ps(data+k+64);
				ymm9 = _mm256_loadu_ps(data+k+72);
				ymm8 = _mm256_add_ps(ymm8,ymm9);

				ymm10 = _mm256_loadu_ps(data+k+80);
				ymm11 = _mm256_loadu_ps(data+k+88);
				ymm10 = _mm256_add_ps(ymm10,ymm11);

				ymm12 = _mm256_loadu_ps(data+k+96);
				ymm13 = _mm256_loadu_ps(data+k+104);
				ymm12 = _mm256_add_ps(ymm12,ymm13);

				ymm14 = _mm256_loadu_ps(data+k+112);
				ymm15 = _mm256_loadu_ps(data+k+120);
				ymm14 = _mm256_add_ps(ymm14,ymm15);
				
				ymm0 = _mm256_add_ps(ymm0,ymm2);
				ymm4 = _mm256_add_ps(ymm4,ymm6);
				ymm8 = _mm256_add_ps(ymm8,ymm10);
				ymm12 = _mm256_add_ps(ymm12,ymm14);

				ymm0 = _mm256_add_ps(ymm0,ymm4);
				ymm8 = _mm256_add_ps(ymm8,ymm12);

				ymm0 = _mm256_add_ps(ymm0,ymm8);

				ymm0 = _mm256_hadd_ps(ymm0,ymm0);
				ymm0 = _mm256_hadd_ps(ymm0,ymm0);
				result[i+k/128] = fymm0[0]+fymm0[4];
			}
			data = result+i;
		}
		result[i] = std::accumulate(result+i,result+i+powers[i],0.0);
		offset += powers[i]*(1<<(7*i));
		data = in+offset;
	}
	for(int i=1;i<10;i++){result[0]+=result[i];}
}

template<>
void reduceAVX<double>(uint64_t N, double* in, double* out)
{
	__m256d ymm0,ymm1,ymm2,ymm3,ymm4,ymm5,ymm6,ymm7,ymm8,ymm9,ymm10,ymm11,ymm12,ymm13,ymm14,ymm15;
	double* fymm0 = (double*) &ymm0;

	double* data=in; 
	double* result=out;

	uint64_t powers[11];
	for(int i=10;i>=0;i--){powers[i]=N>>(6*i);N^=(N>>(6*i)<<(6*i));}
	
	result[0] = std::accumulate(data,data+powers[0],0.0);
	
	uint64_t offset = powers[0];
	data = in+offset;

	for(uint8_t i=1;i<11;i++)
	{
		for(uint8_t j=0;j<i;j++)
		{
			for(uint64_t k=0;k<powers[i]*(1<<(6*(i-j)));k+=64)
			{
				ymm0 = _mm256_loadu_pd(data+k);
				ymm1 = _mm256_loadu_pd(data+k+4);
				ymm0 = _mm256_add_pd(ymm0,ymm1);
				
				ymm2 = _mm256_loadu_pd(data+k+8);
				ymm3 = _mm256_loadu_pd(data+k+12);
				ymm2 = _mm256_add_pd(ymm2,ymm3);

				ymm4 = _mm256_loadu_pd(data+k+16);
				ymm5 = _mm256_loadu_pd(data+k+20);
				ymm4 = _mm256_add_pd(ymm4,ymm5);
				
				ymm6 = _mm256_loadu_pd(data+k+24);
				ymm7 = _mm256_loadu_pd(data+k+28);
				ymm6 = _mm256_add_pd(ymm6,ymm7);
							
				ymm8 = _mm256_loadu_pd(data+k+32);
				ymm9 = _mm256_loadu_pd(data+k+36);
				ymm8 = _mm256_add_pd(ymm8,ymm9);

				ymm10 = _mm256_loadu_pd(data+k+40);
				ymm11 = _mm256_loadu_pd(data+k+44);
				ymm10 = _mm256_add_pd(ymm10,ymm11);

				ymm12 = _mm256_loadu_pd(data+k+48);
				ymm13 = _mm256_loadu_pd(data+k+52);
				ymm12 = _mm256_add_pd(ymm12,ymm13);

				ymm14 = _mm256_loadu_pd(data+k+56);
				ymm15 = _mm256_loadu_pd(data+k+60);
				ymm14 = _mm256_add_pd(ymm14,ymm15);
				
				ymm0 = _mm256_add_pd(ymm0,ymm2);
				ymm4 = _mm256_add_pd(ymm4,ymm6);
				ymm8 = _mm256_add_pd(ymm8,ymm10);
				ymm12 = _mm256_add_pd(ymm12,ymm14);

				ymm0 = _mm256_add_pd(ymm0,ymm4);
				ymm8 = _mm256_add_pd(ymm8,ymm12);

				ymm0 = _mm256_add_pd(ymm0,ymm8);

				ymm0 = _mm256_hadd_pd(ymm0,ymm0);
				result[i+k/64] = fymm0[0]+fymm0[2];
			}
			data = result+i;
		}
		result[i] = std::accumulate(result+i,result+i+powers[i],0.0);
		offset += powers[i]*(1<<(6*i));
		data = in+offset;
	}
	for(int i=1;i<11;i++){result[0]+=result[i];}
}

template<class DataType>
void reduceInPlaceBlockAVX(uint64_t N, uint64_t size, DataType* data){}

template<>
void reduceInPlaceBlockAVX<float>(uint64_t N, uint64_t size, float* data)
{
	__m256 ymm0,ymm1;

	uint64_t howmany = N/size;
	uint64_t registers = size/8;
	uint64_t excess = size-(registers*8);

	uint64_t powers[64];
	for(int i=63;i>=0;i--){powers[i]=(howmany>>i)&1;}

	uint64_t offset = powers[0]*size;
	if(offset != 0 && howmany != 1)
	{
		for(uint64_t i=0;i<size;i++){data[i]+=data[N-1-size+i];offset=0;}
	}

	if(howmany > 1)
	{
		for(uint8_t i=1;i<63;i++)
		{
			for(uint8_t j=0;j<i;j++)
			{
				for(uint64_t k=0;k<size*powers[i]*(1<<(i-j));k+=2*size)
				{
					for(uint64_t l=0;l<registers;l++)
					{
						ymm0 = _mm256_loadu_ps(data+k+l*8+offset);
						ymm1 = _mm256_loadu_ps(data+k+l*8+size+offset);
						ymm0 = _mm256_add_ps(ymm0,ymm1);
						_mm256_storeu_ps(data+k/2+l*8,ymm0);
					}
					for(uint64_t l=0;l<excess;l++)
					{
						data[k/2+8*registers+l] = data[k+8*registers+l+offset]
														+data[k+8*registers+size+l+offset];
					}
				}
			}
			offset += size*powers[i]*(1<<i);
		}
	}
}

template<>
void reduceInPlaceBlockAVX<double>(uint64_t N, uint64_t size, double* data)
{
	__m256d ymm0,ymm1;

	uint64_t howmany = N/size;
	uint64_t registers = size/4;
	uint64_t excess = size-(registers*4);

	uint64_t powers[64];
	for(int i=63;i>=0;i--){powers[i]=(howmany>>i)&1;}
	
	uint64_t offset = powers[0]*size;
	if(offset != 0 && howmany != 1)
	{
		for(uint64_t i=0;i<size;i++){data[i]+=data[N-1-size+i];offset=0;}
	}

	if(howmany > 1)
	{
		for(uint8_t i=1;i<63;i++)
		{
			for(uint8_t j=0;j<i;j++)
			{
				for(uint64_t k=0;k<size*powers[i]*(1<<(i-j));k+=2*size)
				{
					for(uint64_t l=0;l<registers;l++)
					{
						ymm0 = _mm256_loadu_pd(data+k+l*4+offset);
						ymm1 = _mm256_loadu_pd(data+k+l*4+size+offset);
						ymm0 = _mm256_add_pd(ymm0,ymm1);
						_mm256_storeu_pd(data+k/2+l*4,ymm0);
					}
					for(uint64_t l=0;l<excess;l++)
					{
						data[k/2+4*registers+l] = data[k+4*registers+l+offset]
														+data[k+4*registers+size+l+offset];
					}
				}
			}	
			offset += size*powers[i]*(1<<i);
		}
	}
}
