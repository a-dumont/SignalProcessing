///////////////////////////////////////////////////////////////////
//                       _    ____                               //
//                      / \  / ___|___  _ __ _ __                //
//                     / _ \| |   / _ \| '__| '__|               //
//                    / ___ \ |__| (_) | |  | |                  //
//                   /_/   \_\____\___/|_|  |_|                  //
///////////////////////////////////////////////////////////////////

#include <numeric>
template<class DataType>
void aCorrFreqAVX(uint64_t N, DataType* in, DataType* out){}

template<>
void aCorrFreqAVX<float>(uint64_t N, float* in, float* out)
{
	__m256 ymm0, ymm1, ymm2;
	__m256i ymm15;
	float *out0;

	uint64_t howmany = N/16;
	uint64_t j=0;

	ymm15 = _mm256_set_epi32(7,6,3,2,5,4,1,0);

	for(uint64_t i=0;i<howmany;i++)
	{
		j = 16*i;
		out0 = out+(j/2);

		// Acorr
		ymm0 = _mm256_loadu_ps(in+j);
		ymm1 = _mm256_loadu_ps(in+j+8);

		ymm0 = _mm256_mul_ps(ymm0,ymm0);
		ymm1 = _mm256_mul_ps(ymm1,ymm1);

		ymm2 = _mm256_hadd_ps(ymm0,ymm1);
		ymm2 = _mm256_permutevar8x32_ps(ymm2,ymm15);

		// Store result
    	_mm256_storeu_ps(out0,ymm2);
	}
	for(uint64_t i=(N-16*howmany);i<N;i+=2){out[i/2] = in[i]*in[i]+in[i+1]*in[i+1];}
}

template<>
void aCorrFreqAVX<double>(uint64_t N, double* in, double* out)
{
	__m256d ymm0, ymm1, ymm2;
	double *out0;

	uint64_t howmany = N/8;
	uint64_t j=0;
	
	for(uint64_t i=0;i<howmany;i++)
	{
		j = 8*i;
		out0 = out+(j/2);

		// Acorr
		ymm0 = _mm256_loadu_pd(in+j);
		ymm1 = _mm256_loadu_pd(in+j+4);

		ymm0 = _mm256_mul_pd(ymm0,ymm0);
		ymm1 = _mm256_mul_pd(ymm1,ymm1);

		ymm2 = _mm256_hadd_pd(ymm0,ymm1);
		ymm2 = _mm256_permute4x64_pd(ymm2,0b11011000);

		// Store result
    	_mm256_storeu_pd(out0,ymm2);
	}
	for(uint64_t i=(N-8*howmany);i<N;i+=2){out[i/2] = in[i]*in[i]+in[i+1]*in[i+1];}
}

///////////////////////////////////////////////////////////////////
//                      __  ______                               //
//                      \ \/ / ___|___  _ __ _ __                //
//                       \  / |   / _ \| '__| '__|               //
//                       /  \ |__| (_) | |  | |                  //
//                      /_/\_\____\___/|_|  |_|                  //
///////////////////////////////////////////////////////////////////




///////////////////////////////////////////////////////////////////
//                       _____ ____                              //
//                      |  ___/ ___|___  _ __ _ __               //
//                      | |_ | |   / _ \| '__| '__|              //
//                      |  _|| |__| (_) | |  | |                 //
//                      |_|   \____\___/|_|  |_|                 //
///////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////
//               ___ _____ _   _ _____ ____  ____                //
//				/ _ \_   _| | | | ____|  _ \/ ___|               //
//			   | | | || | | |_| |  _| | |_) \___ \               //
//			   | |_| || | |  _  | |___|  _ < ___) |              //
//				\___/ |_| |_| |_|_____|_| \_\____/               //
///////////////////////////////////////////////////////////////////

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
	__m256 ymm0, ymm1;
	float* fymm0 = (float*) &ymm0;

	float* data=in; 
	float* result=out;

	uint64_t powers[16];
	for(int i=15;i>=0;i--){powers[i]=N>>(4*i);N^=(N>>(4*i)<<(4*i));}
	
	result[0] = std::accumulate(data,data+powers[0],0.0);
	
	uint64_t offset = powers[0];
	data = in+offset;

	for(uint8_t i=1;i<16;i++)
	{
		for(uint8_t j=0;j<i;j++)
		{
			for(uint64_t k=0;k<powers[i]*(1<<(4*(i-j)));k+=16)
			{
				ymm0 = _mm256_loadu_ps(data+k);
				ymm1 = _mm256_loadu_ps(data+k+8);
				ymm0 = _mm256_add_ps(ymm0,ymm1);
				ymm0 = _mm256_hadd_ps(ymm0,ymm0);
				ymm0 = _mm256_hadd_ps(ymm0,ymm0);
				result[i+k/16] = fymm0[0]+fymm0[4];
			}
			data = result+i;
		}
		result[i] = std::accumulate(result+i,result+i+powers[i],0.0);
		offset += powers[i]*(1<<(4*i));
		data = in+offset;
	}
	for(int i=1;i<16;i++){result[0]+=result[i];}
}

template<>
void reduceAVX<double>(uint64_t N, double* in, double* out)
{
	__m256d ymm0, ymm1;
	double* fymm0 = (double*) &ymm0;

	double* data=in; 
	double* result=out;

	uint64_t powers[22];
	for(int i=21;i>=0;i--){powers[i]=N>>(3*i);N^=(N>>(3*i)<<(3*i));}
	
	result[0] = std::accumulate(data,data+powers[0],0.0);
	
	uint64_t offset = powers[0];
	data = in+offset;

	for(uint8_t i=1;i<22;i++)
	{
		for(uint8_t j=0;j<i;j++)
		{
			for(uint64_t k=0;k<powers[i]*(1<<(3*(i-j)));k+=8)
			{
				ymm0 = _mm256_loadu_pd(data+k);
				ymm1 = _mm256_loadu_pd(data+k+4);
				ymm0 = _mm256_add_pd(ymm0,ymm1);
				ymm0 = _mm256_hadd_pd(ymm0,ymm0);
				result[i+k/8] = fymm0[0]+fymm0[2];
			}
			data = result+i;
		}
		result[i] = std::accumulate(result+i,result+i+powers[i],0.0);
		offset += powers[i]*(1<<(3*i));
		data = in+offset;
	}
	for(int i=1;i<22;i++){result[0]+=result[i];}
}
