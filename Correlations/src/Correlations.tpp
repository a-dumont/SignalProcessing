///////////////////////////////////////////////////////////////////
//                       _    ____                               //
//                      / \  / ___|___  _ __ _ __                //
//                     / _ \| |   / _ \| '__| '__|               //
//                    / ___ \ |__| (_) | |  | |                  //
//                   /_/   \_\____\___/|_|  |_|                  //
///////////////////////////////////////////////////////////////////
template<class DataType>
void aCorrCircFreqReduceAVX(uint64_t N, uint64_t size, DataType* data){}

template<>
void aCorrCircFreqReduceAVX<float>(uint64_t N, uint64_t size, float* data)
{
	__m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
	__m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
	uint64_t howmany = N/size;
	uint64_t howmany2 = howmany/16;
	uint64_t extras = howmany-16*howmany2;
	uint64_t Nregisters = size/8;
	uint64_t I = 0;
	uint64_t J = 0;
	float temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;
	
	if(howmany == 1){for(uint64_t j=0;j<size;j++){data[j] *= data[j];}return;}
	if(howmany < 16)
	{
		for(uint64_t i=0;i<size;i++)
		{
			data[i] *= data[i];
			for(uint64_t j=1;j<howmany;j++)
			{
				data[i] += data[i+j*size]*data[i+j*size]; 
			}
		}
		return;
	}

	for(uint64_t i=0;i<howmany2;i++)
	{
		I = i<<4;
		for(uint64_t j=0;j<Nregisters;j++)
		{
			J = j<<3;
			ymm0 = _mm256_loadu_ps(data+I*size+J);
			ymm1 = _mm256_loadu_ps(data+(I+1)*size+J);
			ymm2 = _mm256_loadu_ps(data+(I+2)*size+J);
			ymm3 = _mm256_loadu_ps(data+(I+3)*size+J);
			ymm4 = _mm256_loadu_ps(data+(I+4)*size+J);
			ymm5 = _mm256_loadu_ps(data+(I+5)*size+J);
			ymm6 = _mm256_loadu_ps(data+(I+6)*size+J);
			ymm7 = _mm256_loadu_ps(data+(I+7)*size+J);
			ymm8 = _mm256_loadu_ps(data+(I+8)*size+J);
			ymm9 = _mm256_loadu_ps(data+(I+9)*size+J);
			ymm10 = _mm256_loadu_ps(data+(I+10)*size+J);
			ymm11 = _mm256_loadu_ps(data+(I+11)*size+J);
			ymm12 = _mm256_loadu_ps(data+(I+12)*size+J);
			ymm13 = _mm256_loadu_ps(data+(I+13)*size+J);
			ymm14 = _mm256_loadu_ps(data+(I+14)*size+J);
			ymm15 = _mm256_loadu_ps(data+(I+15)*size+J);
		
			ymm0 = _mm256_mul_ps(ymm0,ymm0);
			ymm1 = _mm256_mul_ps(ymm1,ymm1);
			ymm2 = _mm256_mul_ps(ymm2,ymm2);
			ymm3 = _mm256_mul_ps(ymm3,ymm3);
			ymm4 = _mm256_mul_ps(ymm4,ymm4);
			ymm5 = _mm256_mul_ps(ymm5,ymm5);
			ymm6 = _mm256_mul_ps(ymm6,ymm6);
			ymm7 = _mm256_mul_ps(ymm7,ymm7);
			ymm8 = _mm256_mul_ps(ymm8,ymm8);
			ymm9 = _mm256_mul_ps(ymm9,ymm9);
			ymm10 = _mm256_mul_ps(ymm10,ymm10);
			ymm11 = _mm256_mul_ps(ymm11,ymm11);
			ymm12 = _mm256_mul_ps(ymm12,ymm12);
			ymm13 = _mm256_mul_ps(ymm13,ymm13);
			ymm14 = _mm256_mul_ps(ymm14,ymm14);
			ymm15 = _mm256_mul_ps(ymm15,ymm15);

			ymm0 = _mm256_add_ps(ymm0,ymm8);
			ymm1 = _mm256_add_ps(ymm1,ymm9);
			ymm2 = _mm256_add_ps(ymm2,ymm10);
			ymm3 = _mm256_add_ps(ymm3,ymm11);
			ymm4 = _mm256_add_ps(ymm4,ymm12);
			ymm5 = _mm256_add_ps(ymm5,ymm13);
			ymm6 = _mm256_add_ps(ymm6,ymm14);
			ymm7 = _mm256_add_ps(ymm7,ymm15);

			ymm0 = _mm256_add_ps(ymm0,ymm4);
			ymm1 = _mm256_add_ps(ymm1,ymm5);
			ymm2 = _mm256_add_ps(ymm2,ymm6);
			ymm3 = _mm256_add_ps(ymm3,ymm7);
			
			ymm0 = _mm256_add_ps(ymm0,ymm2);
			ymm1 = _mm256_add_ps(ymm1,ymm3);

			ymm0 = _mm256_add_ps(ymm0,ymm1);
			
			ymm2 = _mm256_set_ps(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
			for(uint64_t k=0;k<extras;k++)
			{
				ymm1 = _mm256_loadu_ps(data+(16*howmany2+k)*size+J);
				ymm1 = _mm256_mul_ps(ymm1,ymm1);
				ymm2 = _mm256_add_ps(ymm2,ymm1);	
			}

			ymm0 = _mm256_add_ps(ymm0,ymm2);

			_mm256_storeu_ps(data+i*size+J,ymm0);
		}
		for(uint64_t j=8*Nregisters;j<size;j++)
		{
			temp0 = data[I*size+j]*data[I*size+j];
			temp0 += data[(I+1)*size+j]*data[(I+1)*size+j];
			temp1 = data[(I+2)*size+j]*data[(I+2)*size+j];
			temp1 += data[(I+3)*size+j]*data[(I+3)*size+j];
			temp2 = data[(I+4)*size+j]*data[(I+4)*size+j];
			temp2 += data[(I+5)*size+j]*data[(I+5)*size+j];
			temp3 = data[(I+6)*size+j]*data[(I+6)*size+j];
			temp3 += data[(I+7)*size+j]*data[(I+7)*size+j];
			temp4 = data[(I+8)*size+j]*data[(I+8)*size+j];
			temp4 += data[(I+9)*size+j]*data[(I+9)*size+j];
			temp5 = data[(I+10)*size+j]*data[(I+10)*size+j];
			temp5 += data[(I+11)*size+j]*data[(I+11)*size+j];
			temp6 = data[(I+12)*size+j]*data[(I+12)*size+j];
			temp6 += data[(I+13)*size+j]*data[(I+13)*size+j];
			temp7 = data[(I+14)*size+j]*data[(I+14)*size+j];
			temp7 += data[(I+15)*size+j]*data[(I+15)*size+j];

			temp0 += temp4;
			temp1 += temp5;
			temp2 += temp6;
			temp3 += temp7;

			temp0 += temp2;
			temp1 += temp3;

			temp0 += temp1;

			temp2 = 0.0;
			for(uint64_t k=0;k<extras;k++)
			{
				temp1 = data[(16*howmany2+k)*size+j]*data[(16*howmany2+k)*size+j];
				temp2 += temp1;	
			}

			temp0 += temp2;

			data[i*size+j] = temp0;
		}
		extras = 0;
	}
}

template<>
void aCorrCircFreqReduceAVX<double>(uint64_t N, uint64_t size, double* data)
{
	__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
	__m256d ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
	uint64_t howmany = N/size;
	uint64_t howmany2 = howmany/16;
	uint64_t extras = howmany-16*howmany2;
	uint64_t Nregisters = size/4;
	uint64_t I = 0;
	uint64_t J = 0;
	double temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;
	
	if(howmany == 1){for(uint64_t j=0;j<size;j++){data[j] *= data[j];}return;}
	if(howmany < 16)
	{
		for(uint64_t i=0;i<size;i++)
		{
			data[i] *= data[i];
			for(uint64_t j=1;j<howmany;j++)
			{
				data[i] += data[i+j*size]*data[i+j*size]; 
			}
		}
		return;
	}

	for(uint64_t i=0;i<howmany2;i++)
	{
		I = i<<4;
		for(uint64_t j=0;j<Nregisters;j++)
		{
			J = j<<2;
			ymm0 = _mm256_loadu_pd(data+I*size+J);
			ymm1 = _mm256_loadu_pd(data+(I+1)*size+J);
			ymm2 = _mm256_loadu_pd(data+(I+2)*size+J);
			ymm3 = _mm256_loadu_pd(data+(I+3)*size+J);
			ymm4 = _mm256_loadu_pd(data+(I+4)*size+J);
			ymm5 = _mm256_loadu_pd(data+(I+5)*size+J);
			ymm6 = _mm256_loadu_pd(data+(I+6)*size+J);
			ymm7 = _mm256_loadu_pd(data+(I+7)*size+J);
			ymm8 = _mm256_loadu_pd(data+(I+8)*size+J);
			ymm9 = _mm256_loadu_pd(data+(I+9)*size+J);
			ymm10 = _mm256_loadu_pd(data+(I+10)*size+J);
			ymm11 = _mm256_loadu_pd(data+(I+11)*size+J);
			ymm12 = _mm256_loadu_pd(data+(I+12)*size+J);
			ymm13 = _mm256_loadu_pd(data+(I+13)*size+J);
			ymm14 = _mm256_loadu_pd(data+(I+14)*size+J);
			ymm15 = _mm256_loadu_pd(data+(I+15)*size+J);
		
			ymm0 = _mm256_mul_pd(ymm0,ymm0);
			ymm1 = _mm256_mul_pd(ymm1,ymm1);
			ymm2 = _mm256_mul_pd(ymm2,ymm2);
			ymm3 = _mm256_mul_pd(ymm3,ymm3);
			ymm4 = _mm256_mul_pd(ymm4,ymm4);
			ymm5 = _mm256_mul_pd(ymm5,ymm5);
			ymm6 = _mm256_mul_pd(ymm6,ymm6);
			ymm7 = _mm256_mul_pd(ymm7,ymm7);
			ymm8 = _mm256_mul_pd(ymm8,ymm8);
			ymm9 = _mm256_mul_pd(ymm9,ymm9);
			ymm10 = _mm256_mul_pd(ymm10,ymm10);
			ymm11 = _mm256_mul_pd(ymm11,ymm11);
			ymm12 = _mm256_mul_pd(ymm12,ymm12);
			ymm13 = _mm256_mul_pd(ymm13,ymm13);
			ymm14 = _mm256_mul_pd(ymm14,ymm14);
			ymm15 = _mm256_mul_pd(ymm15,ymm15);

			ymm0 = _mm256_add_pd(ymm0,ymm8);
			ymm1 = _mm256_add_pd(ymm1,ymm9);
			ymm2 = _mm256_add_pd(ymm2,ymm10);
			ymm3 = _mm256_add_pd(ymm3,ymm11);
			ymm4 = _mm256_add_pd(ymm4,ymm12);
			ymm5 = _mm256_add_pd(ymm5,ymm13);
			ymm6 = _mm256_add_pd(ymm6,ymm14);
			ymm7 = _mm256_add_pd(ymm7,ymm15);

			ymm0 = _mm256_add_pd(ymm0,ymm4);
			ymm1 = _mm256_add_pd(ymm1,ymm5);
			ymm2 = _mm256_add_pd(ymm2,ymm6);
			ymm3 = _mm256_add_pd(ymm3,ymm7);
			
			ymm0 = _mm256_add_pd(ymm0,ymm2);
			ymm1 = _mm256_add_pd(ymm1,ymm3);

			ymm0 = _mm256_add_pd(ymm0,ymm1);
			
			ymm2 = _mm256_set_pd(0.0,0.0,0.0,0.0);
			for(uint64_t k=0;k<extras;k++)
			{
				ymm1 = _mm256_loadu_pd(data+(16*howmany2+k)*size+J);
				ymm1 = _mm256_mul_pd(ymm1,ymm1);
				ymm2 = _mm256_add_pd(ymm2,ymm1);	
			}

			ymm0 = _mm256_add_pd(ymm0,ymm2);

			_mm256_storeu_pd(data+i*size+J,ymm0);
		}
		for(uint64_t j=4*Nregisters;j<size;j++)
		{
			temp0 = data[I*size+j]*data[I*size+j];
			temp0 += data[(I+1)*size+j]*data[(I+1)*size+j];
			temp1 = data[(I+2)*size+j]*data[(I+2)*size+j];
			temp1 += data[(I+3)*size+j]*data[(I+3)*size+j];
			temp2 = data[(I+4)*size+j]*data[(I+4)*size+j];
			temp2 += data[(I+5)*size+j]*data[(I+5)*size+j];
			temp3 = data[(I+6)*size+j]*data[(I+6)*size+j];
			temp3 += data[(I+7)*size+j]*data[(I+7)*size+j];
			temp4 = data[(I+8)*size+j]*data[(I+8)*size+j];
			temp4 += data[(I+9)*size+j]*data[(I+9)*size+j];
			temp5 = data[(I+10)*size+j]*data[(I+10)*size+j];
			temp5 += data[(I+11)*size+j]*data[(I+11)*size+j];
			temp6 = data[(I+12)*size+j]*data[(I+12)*size+j];
			temp6 += data[(I+13)*size+j]*data[(I+13)*size+j];
			temp7 = data[(I+14)*size+j]*data[(I+14)*size+j];
			temp7 += data[(I+15)*size+j]*data[(I+15)*size+j];

			temp0 += temp4;
			temp1 += temp5;
			temp2 += temp6;
			temp3 += temp7;

			temp0 += temp2;
			temp1 += temp3;

			temp0 += temp1;

			temp2 = 0.0;
			for(uint64_t k=0;k<extras;k++)
			{
				temp1 = data[(16*howmany2+k)*size+j]*data[(16*howmany2+k)*size+j];
				temp2 += temp1;	
			}

			temp0 += temp2;

			data[i*size+j] = temp0;
		}
		extras = 0;
	}
}

///////////////////////////////////////////////////////////////////
//                      __  ______                               //
//                      \ \/ / ___|___  _ __ _ __                //
//                       \  / |   / _ \| '__| '__|               //
//                       /  \ |__| (_) | |  | |                  //
//                      /_/\_\____\___/|_|  |_|                  //
///////////////////////////////////////////////////////////////////
template<class DataType>
void xCorrCircFreqReduceAVX(uint64_t N, uint64_t size, DataType* data1, DataType* data2){}

template<>
void xCorrCircFreqReduceAVX<float>(uint64_t N, uint64_t size, float* data1, float* data2)
{
	__m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
	__m256 ymm12, ymm13, ymm14, ymm15;
	uint64_t howmany = N/size;
	uint64_t howmany2 = howmany/16;
	uint64_t extras = howmany-16*howmany2;
	uint64_t Nregisters = size/8;
	uint64_t cSize = size/2;
	uint64_t I = 0;
	uint64_t J = 0;
	float temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;
	
	if(howmany == 1)
	{
		for(uint64_t j=0;j<cSize;j++)
		{
			J = j<<1;
			temp0 = data1[J]*data2[J]+data1[J+1]*data2[J+1];
			temp1 = data1[J+1]*data2[J]-data1[J]*data2[J+1];
			data1[J] = temp0;
			data1[J+1] = temp1;
		}
		return;
	}
	if(howmany < 16)
	{
		for(uint64_t i=0;i<cSize;i++)
		{
			I = i<<1;
			temp0 = data1[I]*data2[I]+data1[I+1]*data2[I+1];
			temp1 = data1[I+1]*data2[I]-data1[I]*data2[I+1];
			data1[I] = temp0;
			data1[I+1] = temp1;
			for(uint64_t j=1;j<howmany;j++)
			{
				J = I+j*size;
				temp0 = data1[J]*data2[J]+data1[J+1]*data2[J+1];
				temp1 = data1[J+1]*data2[J]-data1[J]*data2[J+1];
				data1[I] += temp0; 
				data1[I+1] += temp1; 
			}
		}
		return;
	}
	
	ymm15 = _mm256_set_ps(-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0);
	
	for(uint64_t i=0;i<howmany2;i++)
	{
		I = i<<4;
		for(uint64_t j=0;j<Nregisters;j++)
		{
			J = j<<3;
			ymm12 = _mm256_loadu_ps(data1+I*size+J);
			ymm13 = _mm256_loadu_ps(data2+I*size+J);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
	
			ymm0 = _mm256_permute_ps(ymm12,0b11011000);

			ymm12 = _mm256_loadu_ps(data1+(I+1)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+1)*size+J);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
			ymm1 = _mm256_permute_ps(ymm12,0b11011000);
			
			ymm0 = _mm256_add_ps(ymm0,ymm1);

			ymm12 = _mm256_loadu_ps(data1+(I+2)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+2)*size+J);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
	
			ymm1 = _mm256_permute_ps(ymm12,0b11011000);

			ymm12 = _mm256_loadu_ps(data1+(I+3)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+3)*size+J);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
			ymm2 = _mm256_permute_ps(ymm12,0b11011000);
			
			ymm1 = _mm256_add_ps(ymm1,ymm2);
			ymm0 = _mm256_add_ps(ymm0,ymm1);

			ymm12 = _mm256_loadu_ps(data1+(I+4)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+4)*size+J);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
	
			ymm1 = _mm256_permute_ps(ymm12,0b11011000);

			ymm12 = _mm256_loadu_ps(data1+(I+5)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+5)*size+J);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
			ymm2 = _mm256_permute_ps(ymm12,0b11011000);
			
			ymm1 = _mm256_add_ps(ymm1,ymm2);

			ymm12 = _mm256_loadu_ps(data1+(I+6)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+6)*size+J);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
	
			ymm2 = _mm256_permute_ps(ymm12,0b11011000);

			ymm12 = _mm256_loadu_ps(data1+(I+7)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+7)*size+J);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
			ymm3 = _mm256_permute_ps(ymm12,0b11011000);
			
			ymm2 = _mm256_add_ps(ymm2,ymm3);
			ymm1 = _mm256_add_ps(ymm1,ymm2);
			ymm0 = _mm256_add_ps(ymm0,ymm1);
			//
			ymm12 = _mm256_loadu_ps(data1+(I+8)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+8)*size+J);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
	
			ymm4 = _mm256_permute_ps(ymm12,0b11011000);

			ymm12 = _mm256_loadu_ps(data1+(I+9)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+9)*size+J);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
			ymm5 = _mm256_permute_ps(ymm12,0b11011000);
			
			ymm4 = _mm256_add_ps(ymm4,ymm5);

			ymm12 = _mm256_loadu_ps(data1+(I+10)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+10)*size+J);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
	
			ymm5 = _mm256_permute_ps(ymm12,0b11011000);

			ymm12 = _mm256_loadu_ps(data1+(I+11)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+11)*size+J);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
			ymm6 = _mm256_permute_ps(ymm12,0b11011000);
			
			ymm5 = _mm256_add_ps(ymm5,ymm6);
			ymm4 = _mm256_add_ps(ymm4,ymm5);

			ymm12 = _mm256_loadu_ps(data1+(I+12)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+12)*size+J);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
	
			ymm5 = _mm256_permute_ps(ymm12,0b11011000);

			ymm12 = _mm256_loadu_ps(data1+(I+13)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+13)*size+J);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
			ymm6 = _mm256_permute_ps(ymm12,0b11011000);
			
			ymm5 = _mm256_add_ps(ymm5,ymm6);

			ymm12 = _mm256_loadu_ps(data1+(I+14)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+14)*size+J);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
	
			ymm6 = _mm256_permute_ps(ymm12,0b11011000);

			ymm12 = _mm256_loadu_ps(data1+(I+15)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+15)*size+J);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
			ymm7 = _mm256_permute_ps(ymm12,0b11011000);
			
			ymm6 = _mm256_add_ps(ymm6,ymm7);
			ymm5 = _mm256_add_ps(ymm5,ymm6);
			ymm4 = _mm256_add_ps(ymm4,ymm5);
			ymm0 = _mm256_add_ps(ymm0,ymm4);
			
			ymm1 = _mm256_set_ps(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
			for(uint64_t k=0;k<extras;k++)
			{
				ymm12 = _mm256_loadu_ps(data1+(16*howmany2+k)*size+J);
				ymm13 = _mm256_loadu_ps(data2+(16*howmany2+k)*size+J);
				ymm14 = _mm256_mul_ps(ymm12,ymm13);
				ymm13 = _mm256_mul_ps(ymm13,ymm15);
				ymm13 = _mm256_permute_ps(ymm13,0b10110001);
				ymm13 = _mm256_mul_ps(ymm12,ymm13);
				ymm12 = _mm256_hadd_ps(ymm14,ymm13);
				ymm7 = _mm256_permute_ps(ymm12,0b11011000);
				ymm1 = _mm256_add_ps(ymm1,ymm7);	
			}

			ymm0 = _mm256_add_ps(ymm0,ymm1);

			_mm256_storeu_ps(data1+i*size+J,ymm0);
		}
		for(uint64_t j=8*Nregisters;j<size;j+=2)
		{
			temp0 = data1[I*size+j]*data2[I*size+j]+data1[I*size+j+1]*data2[I*size+j+1];
			temp0 += data1[(I+1)*size+j]*data2[(I+1)*size+j];
			temp0 += data1[(I+1)*size+j+1]*data2[(I+1)*size+j+1];
			
			temp1 = data1[I*size+j+1]*data2[I*size+j]-data1[I*size+j]*data2[I*size+j+1];
			temp1 += data1[(I+1)*size+j+1]*data2[(I+1)*size+j];
			temp1 -= data1[(I+1)*size+j]*data2[(I+1)*size+j+1];

			temp2 = data1[(I+2)*size+j]*data2[(I+2)*size+j];
			temp2 += data1[(I+2)*size+j+1]*data2[(I+2)*size+j+1];
			temp2 += data1[(I+3)*size+j]*data2[(I+3)*size+j];
			temp2 += data1[(I+3)*size+j+1]*data2[(I+3)*size+j+1];
			
			temp3 = data1[(I+2)*size+j+1]*data2[(I+2)*size+j];
			temp3 -= data1[(I+2)*size+j]*data2[(I+2)*size+j+1];
			temp3 += data1[(I+3)*size+j+1]*data2[(I+3)*size+j];
			temp3 -= data1[(I+3)*size+j]*data2[(I+3)*size+j+1];

			temp4 = data1[(I+4)*size+j]*data2[(I+4)*size+j];
			temp4 += data1[(I+4)*size+j+1]*data2[(I+4)*size+j+1];
			temp4 += data1[(I+5)*size+j]*data2[(I+5)*size+j];
			temp4 += data1[(I+5)*size+j+1]*data2[(I+5)*size+j+1];
			
			temp5 = data1[(I+4)*size+j+1]*data2[(I+4)*size+j];
			temp5 -= data1[(I+4)*size+j]*data2[(I+4)*size+j+1];
			temp5 += data1[(I+5)*size+j+1]*data2[(I+5)*size+j];
			temp5 -= data1[(I+5)*size+j]*data2[(I+5)*size+j+1];	
			
			temp6 = data1[(I+6)*size+j]*data2[(I+6)*size+j];
			temp6 += data1[(I+6)*size+j+1]*data2[(I+6)*size+j+1];
			temp6 += data1[(I+7)*size+j]*data2[(I+7)*size+j];
			temp6 += data1[(I+7)*size+j+1]*data2[(I+7)*size+j+1];
			
			temp7 = data1[(I+6)*size+j+1]*data2[(I+6)*size+j];
			temp7 -= data1[(I+6)*size+j]*data2[(I+6)*size+j+1];
			temp7 += data1[(I+7)*size+j+1]*data2[(I+7)*size+j];
			temp7 -= data1[(I+7)*size+j]*data2[(I+7)*size+j+1];	
			
			temp0 += temp4;
			temp1 += temp5;
			temp2 += temp6;
			temp3 += temp7;

			temp0 += temp2;
			temp1 += temp3;

			temp2 = data1[(I+8)*size+j]*data2[(I+8)*size+j];
			temp2 += data1[(I+8)*size+j+1]*data2[(I+8)*size+j+1];
			temp2 += data1[(I+9)*size+j]*data2[(I+9)*size+j];
			temp2 += data1[(I+9)*size+j+1]*data2[(I+9)*size+j+1];
			
			temp3 = data1[(I+8)*size+j+1]*data2[(I+8)*size+j];
			temp3 -= data1[(I+8)*size+j]*data2[(I+8)*size+j+1];
			temp3 += data1[(I+9)*size+j+1]*data2[(I+9)*size+j];
			temp3 -= data1[(I+9)*size+j]*data2[(I+9)*size+j+1];

			temp4 = data1[(I+10)*size+j]*data2[(I+10)*size+j];
			temp4 += data1[(I+10)*size+j+1]*data2[(I+10)*size+j+1];
			temp4 += data1[(I+11)*size+j]*data2[(I+11)*size+j];
			temp4 += data1[(I+11)*size+j+1]*data2[(I+11)*size+j+1];
			
			temp5 = data1[(I+10)*size+j+1]*data2[(I+10)*size+j];
			temp5 -= data1[(I+10)*size+j]*data2[(I+10)*size+j+1];
			temp5 += data1[(I+11)*size+j+1]*data2[(I+11)*size+j];
			temp5 -= data1[(I+11)*size+j]*data2[(I+11)*size+j+1];	
			
			temp6 = data1[(I+12)*size+j]*data2[(I+12)*size+j];
			temp6 += data1[(I+12)*size+j+1]*data2[(I+12)*size+j+1];
			temp6 += data1[(I+13)*size+j]*data2[(I+13)*size+j];
			temp6 += data1[(I+13)*size+j+1]*data2[(I+13)*size+j+1];
			
			temp7 = data1[(I+12)*size+j+1]*data2[(I+12)*size+j];
			temp7 -= data1[(I+12)*size+j]*data2[(I+12)*size+j+1];
			temp7 += data1[(I+13)*size+j+1]*data2[(I+13)*size+j];
			temp7 -= data1[(I+13)*size+j]*data2[(I+13)*size+j+1];	

			temp8 = data1[(I+14)*size+j]*data2[(I+14)*size+j];
			temp8 += data1[(I+14)*size+j+1]*data2[(I+14)*size+j+1];
			temp8 += data1[(I+15)*size+j]*data2[(I+15)*size+j];
			temp8 += data1[(I+15)*size+j+1]*data2[(I+15)*size+j+1];
			
			temp9 = data1[(I+14)*size+j+1]*data2[(I+14)*size+j];
			temp9 -= data1[(I+14)*size+j]*data2[(I+14)*size+j+1];
			temp9 += data1[(I+15)*size+j+1]*data2[(I+15)*size+j];
			temp9 -= data1[(I+15)*size+j]*data2[(I+15)*size+j+1];	
			
			temp2 += temp6;
			temp3 += temp7;
			temp4 += temp8;
			temp5 += temp9;

			temp2 += temp4;
			temp3 += temp5;

			temp0 += temp2;
			temp1 += temp3;

			temp2 = 0.0;
			temp3 = 0.0;
			for(uint64_t k=0;k<extras;k++)
			{
				temp4 = data1[(16*howmany2+k)*size+j]*data2[(16*howmany2+k)*size+j];
				temp4 += data1[(16*howmany2+k)*size+j+1]*data2[(16*howmany2+k)*size+j+1];
				temp5 = data1[(16*howmany2+k)*size+j+1]*data2[(16*howmany2+k)*size+j];
				temp5 -= data1[(16*howmany2+k)*size+j]*data2[(16*howmany2+k)*size+j+1];

				temp2 += temp4;
				temp3 += temp5;
			}

			temp0 += temp2;
			temp1 += temp3;
			
			data1[i*size+j] = temp0;
			data1[i*size+j+1] = temp1;
		}
		extras = 0;
	}
}

template<>
void xCorrCircFreqReduceAVX<double>(uint64_t N, uint64_t size, double* data1, double* data2)
{
	__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
	__m256d ymm12, ymm13, ymm14, ymm15;
	uint64_t howmany = N/size;
	uint64_t howmany2 = howmany/16;
	uint64_t extras = howmany-16*howmany2;
	uint64_t Nregisters = size/4;
	uint64_t cSize = size/2;
	uint64_t I = 0;
	uint64_t J = 0;
	double temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;
	
	if(howmany == 1)
	{
		for(uint64_t j=0;j<cSize;j++)
		{
			J = j<<1;
			temp0 = data1[J]*data2[J]+data1[J+1]*data2[J+1];
			temp1 = data1[J+1]*data2[J]-data1[J]*data2[J+1];
			data1[J] = temp0;
			data1[J+1] = temp1;
		}
		return;
	}
	if(howmany < 16)
	{
		for(uint64_t i=0;i<cSize;i++)
		{
			I = i<<1;
			temp0 = data1[I]*data2[I]+data1[I+1]*data2[I+1];
			temp1 = data1[I+1]*data2[I]-data1[I]*data2[I+1];
			data1[I] = temp0;
			data1[I+1] = temp1;
			for(uint64_t j=1;j<howmany;j++)
			{
				J = I+j*size;
				temp0 = data1[J]*data2[J]+data1[J+1]*data2[J+1];
				temp1 = data1[J+1]*data2[J]-data1[J]*data2[J+1];
				data1[I] += temp0; 
				data1[I+1] += temp1; 
			}
		}
		return;
	}
	
	ymm15 = _mm256_set_pd(-1.0,1.0,-1.0,1.0);
	
	for(uint64_t i=0;i<howmany2;i++)
	{
		I = i<<4;
		for(uint64_t j=0;j<Nregisters;j++)
		{
			J = j<<2;
			ymm12 = _mm256_loadu_pd(data1+I*size+J);
			ymm13 = _mm256_loadu_pd(data2+I*size+J);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);	
			
			ymm0 = _mm256_hadd_pd(ymm14,ymm13);
	
			ymm12 = _mm256_loadu_pd(data1+(I+1)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+1)*size+J);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm1 = _mm256_hadd_pd(ymm14,ymm13);
			ymm0 = _mm256_add_pd(ymm0,ymm1);

			ymm12 = _mm256_loadu_pd(data1+(I+2)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+2)*size+J);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);

			ymm1 = _mm256_hadd_pd(ymm14,ymm13);
	
			ymm12 = _mm256_loadu_pd(data1+(I+3)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+3)*size+J);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm2 = _mm256_hadd_pd(ymm14,ymm13);	
			ymm1 = _mm256_add_pd(ymm1,ymm2);
			ymm0 = _mm256_add_pd(ymm0,ymm1);

			ymm12 = _mm256_loadu_pd(data1+(I+4)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+4)*size+J);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm1 = _mm256_hadd_pd(ymm14,ymm13);
	
			ymm12 = _mm256_loadu_pd(data1+(I+5)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+5)*size+J);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
				
			ymm2 = _mm256_hadd_pd(ymm14,ymm13);
			ymm1 = _mm256_add_pd(ymm1,ymm2);

			ymm12 = _mm256_loadu_pd(data1+(I+6)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+6)*size+J);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm2 = _mm256_hadd_pd(ymm14,ymm13);
	
			ymm12 = _mm256_loadu_pd(data1+(I+7)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+7)*size+J);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm3 = _mm256_hadd_pd(ymm14,ymm13);
			ymm2 = _mm256_add_pd(ymm2,ymm3);
			ymm1 = _mm256_add_pd(ymm1,ymm2);
			ymm0 = _mm256_add_pd(ymm0,ymm1);
			
			ymm12 = _mm256_loadu_pd(data1+(I+8)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+8)*size+J);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm4 = _mm256_hadd_pd(ymm14,ymm13);
	
			ymm12 = _mm256_loadu_pd(data1+(I+9)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+9)*size+J);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm5 = _mm256_hadd_pd(ymm14,ymm13);
			ymm4 = _mm256_add_pd(ymm4,ymm5);

			ymm12 = _mm256_loadu_pd(data1+(I+10)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+10)*size+J);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm5 = _mm256_hadd_pd(ymm14,ymm13);
	
			ymm12 = _mm256_loadu_pd(data1+(I+11)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+11)*size+J);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm6 = _mm256_hadd_pd(ymm14,ymm13);
			ymm5 = _mm256_add_pd(ymm5,ymm6);
			ymm4 = _mm256_add_pd(ymm4,ymm5);

			ymm12 = _mm256_loadu_pd(data1+(I+12)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+12)*size+J);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm5 = _mm256_hadd_pd(ymm14,ymm13);
	
			ymm12 = _mm256_loadu_pd(data1+(I+13)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+13)*size+J);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm6 = _mm256_hadd_pd(ymm14,ymm13);
			ymm5 = _mm256_add_pd(ymm5,ymm6);

			ymm12 = _mm256_loadu_pd(data1+(I+14)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+14)*size+J);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm6 = _mm256_hadd_pd(ymm14,ymm13);

			ymm12 = _mm256_loadu_pd(data1+(I+15)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+15)*size+J);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
	
			ymm7 = _mm256_hadd_pd(ymm14,ymm13);
			ymm6 = _mm256_add_pd(ymm6,ymm7);
			ymm5 = _mm256_add_pd(ymm5,ymm6);
			ymm4 = _mm256_add_pd(ymm4,ymm5);
			ymm0 = _mm256_add_pd(ymm0,ymm4);
			
			ymm1 = _mm256_set_pd(0.0,0.0,0.0,0.0);
			for(uint64_t k=0;k<extras;k++)
			{
				ymm12 = _mm256_loadu_pd(data1+(16*howmany2+k)*size+J);
				ymm13 = _mm256_loadu_pd(data2+(16*howmany2+k)*size+J);
				ymm14 = _mm256_mul_pd(ymm12,ymm13);
				ymm13 = _mm256_mul_pd(ymm13,ymm15);
				ymm13 = _mm256_permute_pd(ymm13,0b00000101);
				ymm13 = _mm256_mul_pd(ymm12,ymm13);
				
				ymm7 = _mm256_hadd_pd(ymm14,ymm13);
				ymm1 = _mm256_add_pd(ymm1,ymm7);	
			}

			ymm0 = _mm256_add_pd(ymm0,ymm1);

			_mm256_storeu_pd(data1+i*size+J,ymm0);
		}
		for(uint64_t j=4*Nregisters;j<size;j+=2)
		{
			temp0 = data1[I*size+j]*data2[I*size+j]+data1[I*size+j+1]*data2[I*size+j+1];
			temp0 += data1[(I+1)*size+j]*data2[(I+1)*size+j];
			temp0 += data1[(I+1)*size+j+1]*data2[(I+1)*size+j+1];
			
			temp1 = data1[I*size+j+1]*data2[I*size+j]-data1[I*size+j]*data2[I*size+j+1];
			temp1 += data1[(I+1)*size+j+1]*data2[(I+1)*size+j];
			temp1 -= data1[(I+1)*size+j]*data2[(I+1)*size+j+1];

			temp2 = data1[(I+2)*size+j]*data2[(I+2)*size+j];
			temp2 += data1[(I+2)*size+j+1]*data2[(I+2)*size+j+1];
			temp2 += data1[(I+3)*size+j]*data2[(I+3)*size+j];
			temp2 += data1[(I+3)*size+j+1]*data2[(I+3)*size+j+1];
			
			temp3 = data1[(I+2)*size+j+1]*data2[(I+2)*size+j];
			temp3 -= data1[(I+2)*size+j]*data2[(I+2)*size+j+1];
			temp3 += data1[(I+3)*size+j+1]*data2[(I+3)*size+j];
			temp3 -= data1[(I+3)*size+j]*data2[(I+3)*size+j+1];

			temp4 = data1[(I+4)*size+j]*data2[(I+4)*size+j];
			temp4 += data1[(I+4)*size+j+1]*data2[(I+4)*size+j+1];
			temp4 += data1[(I+5)*size+j]*data2[(I+5)*size+j];
			temp4 += data1[(I+5)*size+j+1]*data2[(I+5)*size+j+1];
			
			temp5 = data1[(I+4)*size+j+1]*data2[(I+4)*size+j];
			temp5 -= data1[(I+4)*size+j]*data2[(I+4)*size+j+1];
			temp5 += data1[(I+5)*size+j+1]*data2[(I+5)*size+j];
			temp5 -= data1[(I+5)*size+j]*data2[(I+5)*size+j+1];	
			
			temp6 = data1[(I+6)*size+j]*data2[(I+6)*size+j];
			temp6 += data1[(I+6)*size+j+1]*data2[(I+6)*size+j+1];
			temp6 += data1[(I+7)*size+j]*data2[(I+7)*size+j];
			temp6 += data1[(I+7)*size+j+1]*data2[(I+7)*size+j+1];
			
			temp7 = data1[(I+6)*size+j+1]*data2[(I+6)*size+j];
			temp7 -= data1[(I+6)*size+j]*data2[(I+6)*size+j+1];
			temp7 += data1[(I+7)*size+j+1]*data2[(I+7)*size+j];
			temp7 -= data1[(I+7)*size+j]*data2[(I+7)*size+j+1];	
			
			temp0 += temp4;
			temp1 += temp5;
			temp2 += temp6;
			temp3 += temp7;

			temp0 += temp2;
			temp1 += temp3;

			temp2 = data1[(I+8)*size+j]*data2[(I+8)*size+j];
			temp2 += data1[(I+8)*size+j+1]*data2[(I+8)*size+j+1];
			temp2 += data1[(I+9)*size+j]*data2[(I+9)*size+j];
			temp2 += data1[(I+9)*size+j+1]*data2[(I+9)*size+j+1];
			
			temp3 = data1[(I+8)*size+j+1]*data2[(I+8)*size+j];
			temp3 -= data1[(I+8)*size+j]*data2[(I+8)*size+j+1];
			temp3 += data1[(I+9)*size+j+1]*data2[(I+9)*size+j];
			temp3 -= data1[(I+9)*size+j]*data2[(I+9)*size+j+1];

			temp4 = data1[(I+10)*size+j]*data2[(I+10)*size+j];
			temp4 += data1[(I+10)*size+j+1]*data2[(I+10)*size+j+1];
			temp4 += data1[(I+11)*size+j]*data2[(I+11)*size+j];
			temp4 += data1[(I+11)*size+j+1]*data2[(I+11)*size+j+1];
			
			temp5 = data1[(I+10)*size+j+1]*data2[(I+10)*size+j];
			temp5 -= data1[(I+10)*size+j]*data2[(I+10)*size+j+1];
			temp5 += data1[(I+11)*size+j+1]*data2[(I+11)*size+j];
			temp5 -= data1[(I+11)*size+j]*data2[(I+11)*size+j+1];	
			
			temp6 = data1[(I+12)*size+j]*data2[(I+12)*size+j];
			temp6 += data1[(I+12)*size+j+1]*data2[(I+12)*size+j+1];
			temp6 += data1[(I+13)*size+j]*data2[(I+13)*size+j];
			temp6 += data1[(I+13)*size+j+1]*data2[(I+13)*size+j+1];
			
			temp7 = data1[(I+12)*size+j+1]*data2[(I+12)*size+j];
			temp7 -= data1[(I+12)*size+j]*data2[(I+12)*size+j+1];
			temp7 += data1[(I+13)*size+j+1]*data2[(I+13)*size+j];
			temp7 -= data1[(I+13)*size+j]*data2[(I+13)*size+j+1];	

			temp8 = data1[(I+14)*size+j]*data2[(I+14)*size+j];
			temp8 += data1[(I+14)*size+j+1]*data2[(I+14)*size+j+1];
			temp8 += data1[(I+15)*size+j]*data2[(I+15)*size+j];
			temp8 += data1[(I+15)*size+j+1]*data2[(I+15)*size+j+1];
			
			temp9 = data1[(I+14)*size+j+1]*data2[(I+14)*size+j];
			temp9 -= data1[(I+14)*size+j]*data2[(I+14)*size+j+1];
			temp9 += data1[(I+15)*size+j+1]*data2[(I+15)*size+j];
			temp9 -= data1[(I+15)*size+j]*data2[(I+15)*size+j+1];	
			
			temp2 += temp6;
			temp3 += temp7;
			temp4 += temp8;
			temp5 += temp9;

			temp2 += temp4;
			temp3 += temp5;

			temp0 += temp2;
			temp1 += temp3;

			temp2 = 0.0;
			temp3 = 0.0;
			for(uint64_t k=0;k<extras;k++)
			{
				temp4 = data1[(16*howmany2+k)*size+j]*data2[(16*howmany2+k)*size+j];
				temp4 += data1[(16*howmany2+k)*size+j+1]*data2[(16*howmany2+k)*size+j+1];
				temp5 = data1[(16*howmany2+k)*size+j+1]*data2[(16*howmany2+k)*size+j];
				temp5 -= data1[(16*howmany2+k)*size+j]*data2[(16*howmany2+k)*size+j+1];

				temp2 += temp4;
				temp3 += temp5;
			}

			temp0 += temp2;
			temp1 += temp3;
			
			data1[i*size+j] = temp0;
			data1[i*size+j+1] = temp1;
		}
		extras = 0;
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
void fCorrCircFreqReduceAVX(uint64_t N, uint64_t size, DataType* data1, DataType* data2){}

template<>
void fCorrCircFreqReduceAVX<float>(uint64_t N, uint64_t size, float* data1, float* data2)
{
	__m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
	__m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
	uint64_t howmany = N/size;
	uint64_t howmany2 = howmany/16;
	uint64_t extras = howmany-16*howmany2;
	uint64_t Nregisters = size/8;
	uint64_t cSize = size/2;
	uint64_t I = 0;
	uint64_t J = 0;
	uint64_t K = 0;
	uint64_t L = 0;
	float temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;
	float temp10, temp11, temp12, temp13, temp14, temp15, temp16, temp17, temp18, temp19;
	float temp20, temp21;
	
	if(howmany == 1)
	{
		for(uint64_t j=0;j<(cSize-1);j+=2)
		{
			J = j<<1;
			K = J-(j%2);
			L = K+2-((K+2)/(2*cSize));
			temp0 = data1[J]*data2[J]+data1[J+1]*data2[J+1];
			temp1 = data1[J+1]*data2[J]-data1[J]*data2[J+1];
			temp2 = data1[J+2]*data2[J+2]+data1[J+3]*data2[J+3];
			temp3 = data1[J+3]*data2[J+2]-data1[J+2]*data2[J+3];
			temp4 = data1[J]*data1[J]+data1[J+1]*data1[J+1];
			temp5 = data1[J+2]*data1[J+2]+data1[J+3]*data1[J+3];
			temp6 = data2[J]*data2[J]+data2[J+1]*data2[J+1];
			temp7 = data2[J+2]*data2[J+2]+data2[J+3]*data2[J+3];
			data1[K] = temp4;
			data1[K+1] = temp5;
			data1[L] = temp6;
			data1[L+1] = temp7;
			data2[J] = temp0;
			data2[J+1] = temp1;
			data2[J+2] = temp2;
			data2[J+3] = temp3;
		}
		J = (cSize-1)<<1;
		K = J-((cSize-1)%2);
		L = K+2-((K+2)/(2*cSize));
		temp0 = data1[J]*data2[J]+data1[J+1]*data2[J+1];
		temp1 = data1[J+1]*data2[J]-data1[J]*data2[J+1];
		temp4 = data1[J]*data1[J]+data1[J+1]*data1[J+1];
		temp6 = data2[J]*data2[J]+data2[J+1]*data2[J+1];
		data1[K] = temp4;
		data1[L] = temp6;
		data2[J] = temp0;
		data2[J+1] = temp1;
		return;
	}
	if(howmany < 16)
	{
		for(uint64_t i=0;i<(cSize-1);i+=2)
		{
			I = i<<1;
			K = I-(i%2);
			L = K+2-((K+2)/(2*cSize));
			temp0 = data1[I]*data2[I]+data1[I+1]*data2[I+1];
			temp1 = data1[I+1]*data2[I]-data1[I]*data2[I+1];
			temp2 = data1[I+2]*data2[I+2]+data1[I+3]*data2[I+3];
			temp3 = data1[I+3]*data2[I+2]-data1[I+2]*data2[I+3];
			temp4 = data1[I]*data1[I]+data1[I+1]*data1[I+1];
			temp5 = data1[I+2]*data1[I+2]+data1[I+3]*data1[I+3];
			temp6 = data2[I]*data2[I]+data2[I+1]*data2[I+1];
			temp7 = data2[I+2]*data2[I+2]+data2[I+3]*data2[I+3];
			data1[K] = temp4;
			data1[K+1] = temp5;
			data1[L] = temp6;
			data1[L+1] = temp7;
			data2[I] = temp0;
			data2[I+1] = temp1;
			data2[I+2] = temp2;
			data2[I+3] = temp3;
			for(uint64_t j=1;j<howmany;j++)
			{
				J = I+j*size;
				temp0 = data1[J]*data2[J]+data1[J+1]*data2[J+1];
				temp1 = data1[J+1]*data2[J]-data1[J]*data2[J+1];
				temp2 = data1[J+2]*data2[J+2]+data1[J+3]*data2[J+3];
				temp3 = data1[J+3]*data2[J+2]-data1[J+2]*data2[J+3];
				temp4 = data1[J]*data1[J]+data1[J+1]*data1[J+1];
				temp5 = data1[J+2]*data1[J+2]+data1[J+3]*data1[J+3];
				temp6 = data2[J]*data2[J]+data2[J+1]*data2[J+1];
				temp7 = data2[J+2]*data2[J+2]+data2[J+3]*data2[J+3];
				data1[K] += temp4;
				data1[K+1] += temp5;
				data1[L] += temp6;
				data1[L+1] += temp7;
				data2[I] += temp0;
				data2[I+1] += temp1;
				data2[I+2] += temp2;
				data2[I+3] += temp3;
			}

		}
		I = (cSize-1)<<1;
		K = I-((cSize-1)%2);
		L = K+2-((K+2)/(2*cSize));
		temp0 = data1[I]*data2[I]+data1[I+1]*data2[I+1];
		temp1 = data1[I+1]*data2[I]-data1[I]*data2[I+1];
		temp4 = data1[I]*data1[I]+data1[I+1]*data1[I+1];
		temp6 = data2[I]*data2[I]+data2[I+1]*data2[I+1];
		data1[K] = temp4;
		data1[L] = temp6;
		data2[I] = temp0;
		data2[I+1] = temp1;
		for(uint64_t j=1;j<howmany;j++)
		{
			J = I+j*size;
			temp0 = data1[J]*data2[J]+data1[J+1]*data2[J+1];
			temp1 = data1[J+1]*data2[J]-data1[J]*data2[J+1];
			temp4 = data1[J]*data1[J]+data1[J+1]*data1[J+1];
			temp6 = data2[J]*data2[J]+data2[J+1]*data2[J+1];
			data1[K] += temp4;
			data1[L] += temp6;
			data2[I] += temp0;
			data2[I+1] += temp1;
		}
		return;
	}
	
	ymm15 = _mm256_set_ps(-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0);
	
	for(uint64_t i=0;i<howmany2;i++)
	{
		I = i<<4;
		for(uint64_t j=0;j<Nregisters;j++)
		{
			J = j<<3;
			ymm12 = _mm256_loadu_ps(data1+I*size+J);
			ymm13 = _mm256_loadu_ps(data2+I*size+J);
			ymm10 = _mm256_mul_ps(ymm12,ymm12);
			ymm11 = _mm256_mul_ps(ymm13,ymm13);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
	
			ymm8 = _mm256_hadd_ps(ymm10,ymm11);
			ymm0 = _mm256_permute_ps(ymm12,0b11011000);

			ymm12 = _mm256_loadu_ps(data1+(I+1)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+1)*size+J);
			ymm10 = _mm256_mul_ps(ymm12,ymm12);
			ymm11 = _mm256_mul_ps(ymm13,ymm13);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
			ymm1 = _mm256_permute_ps(ymm12,0b11011000);
			
			ymm10 = _mm256_hadd_ps(ymm10,ymm11);
			ymm8 = _mm256_add_ps(ymm8,ymm10);
			ymm0 = _mm256_add_ps(ymm0,ymm1);

			ymm12 = _mm256_loadu_ps(data1+(I+2)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+2)*size+J);
			ymm10 = _mm256_mul_ps(ymm12,ymm12);
			ymm11 = _mm256_mul_ps(ymm13,ymm13);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
	
			ymm9 = _mm256_hadd_ps(ymm10,ymm11);
			ymm1 = _mm256_permute_ps(ymm12,0b11011000);

			ymm12 = _mm256_loadu_ps(data1+(I+3)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+3)*size+J);
			ymm10 = _mm256_mul_ps(ymm12,ymm12);
			ymm11 = _mm256_mul_ps(ymm13,ymm13);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
			ymm2 = _mm256_permute_ps(ymm12,0b11011000);
			
			ymm10 = _mm256_hadd_ps(ymm10,ymm11);
			ymm9 = _mm256_add_ps(ymm9,ymm10);
			ymm8 = _mm256_add_ps(ymm8,ymm9);
			ymm1 = _mm256_add_ps(ymm1,ymm2);
			ymm0 = _mm256_add_ps(ymm0,ymm1);

			ymm12 = _mm256_loadu_ps(data1+(I+4)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+4)*size+J);
			ymm10 = _mm256_mul_ps(ymm12,ymm12);
			ymm11 = _mm256_mul_ps(ymm13,ymm13);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
	
			ymm7 = _mm256_hadd_ps(ymm10,ymm11);
			ymm1 = _mm256_permute_ps(ymm12,0b11011000);

			ymm12 = _mm256_loadu_ps(data1+(I+5)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+5)*size+J);
			ymm10 = _mm256_mul_ps(ymm12,ymm12);
			ymm11 = _mm256_mul_ps(ymm13,ymm13);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
			ymm2 = _mm256_permute_ps(ymm12,0b11011000);
			
			ymm6 = _mm256_hadd_ps(ymm10,ymm11);
			ymm9 = _mm256_add_ps(ymm6,ymm7);
			ymm1 = _mm256_add_ps(ymm1,ymm2);

			ymm12 = _mm256_loadu_ps(data1+(I+6)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+6)*size+J);
			ymm10 = _mm256_mul_ps(ymm12,ymm12);
			ymm11 = _mm256_mul_ps(ymm13,ymm13);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
	
			ymm7 = _mm256_hadd_ps(ymm10,ymm11);
			ymm2 = _mm256_permute_ps(ymm12,0b11011000);

			ymm12 = _mm256_loadu_ps(data1+(I+7)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+7)*size+J);
			ymm10 = _mm256_mul_ps(ymm12,ymm12);
			ymm11 = _mm256_mul_ps(ymm13,ymm13);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
			ymm3 = _mm256_permute_ps(ymm12,0b11011000);
			
			ymm6 = _mm256_hadd_ps(ymm10,ymm11);
			ymm7 = _mm256_add_ps(ymm7,ymm6);
			ymm9 = _mm256_add_ps(ymm7,ymm9);
			ymm8 = _mm256_add_ps(ymm8,ymm9);
			ymm2 = _mm256_add_ps(ymm2,ymm3);
			ymm1 = _mm256_add_ps(ymm1,ymm2);
			ymm0 = _mm256_add_ps(ymm0,ymm1);
			//
			ymm12 = _mm256_loadu_ps(data1+(I+8)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+8)*size+J);
			ymm10 = _mm256_mul_ps(ymm12,ymm12);
			ymm11 = _mm256_mul_ps(ymm13,ymm13);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
	
			ymm7 = _mm256_hadd_ps(ymm10,ymm11);
			ymm4 = _mm256_permute_ps(ymm12,0b11011000);

			ymm12 = _mm256_loadu_ps(data1+(I+9)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+9)*size+J);
			ymm10 = _mm256_mul_ps(ymm12,ymm12);
			ymm11 = _mm256_mul_ps(ymm13,ymm13);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
			ymm5 = _mm256_permute_ps(ymm12,0b11011000);
			
			ymm6 = _mm256_hadd_ps(ymm10,ymm11);
			ymm4 = _mm256_add_ps(ymm4,ymm5);

			ymm12 = _mm256_loadu_ps(data1+(I+10)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+10)*size+J);
			ymm10 = _mm256_mul_ps(ymm12,ymm12);
			ymm11 = _mm256_mul_ps(ymm13,ymm13);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
	
			ymm10 = _mm256_hadd_ps(ymm10,ymm11);
			ymm6 = _mm256_add_ps(ymm6,ymm10);
			ymm5 = _mm256_permute_ps(ymm12,0b11011000);

			ymm12 = _mm256_loadu_ps(data1+(I+11)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+11)*size+J);
			ymm10 = _mm256_mul_ps(ymm12,ymm12);
			ymm11 = _mm256_mul_ps(ymm13,ymm13);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
			
			ymm10 = _mm256_hadd_ps(ymm10,ymm11);
			ymm7 = _mm256_add_ps(ymm7,ymm10);
			ymm9 = _mm256_add_ps(ymm6,ymm7);
			ymm6 = _mm256_permute_ps(ymm12,0b11011000);	
			ymm5 = _mm256_add_ps(ymm5,ymm6);
			ymm4 = _mm256_add_ps(ymm4,ymm5);

			ymm12 = _mm256_loadu_ps(data1+(I+12)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+12)*size+J);
			ymm10 = _mm256_mul_ps(ymm12,ymm12);
			ymm11 = _mm256_mul_ps(ymm13,ymm13);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
	
			ymm3 = _mm256_hadd_ps(ymm10,ymm11);
			ymm5 = _mm256_permute_ps(ymm12,0b11011000);

			ymm12 = _mm256_loadu_ps(data1+(I+13)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+13)*size+J);
			ymm10 = _mm256_mul_ps(ymm12,ymm12);
			ymm11 = _mm256_mul_ps(ymm13,ymm13);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
			ymm6 = _mm256_permute_ps(ymm12,0b11011000);
			
			ymm2 = _mm256_hadd_ps(ymm10,ymm11);
			ymm5 = _mm256_add_ps(ymm5,ymm6);

			ymm12 = _mm256_loadu_ps(data1+(I+14)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+14)*size+J);
			ymm10 = _mm256_mul_ps(ymm12,ymm12);
			ymm11 = _mm256_mul_ps(ymm13,ymm13);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);

			ymm10 = _mm256_hadd_ps(ymm10,ymm11);
			ymm3 = _mm256_add_ps(ymm3,ymm10);	
			ymm6 = _mm256_permute_ps(ymm12,0b11011000);

			ymm12 = _mm256_loadu_ps(data1+(I+15)*size+J);
			ymm13 = _mm256_loadu_ps(data2+(I+15)*size+J);
			ymm10 = _mm256_mul_ps(ymm12,ymm12);
			ymm11 = _mm256_mul_ps(ymm13,ymm13);
			ymm14 = _mm256_mul_ps(ymm12,ymm13);
			ymm13 = _mm256_mul_ps(ymm13,ymm15);
			ymm13 = _mm256_permute_ps(ymm13,0b10110001);
			ymm13 = _mm256_mul_ps(ymm12,ymm13);
			ymm12 = _mm256_hadd_ps(ymm14,ymm13);
			ymm7 = _mm256_permute_ps(ymm12,0b11011000);
			
			ymm10 = _mm256_hadd_ps(ymm10,ymm11);
			ymm2 = _mm256_add_ps(ymm2,ymm10);	
			ymm2 = _mm256_add_ps(ymm2,ymm3);	
			ymm9 = _mm256_add_ps(ymm2,ymm9);	
			ymm8 = _mm256_add_ps(ymm8,ymm9);	
			ymm6 = _mm256_add_ps(ymm6,ymm7);
			ymm5 = _mm256_add_ps(ymm5,ymm6);
			ymm4 = _mm256_add_ps(ymm4,ymm5);
			ymm0 = _mm256_add_ps(ymm0,ymm4);
			
			ymm1 = _mm256_set_ps(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
			ymm2 = _mm256_set_ps(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
			for(uint64_t k=0;k<extras;k++)
			{
				ymm12 = _mm256_loadu_ps(data1+(16*howmany2+k)*size+J);
				ymm13 = _mm256_loadu_ps(data2+(16*howmany2+k)*size+J);
				ymm10 = _mm256_mul_ps(ymm12,ymm12);
				ymm11 = _mm256_mul_ps(ymm13,ymm13);
				ymm14 = _mm256_mul_ps(ymm12,ymm13);
				ymm13 = _mm256_mul_ps(ymm13,ymm15);
				ymm13 = _mm256_permute_ps(ymm13,0b10110001);
				ymm13 = _mm256_mul_ps(ymm12,ymm13);
				ymm12 = _mm256_hadd_ps(ymm14,ymm13);
				ymm7 = _mm256_permute_ps(ymm12,0b11011000);
				ymm1 = _mm256_add_ps(ymm1,ymm7);
				ymm6 = _mm256_hadd_ps(ymm10,ymm11);
				ymm2 = _mm256_add_ps(ymm2,ymm6);
			}

			ymm0 = _mm256_add_ps(ymm0,ymm1);
			ymm8 = _mm256_add_ps(ymm8,ymm2);

			_mm256_storeu_ps(data1+i*size+J,ymm8);
			_mm256_storeu_ps(data2+i*size+J,ymm0);
		}
		for(uint64_t j=8*Nregisters;j<size;j+=2)
		{
			temp0 = data1[I*size+j]*data2[I*size+j]+data1[I*size+j+1]*data2[I*size+j+1];
			temp0 += data1[(I+1)*size+j]*data2[(I+1)*size+j];
			temp0 += data1[(I+1)*size+j+1]*data2[(I+1)*size+j+1];
			
			temp1 = data1[I*size+j+1]*data2[I*size+j]-data1[I*size+j]*data2[I*size+j+1];
			temp1 += data1[(I+1)*size+j+1]*data2[(I+1)*size+j];
			temp1 -= data1[(I+1)*size+j]*data2[(I+1)*size+j+1];

			temp2 = data1[(I+2)*size+j]*data2[(I+2)*size+j];
			temp2 += data1[(I+2)*size+j+1]*data2[(I+2)*size+j+1];
			temp2 += data1[(I+3)*size+j]*data2[(I+3)*size+j];
			temp2 += data1[(I+3)*size+j+1]*data2[(I+3)*size+j+1];
			
			temp3 = data1[(I+2)*size+j+1]*data2[(I+2)*size+j];
			temp3 -= data1[(I+2)*size+j]*data2[(I+2)*size+j+1];
			temp3 += data1[(I+3)*size+j+1]*data2[(I+3)*size+j];
			temp3 -= data1[(I+3)*size+j]*data2[(I+3)*size+j+1];

			temp4 = data1[(I+4)*size+j]*data2[(I+4)*size+j];
			temp4 += data1[(I+4)*size+j+1]*data2[(I+4)*size+j+1];
			temp4 += data1[(I+5)*size+j]*data2[(I+5)*size+j];
			temp4 += data1[(I+5)*size+j+1]*data2[(I+5)*size+j+1];
			
			temp5 = data1[(I+4)*size+j+1]*data2[(I+4)*size+j];
			temp5 -= data1[(I+4)*size+j]*data2[(I+4)*size+j+1];
			temp5 += data1[(I+5)*size+j+1]*data2[(I+5)*size+j];
			temp5 -= data1[(I+5)*size+j]*data2[(I+5)*size+j+1];	
			
			temp6 = data1[(I+6)*size+j]*data2[(I+6)*size+j];
			temp6 += data1[(I+6)*size+j+1]*data2[(I+6)*size+j+1];
			temp6 += data1[(I+7)*size+j]*data2[(I+7)*size+j];
			temp6 += data1[(I+7)*size+j+1]*data2[(I+7)*size+j+1];
			
			temp7 = data1[(I+6)*size+j+1]*data2[(I+6)*size+j];
			temp7 -= data1[(I+6)*size+j]*data2[(I+6)*size+j+1];
			temp7 += data1[(I+7)*size+j+1]*data2[(I+7)*size+j];
			temp7 -= data1[(I+7)*size+j]*data2[(I+7)*size+j+1];	
			
			temp0 += temp4;
			temp1 += temp5;
			temp2 += temp6;
			temp3 += temp7;

			temp0 += temp2;
			temp1 += temp3;

			temp2 = data1[(I+8)*size+j]*data2[(I+8)*size+j];
			temp2 += data1[(I+8)*size+j+1]*data2[(I+8)*size+j+1];
			temp2 += data1[(I+9)*size+j]*data2[(I+9)*size+j];
			temp2 += data1[(I+9)*size+j+1]*data2[(I+9)*size+j+1];
			
			temp3 = data1[(I+8)*size+j+1]*data2[(I+8)*size+j];
			temp3 -= data1[(I+8)*size+j]*data2[(I+8)*size+j+1];
			temp3 += data1[(I+9)*size+j+1]*data2[(I+9)*size+j];
			temp3 -= data1[(I+9)*size+j]*data2[(I+9)*size+j+1];

			temp4 = data1[(I+10)*size+j]*data2[(I+10)*size+j];
			temp4 += data1[(I+10)*size+j+1]*data2[(I+10)*size+j+1];
			temp4 += data1[(I+11)*size+j]*data2[(I+11)*size+j];
			temp4 += data1[(I+11)*size+j+1]*data2[(I+11)*size+j+1];
			
			temp5 = data1[(I+10)*size+j+1]*data2[(I+10)*size+j];
			temp5 -= data1[(I+10)*size+j]*data2[(I+10)*size+j+1];
			temp5 += data1[(I+11)*size+j+1]*data2[(I+11)*size+j];
			temp5 -= data1[(I+11)*size+j]*data2[(I+11)*size+j+1];	
			
			temp6 = data1[(I+12)*size+j]*data2[(I+12)*size+j];
			temp6 += data1[(I+12)*size+j+1]*data2[(I+12)*size+j+1];
			temp6 += data1[(I+13)*size+j]*data2[(I+13)*size+j];
			temp6 += data1[(I+13)*size+j+1]*data2[(I+13)*size+j+1];
			
			temp7 = data1[(I+12)*size+j+1]*data2[(I+12)*size+j];
			temp7 -= data1[(I+12)*size+j]*data2[(I+12)*size+j+1];
			temp7 += data1[(I+13)*size+j+1]*data2[(I+13)*size+j];
			temp7 -= data1[(I+13)*size+j]*data2[(I+13)*size+j+1];	

			temp8 = data1[(I+14)*size+j]*data2[(I+14)*size+j];
			temp8 += data1[(I+14)*size+j+1]*data2[(I+14)*size+j+1];
			temp8 += data1[(I+15)*size+j]*data2[(I+15)*size+j];
			temp8 += data1[(I+15)*size+j+1]*data2[(I+15)*size+j+1];
			
			temp9 = data1[(I+14)*size+j+1]*data2[(I+14)*size+j];
			temp9 -= data1[(I+14)*size+j]*data2[(I+14)*size+j+1];
			temp9 += data1[(I+15)*size+j+1]*data2[(I+15)*size+j];
			temp9 -= data1[(I+15)*size+j]*data2[(I+15)*size+j+1];	
			
			temp2 += temp6;
			temp3 += temp7;
			temp4 += temp8;
			temp5 += temp9;

			temp2 += temp4;
			temp3 += temp5;

			temp0 += temp2;
			temp1 += temp3;

			temp2 = data1[I*size+j]*data1[I*size+j];
			temp2 += data1[(I+8)*size+j]*data1[(I+8)*size+j];
			temp3 = data1[I*size+j+1]*data1[I*size+j+1];
			temp3 += data1[(I+8)*size+j+1]*data1[(I+8)*size+j+1];
			
			temp4 = data2[I*size+j]*data2[I*size+j];
			temp4 += data2[(I+8)*size+j]*data2[(I+8)*size+j];
			temp5 = data2[I*size+j+1]*data2[I*size+j+1];
			temp5 += data2[(I+8)*size+j+1]*data2[(I+8)*size+j+1];
			
			temp6 = data1[(I+1)*size+j]*data1[(I+1)*size+j];
			temp6 += data1[(I+9)*size+j]*data1[(I+9)*size+j];
			temp7 = data1[(I+1)*size+j+1]*data1[(I+1)*size+j+1];
			temp7 += data1[(I+9)*size+j+1]*data1[(I+9)*size+j+1];
			
			temp8 = data2[(I+1)*size+j]*data2[(I+1)*size+j];
			temp8 += data2[(I+9)*size+j]*data2[(I+9)*size+j];
			temp9 = data2[(I+1)*size+j+1]*data2[(I+1)*size+j+1];
			temp9 += data2[(I+9)*size+j+1]*data2[(I+9)*size+j+1];
			
			temp10 = data1[(I+2)*size+j]*data1[(I+2)*size+j];
			temp10 += data1[(I+10)*size+j]*data1[(I+10)*size+j];
			temp11 = data1[(I+2)*size+j+1]*data1[(I+2)*size+j+1];
			temp11 += data1[(I+10)*size+j+1]*data1[(I+10)*size+j+1];
			
			temp12 = data2[(I+2)*size+j]*data2[(I+2)*size+j];
			temp12 += data2[(I+10)*size+j]*data2[(I+10)*size+j];
			temp13 = data2[(I+2)*size+j+1]*data2[(I+2)*size+j+1];
			temp13 += data2[(I+10)*size+j+1]*data2[(I+10)*size+j+1];
			
			temp14 = data1[(I+3)*size+j]*data1[(I+3)*size+j];
			temp14 += data1[(I+11)*size+j]*data1[(I+11)*size+j];
			temp15 = data1[(I+3)*size+j+1]*data1[(I+3)*size+j+1];
			temp15 += data1[(I+11)*size+j+1]*data1[(I+11)*size+j+1];
			
			temp16 = data2[(I+3)*size+j]*data2[(I+3)*size+j];
			temp16 += data2[(I+11)*size+j]*data2[(I+11)*size+j];
			temp17 = data2[(I+3)*size+j+1]*data2[(I+3)*size+j+1];
			temp17 += data2[(I+11)*size+j+1]*data2[(I+11)*size+j+1];
			
			temp2 += temp10;
			temp3 += temp11;
			temp4 += temp12;
			temp5 += temp13;
			temp6 += temp14;
			temp7 += temp15;
			temp8 += temp16;
			temp9 += temp17;

			temp2 += temp6;
			temp3 += temp7;
			temp4 += temp8;
			temp5 += temp9;

			temp6 = data1[(I+4)*size+j]*data1[(I+4)*size+j];
			temp6 += data1[(I+12)*size+j]*data1[(I+12)*size+j];
			temp7 = data1[(I+4)*size+j+1]*data1[(I+4)*size+j+1];
			temp7 += data1[(I+12)*size+j+1]*data1[(I+12)*size+j+1];
			
			temp8 = data2[(I+4)*size+j]*data2[(I+4)*size+j];
			temp8 += data2[(I+12)*size+j]*data2[(I+12)*size+j];
			temp9 = data2[(I+4)*size+j+1]*data2[(I+4)*size+j+1];
			temp9 += data2[(I+12)*size+j+1]*data2[(I+12)*size+j+1];
			
			temp10 = data1[(I+5)*size+j]*data1[(I+5)*size+j];
			temp10 += data1[(I+13)*size+j]*data1[(I+13)*size+j];
			temp11 = data1[(I+5)*size+j+1]*data1[(I+5)*size+j+1];
			temp11 += data1[(I+13)*size+j+1]*data1[(I+13)*size+j+1];
			
			temp12 = data2[(I+5)*size+j]*data2[(I+5)*size+j];
			temp12 += data2[(I+13)*size+j]*data2[(I+13)*size+j];
			temp13 = data2[(I+5)*size+j+1]*data2[(I+5)*size+j+1];
			temp13 += data2[(I+13)*size+j+1]*data2[(I+13)*size+j+1];
			
			temp14 = data1[(I+6)*size+j]*data1[(I+6)*size+j];
			temp14 += data1[(I+14)*size+j]*data1[(I+14)*size+j];
			temp15 = data1[(I+6)*size+j+1]*data1[(I+6)*size+j+1];
			temp15 += data1[(I+14)*size+j+1]*data1[(I+14)*size+j+1];
			
			temp16 = data2[(I+6)*size+j]*data2[(I+6)*size+j];
			temp16 += data2[(I+14)*size+j]*data2[(I+14)*size+j];
			temp17 = data2[(I+6)*size+j+1]*data2[(I+6)*size+j+1];
			temp17 += data2[(I+14)*size+j+1]*data2[(I+14)*size+j+1];
			
			temp18 = data1[(I+7)*size+j]*data1[(I+7)*size+j];
			temp18 += data1[(I+15)*size+j]*data1[(I+15)*size+j];
			temp19 = data1[(I+7)*size+j+1]*data1[(I+7)*size+j+1];
			temp19 += data1[(I+15)*size+j+1]*data1[(I+15)*size+j+1];
			
			temp20 = data2[(I+7)*size+j]*data2[(I+7)*size+j];
			temp20 += data2[(I+15)*size+j]*data2[(I+15)*size+j];
			temp21 = data2[(I+7)*size+j+1]*data2[(I+7)*size+j+1];
			temp21 += data2[(I+15)*size+j+1]*data2[(I+15)*size+j+1];
			
			temp6 += temp14;
			temp7 += temp15;
			temp8 += temp16;
			temp9 += temp17;
			temp10 += temp18;
			temp11 += temp19;
			temp12 += temp20;
			temp13 += temp21;

			temp6 += temp10;
			temp7 += temp11;
			temp8 += temp12;
			temp9 += temp13;

			temp2 += temp6;
			temp3 += temp7;
			temp4 += temp8;
			temp5 += temp9;

			temp6 = 0.0;
			temp7 = 0.0;
			temp8 = 0.0;
			temp9 = 0.0;
			temp10 = 0.0;
			temp11 = 0.0;
			for(uint64_t k=0;k<extras;k++)
			{
				temp12 = data1[(16*howmany2+k)*size+j]*data2[(16*howmany2+k)*size+j];
				temp12 += data1[(16*howmany2+k)*size+j+1]*data2[(16*howmany2+k)*size+j+1];
				temp13 = data1[(16*howmany2+k)*size+j+1]*data2[(16*howmany2+k)*size+j];
				temp13 -= data1[(16*howmany2+k)*size+j]*data2[(16*howmany2+k)*size+j+1];

				temp14 = data1[(16*howmany2+k)*size+j]*data1[(16*howmany2+k)*size+j];
				temp15 = data1[(16*howmany2+k)*size+j+1]*data1[(16*howmany2+k)*size+j+1];
				temp16 = data2[(16*howmany2+k)*size+j]*data2[(16*howmany2+k)*size+j];
				temp17 = data2[(16*howmany2+k)*size+j+1]*data2[(16*howmany2+k)*size+j+1];
				
				temp6 += temp12;
				temp7 += temp13;
				temp8 += temp14;
				temp9 += temp15;
				temp10 += temp16;
				temp11 += temp17;
			}

			temp0 += temp6;
			temp1 += temp7;
			temp2 += temp8;
			temp3 += temp9;
			temp4 += temp10;
			temp5 += temp11;
			temp2 += temp3;
			temp4 += temp5;
			
			data1[i*size+j] = temp2;
			data1[i*size+j+1] = temp4;
			data2[i*size+j] = temp0;
			data2[i*size+j+1] = temp1;
		}
		extras = 0;
	}
}

template<>
void fCorrCircFreqReduceAVX<double>(uint64_t N, uint64_t size, double* data1, double* data2)
{
	__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
	__m256d ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
	uint64_t howmany = N/size;
	uint64_t howmany2 = howmany/16;
	uint64_t extras = howmany-16*howmany2;
	uint64_t Nregisters = size/4;
	uint64_t cSize = size/2;
	uint64_t I = 0;
	uint64_t J = 0;
	double temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;
	double temp10, temp11, temp12, temp13, temp14, temp15, temp16, temp17, temp18, temp19;
	double temp20, temp21;
	
	if(howmany == 1)
	{
		for(uint64_t j=0;j<cSize;j++)
		{
			J = j<<1;
			temp0 = data1[J]*data2[J]+data1[J+1]*data2[J+1];
			temp1 = data1[J+1]*data2[J]-data1[J]*data2[J+1];
			temp2 = data1[J]*data1[J]+data1[J+1]*data1[J+1];
			temp3 = data2[J]*data2[J]+data2[J+1]*data2[J+1];
			data1[J] = temp2;
			data1[J+1] = temp3;
			data2[J] = temp0;
			data2[J+1] = temp1;
		}
		return;
	}
	if(howmany < 16)
	{
		for(uint64_t i=0;i<cSize;i++)
		{
			I = i<<1;
			temp0 = data1[I]*data2[I]+data1[I+1]*data2[I+1];
			temp1 = data1[I+1]*data2[I]-data1[I]*data2[I+1];
			temp2 = data1[I]*data1[I]+data1[I+1]*data1[I+1];
			temp3 = data2[I]*data2[I]+data2[I+1]*data2[I+1];
			data1[I] = temp2;
			data1[I+1] = temp3;
			data2[I] = temp0;
			data2[I+1] = temp1;
			for(uint64_t j=1;j<howmany;j++)
			{
				J = I+j*size;
				temp0 = data1[J]*data2[J]+data1[J+1]*data2[J+1];
				temp1 = data1[J+1]*data2[J]-data1[J]*data2[J+1];
				temp2 = data1[J]*data1[J]+data1[J+1]*data1[J+1];
				temp3 = data2[J]*data2[J]+data2[J+1]*data2[J+1];
				data1[I] += temp2;
				data1[I+1] += temp3;
				data2[I] += temp0;
				data2[I+1] += temp1;
			}

		}
		return;
	}

	ymm15 = _mm256_set_pd(-1.0,1.0,-1.0,1.0);
	
	for(uint64_t i=0;i<howmany2;i++)
	{
		I = i<<4;
		for(uint64_t j=0;j<Nregisters;j++)
		{
			J = j<<2;
			ymm12 = _mm256_loadu_pd(data1+I*size+J);
			ymm13 = _mm256_loadu_pd(data2+I*size+J);
			ymm10 = _mm256_mul_pd(ymm12,ymm12);
			ymm11 = _mm256_mul_pd(ymm13,ymm13);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);	
		
			ymm4 = _mm256_hadd_pd(ymm10,ymm11);	
			ymm0 = _mm256_hadd_pd(ymm14,ymm13);
	
			ymm12 = _mm256_loadu_pd(data1+(I+1)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+1)*size+J);
			ymm10 = _mm256_mul_pd(ymm12,ymm12);
			ymm11 = _mm256_mul_pd(ymm13,ymm13);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm5 = _mm256_hadd_pd(ymm10,ymm11);	
			ymm1 = _mm256_hadd_pd(ymm14,ymm13);
			ymm0 = _mm256_add_pd(ymm0,ymm1);

			ymm12 = _mm256_loadu_pd(data1+(I+2)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+2)*size+J);
			ymm10 = _mm256_mul_pd(ymm12,ymm12);
			ymm11 = _mm256_mul_pd(ymm13,ymm13);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);

			ymm6 = _mm256_hadd_pd(ymm10,ymm11);	
			ymm1 = _mm256_hadd_pd(ymm14,ymm13);
	
			ymm12 = _mm256_loadu_pd(data1+(I+3)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+3)*size+J);
			ymm10 = _mm256_mul_pd(ymm12,ymm12);
			ymm11 = _mm256_mul_pd(ymm13,ymm13);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm7 = _mm256_hadd_pd(ymm10,ymm11);
			ymm4 = _mm256_add_pd(ymm4,ymm5);	
			ymm6 = _mm256_add_pd(ymm6,ymm7);
			ymm8 = _mm256_add_pd(ymm4,ymm6);	
			ymm2 = _mm256_hadd_pd(ymm14,ymm13);	
			ymm1 = _mm256_add_pd(ymm1,ymm2);
			ymm0 = _mm256_add_pd(ymm0,ymm1);

			ymm12 = _mm256_loadu_pd(data1+(I+4)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+4)*size+J);
			ymm10 = _mm256_mul_pd(ymm12,ymm12);
			ymm11 = _mm256_mul_pd(ymm13,ymm13);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm4 = _mm256_hadd_pd(ymm10,ymm11);	
			ymm1 = _mm256_hadd_pd(ymm14,ymm13);
	
			ymm12 = _mm256_loadu_pd(data1+(I+5)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+5)*size+J);
			ymm10 = _mm256_mul_pd(ymm12,ymm12);
			ymm11 = _mm256_mul_pd(ymm13,ymm13);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
				
			ymm5 = _mm256_hadd_pd(ymm10,ymm11);	
			ymm2 = _mm256_hadd_pd(ymm14,ymm13);
			ymm1 = _mm256_add_pd(ymm1,ymm2);

			ymm12 = _mm256_loadu_pd(data1+(I+6)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+6)*size+J);
			ymm10 = _mm256_mul_pd(ymm12,ymm12);
			ymm11 = _mm256_mul_pd(ymm13,ymm13);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm6 = _mm256_hadd_pd(ymm10,ymm11);	
			ymm2 = _mm256_hadd_pd(ymm14,ymm13);
	
			ymm12 = _mm256_loadu_pd(data1+(I+7)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+7)*size+J);
			ymm10 = _mm256_mul_pd(ymm12,ymm12);
			ymm11 = _mm256_mul_pd(ymm13,ymm13);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm7 = _mm256_hadd_pd(ymm10,ymm11);
			ymm4 = _mm256_add_pd(ymm4,ymm5);	
			ymm6 = _mm256_add_pd(ymm6,ymm7);
			ymm9 = _mm256_add_pd(ymm4,ymm6);
			ymm8 = _mm256_add_pd(ymm8,ymm9);	
			ymm3 = _mm256_hadd_pd(ymm14,ymm13);
			ymm2 = _mm256_add_pd(ymm2,ymm3);
			ymm1 = _mm256_add_pd(ymm1,ymm2);
			ymm0 = _mm256_add_pd(ymm0,ymm1);
			
			ymm12 = _mm256_loadu_pd(data1+(I+8)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+8)*size+J);
			ymm10 = _mm256_mul_pd(ymm12,ymm12);
			ymm11 = _mm256_mul_pd(ymm13,ymm13);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm1 = _mm256_hadd_pd(ymm10,ymm11);
			ymm4 = _mm256_hadd_pd(ymm14,ymm13);
	
			ymm12 = _mm256_loadu_pd(data1+(I+9)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+9)*size+J);
			ymm10 = _mm256_mul_pd(ymm12,ymm12);
			ymm11 = _mm256_mul_pd(ymm13,ymm13);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm2 = _mm256_hadd_pd(ymm10,ymm11);
			ymm5 = _mm256_hadd_pd(ymm14,ymm13);
			ymm4 = _mm256_add_pd(ymm4,ymm5);

			ymm12 = _mm256_loadu_pd(data1+(I+10)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+10)*size+J);
			ymm10 = _mm256_mul_pd(ymm12,ymm12);
			ymm11 = _mm256_mul_pd(ymm13,ymm13);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm3 = _mm256_hadd_pd(ymm10,ymm11);
			ymm5 = _mm256_hadd_pd(ymm14,ymm13);
	
			ymm12 = _mm256_loadu_pd(data1+(I+11)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+11)*size+J);
			ymm10 = _mm256_mul_pd(ymm12,ymm12);
			ymm11 = _mm256_mul_pd(ymm13,ymm13);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm10 = _mm256_hadd_pd(ymm10,ymm11);
			ymm1 = _mm256_add_pd(ymm1,ymm10);
			ymm2 = _mm256_add_pd(ymm2,ymm3);
			ymm9 = _mm256_add_pd(ymm1,ymm2);
			ymm6 = _mm256_hadd_pd(ymm14,ymm13);
			ymm5 = _mm256_add_pd(ymm5,ymm6);
			ymm4 = _mm256_add_pd(ymm4,ymm5);

			ymm12 = _mm256_loadu_pd(data1+(I+12)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+12)*size+J);
			ymm10 = _mm256_mul_pd(ymm12,ymm12);
			ymm11 = _mm256_mul_pd(ymm13,ymm13);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm1 = _mm256_hadd_pd(ymm10,ymm11);
			ymm5 = _mm256_hadd_pd(ymm14,ymm13);
	
			ymm12 = _mm256_loadu_pd(data1+(I+13)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+13)*size+J);
			ymm10 = _mm256_mul_pd(ymm12,ymm12);
			ymm11 = _mm256_mul_pd(ymm13,ymm13);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm2 = _mm256_hadd_pd(ymm10,ymm11);
			ymm6 = _mm256_hadd_pd(ymm14,ymm13);
			ymm5 = _mm256_add_pd(ymm5,ymm6);

			ymm12 = _mm256_loadu_pd(data1+(I+14)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+14)*size+J);
			ymm10 = _mm256_mul_pd(ymm12,ymm12);
			ymm11 = _mm256_mul_pd(ymm13,ymm13);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
			
			ymm3 = _mm256_hadd_pd(ymm10,ymm11);
			ymm6 = _mm256_hadd_pd(ymm14,ymm13);

			ymm12 = _mm256_loadu_pd(data1+(I+15)*size+J);
			ymm13 = _mm256_loadu_pd(data2+(I+15)*size+J);
			ymm10 = _mm256_mul_pd(ymm12,ymm12);
			ymm11 = _mm256_mul_pd(ymm13,ymm13);
			ymm14 = _mm256_mul_pd(ymm12,ymm13);
			ymm13 = _mm256_mul_pd(ymm13,ymm15);
			ymm13 = _mm256_permute_pd(ymm13,0b00000101);
			ymm13 = _mm256_mul_pd(ymm12,ymm13);
	
			ymm10 = _mm256_hadd_pd(ymm10,ymm11);
			ymm1 = _mm256_add_pd(ymm1,ymm10);
			ymm2 = _mm256_add_pd(ymm2,ymm3);
			ymm10 = _mm256_add_pd(ymm1,ymm2);
			ymm9 = _mm256_add_pd(ymm9,ymm10);
			ymm8 = _mm256_add_pd(ymm8,ymm9);
			ymm7 = _mm256_hadd_pd(ymm14,ymm13);
			ymm6 = _mm256_add_pd(ymm6,ymm7);
			ymm5 = _mm256_add_pd(ymm5,ymm6);
			ymm4 = _mm256_add_pd(ymm4,ymm5);
			ymm0 = _mm256_add_pd(ymm0,ymm4);
			
			ymm1 = _mm256_set_pd(0.0,0.0,0.0,0.0);
			ymm2 = _mm256_set_pd(0.0,0.0,0.0,0.0);
			for(uint64_t k=0;k<extras;k++)
			{
				ymm12 = _mm256_loadu_pd(data1+(16*howmany2+k)*size+J);
				ymm13 = _mm256_loadu_pd(data2+(16*howmany2+k)*size+J);
				ymm10 = _mm256_mul_pd(ymm12,ymm12);
				ymm11 = _mm256_mul_pd(ymm13,ymm13);
				ymm14 = _mm256_mul_pd(ymm12,ymm13);
				ymm13 = _mm256_mul_pd(ymm13,ymm15);
				ymm13 = _mm256_permute_pd(ymm13,0b00000101);
				ymm13 = _mm256_mul_pd(ymm12,ymm13);
				
				ymm6 = _mm256_hadd_pd(ymm10,ymm11);
				ymm7 = _mm256_hadd_pd(ymm14,ymm13);
				ymm1 = _mm256_add_pd(ymm1,ymm7);
				ymm2 = _mm256_add_pd(ymm2,ymm6);	
			}

			ymm0 = _mm256_add_pd(ymm0,ymm1);
			ymm8 = _mm256_add_pd(ymm8,ymm2);

			_mm256_storeu_pd(data1+i*size+J,ymm8);
			_mm256_storeu_pd(data2+i*size+J,ymm0);
		}
		for(uint64_t j=4*Nregisters;j<size;j+=2)
		{
			temp0 = data1[I*size+j]*data2[I*size+j]+data1[I*size+j+1]*data2[I*size+j+1];
			temp0 += data1[(I+1)*size+j]*data2[(I+1)*size+j];
			temp0 += data1[(I+1)*size+j+1]*data2[(I+1)*size+j+1];
			
			temp1 = data1[I*size+j+1]*data2[I*size+j]-data1[I*size+j]*data2[I*size+j+1];
			temp1 += data1[(I+1)*size+j+1]*data2[(I+1)*size+j];
			temp1 -= data1[(I+1)*size+j]*data2[(I+1)*size+j+1];

			temp2 = data1[(I+2)*size+j]*data2[(I+2)*size+j];
			temp2 += data1[(I+2)*size+j+1]*data2[(I+2)*size+j+1];
			temp2 += data1[(I+3)*size+j]*data2[(I+3)*size+j];
			temp2 += data1[(I+3)*size+j+1]*data2[(I+3)*size+j+1];
			
			temp3 = data1[(I+2)*size+j+1]*data2[(I+2)*size+j];
			temp3 -= data1[(I+2)*size+j]*data2[(I+2)*size+j+1];
			temp3 += data1[(I+3)*size+j+1]*data2[(I+3)*size+j];
			temp3 -= data1[(I+3)*size+j]*data2[(I+3)*size+j+1];

			temp4 = data1[(I+4)*size+j]*data2[(I+4)*size+j];
			temp4 += data1[(I+4)*size+j+1]*data2[(I+4)*size+j+1];
			temp4 += data1[(I+5)*size+j]*data2[(I+5)*size+j];
			temp4 += data1[(I+5)*size+j+1]*data2[(I+5)*size+j+1];
			
			temp5 = data1[(I+4)*size+j+1]*data2[(I+4)*size+j];
			temp5 -= data1[(I+4)*size+j]*data2[(I+4)*size+j+1];
			temp5 += data1[(I+5)*size+j+1]*data2[(I+5)*size+j];
			temp5 -= data1[(I+5)*size+j]*data2[(I+5)*size+j+1];	
			
			temp6 = data1[(I+6)*size+j]*data2[(I+6)*size+j];
			temp6 += data1[(I+6)*size+j+1]*data2[(I+6)*size+j+1];
			temp6 += data1[(I+7)*size+j]*data2[(I+7)*size+j];
			temp6 += data1[(I+7)*size+j+1]*data2[(I+7)*size+j+1];
			
			temp7 = data1[(I+6)*size+j+1]*data2[(I+6)*size+j];
			temp7 -= data1[(I+6)*size+j]*data2[(I+6)*size+j+1];
			temp7 += data1[(I+7)*size+j+1]*data2[(I+7)*size+j];
			temp7 -= data1[(I+7)*size+j]*data2[(I+7)*size+j+1];	
			
			temp0 += temp4;
			temp1 += temp5;
			temp2 += temp6;
			temp3 += temp7;

			temp0 += temp2;
			temp1 += temp3;

			temp2 = data1[(I+8)*size+j]*data2[(I+8)*size+j];
			temp2 += data1[(I+8)*size+j+1]*data2[(I+8)*size+j+1];
			temp2 += data1[(I+9)*size+j]*data2[(I+9)*size+j];
			temp2 += data1[(I+9)*size+j+1]*data2[(I+9)*size+j+1];
			
			temp3 = data1[(I+8)*size+j+1]*data2[(I+8)*size+j];
			temp3 -= data1[(I+8)*size+j]*data2[(I+8)*size+j+1];
			temp3 += data1[(I+9)*size+j+1]*data2[(I+9)*size+j];
			temp3 -= data1[(I+9)*size+j]*data2[(I+9)*size+j+1];

			temp4 = data1[(I+10)*size+j]*data2[(I+10)*size+j];
			temp4 += data1[(I+10)*size+j+1]*data2[(I+10)*size+j+1];
			temp4 += data1[(I+11)*size+j]*data2[(I+11)*size+j];
			temp4 += data1[(I+11)*size+j+1]*data2[(I+11)*size+j+1];
			
			temp5 = data1[(I+10)*size+j+1]*data2[(I+10)*size+j];
			temp5 -= data1[(I+10)*size+j]*data2[(I+10)*size+j+1];
			temp5 += data1[(I+11)*size+j+1]*data2[(I+11)*size+j];
			temp5 -= data1[(I+11)*size+j]*data2[(I+11)*size+j+1];	
			
			temp6 = data1[(I+12)*size+j]*data2[(I+12)*size+j];
			temp6 += data1[(I+12)*size+j+1]*data2[(I+12)*size+j+1];
			temp6 += data1[(I+13)*size+j]*data2[(I+13)*size+j];
			temp6 += data1[(I+13)*size+j+1]*data2[(I+13)*size+j+1];
			
			temp7 = data1[(I+12)*size+j+1]*data2[(I+12)*size+j];
			temp7 -= data1[(I+12)*size+j]*data2[(I+12)*size+j+1];
			temp7 += data1[(I+13)*size+j+1]*data2[(I+13)*size+j];
			temp7 -= data1[(I+13)*size+j]*data2[(I+13)*size+j+1];	

			temp8 = data1[(I+14)*size+j]*data2[(I+14)*size+j];
			temp8 += data1[(I+14)*size+j+1]*data2[(I+14)*size+j+1];
			temp8 += data1[(I+15)*size+j]*data2[(I+15)*size+j];
			temp8 += data1[(I+15)*size+j+1]*data2[(I+15)*size+j+1];
			
			temp9 = data1[(I+14)*size+j+1]*data2[(I+14)*size+j];
			temp9 -= data1[(I+14)*size+j]*data2[(I+14)*size+j+1];
			temp9 += data1[(I+15)*size+j+1]*data2[(I+15)*size+j];
			temp9 -= data1[(I+15)*size+j]*data2[(I+15)*size+j+1];	
			
			temp2 += temp6;
			temp3 += temp7;
			temp4 += temp8;
			temp5 += temp9;

			temp2 += temp4;
			temp3 += temp5;

			temp0 += temp2;
			temp1 += temp3;

			temp2 = data1[I*size+j]*data1[I*size+j];
			temp2 += data1[(I+8)*size+j]*data1[(I+8)*size+j];
			temp3 = data1[I*size+j+1]*data1[I*size+j+1];
			temp3 += data1[(I+8)*size+j+1]*data1[(I+8)*size+j+1];
			
			temp4 = data2[I*size+j]*data2[I*size+j];
			temp4 += data2[(I+8)*size+j]*data2[(I+8)*size+j];
			temp5 = data2[I*size+j+1]*data2[I*size+j+1];
			temp5 += data2[(I+8)*size+j+1]*data2[(I+8)*size+j+1];
			
			temp6 = data1[(I+1)*size+j]*data1[(I+1)*size+j];
			temp6 += data1[(I+9)*size+j]*data1[(I+9)*size+j];
			temp7 = data1[(I+1)*size+j+1]*data1[(I+1)*size+j+1];
			temp7 += data1[(I+9)*size+j+1]*data1[(I+9)*size+j+1];
			
			temp8 = data2[(I+1)*size+j]*data2[(I+1)*size+j];
			temp8 += data2[(I+9)*size+j]*data2[(I+9)*size+j];
			temp9 = data2[(I+1)*size+j+1]*data2[(I+1)*size+j+1];
			temp9 += data2[(I+9)*size+j+1]*data2[(I+9)*size+j+1];
			
			temp10 = data1[(I+2)*size+j]*data1[(I+2)*size+j];
			temp10 += data1[(I+10)*size+j]*data1[(I+10)*size+j];
			temp11 = data1[(I+2)*size+j+1]*data1[(I+2)*size+j+1];
			temp11 += data1[(I+10)*size+j+1]*data1[(I+10)*size+j+1];
			
			temp12 = data2[(I+2)*size+j]*data2[(I+2)*size+j];
			temp12 += data2[(I+10)*size+j]*data2[(I+10)*size+j];
			temp13 = data2[(I+2)*size+j+1]*data2[(I+2)*size+j+1];
			temp13 += data2[(I+10)*size+j+1]*data2[(I+10)*size+j+1];
			
			temp14 = data1[(I+3)*size+j]*data1[(I+3)*size+j];
			temp14 += data1[(I+11)*size+j]*data1[(I+11)*size+j];
			temp15 = data1[(I+3)*size+j+1]*data1[(I+3)*size+j+1];
			temp15 += data1[(I+11)*size+j+1]*data1[(I+11)*size+j+1];
			
			temp16 = data2[(I+3)*size+j]*data2[(I+3)*size+j];
			temp16 += data2[(I+11)*size+j]*data2[(I+11)*size+j];
			temp17 = data2[(I+3)*size+j+1]*data2[(I+3)*size+j+1];
			temp17 += data2[(I+11)*size+j+1]*data2[(I+11)*size+j+1];
			
			temp2 += temp10;
			temp3 += temp11;
			temp4 += temp12;
			temp5 += temp13;
			temp6 += temp14;
			temp7 += temp15;
			temp8 += temp16;
			temp9 += temp17;

			temp2 += temp6;
			temp3 += temp7;
			temp4 += temp8;
			temp5 += temp9;

			temp6 = data1[(I+4)*size+j]*data1[(I+4)*size+j];
			temp6 += data1[(I+12)*size+j]*data1[(I+12)*size+j];
			temp7 = data1[(I+4)*size+j+1]*data1[(I+4)*size+j+1];
			temp7 += data1[(I+12)*size+j+1]*data1[(I+12)*size+j+1];
			
			temp8 = data2[(I+4)*size+j]*data2[(I+4)*size+j];
			temp8 += data2[(I+12)*size+j]*data2[(I+12)*size+j];
			temp9 = data2[(I+4)*size+j+1]*data2[(I+4)*size+j+1];
			temp9 += data2[(I+12)*size+j+1]*data2[(I+12)*size+j+1];
			
			temp10 = data1[(I+5)*size+j]*data1[(I+5)*size+j];
			temp10 += data1[(I+13)*size+j]*data1[(I+13)*size+j];
			temp11 = data1[(I+5)*size+j+1]*data1[(I+5)*size+j+1];
			temp11 += data1[(I+13)*size+j+1]*data1[(I+13)*size+j+1];
			
			temp12 = data2[(I+5)*size+j]*data2[(I+5)*size+j];
			temp12 += data2[(I+13)*size+j]*data2[(I+13)*size+j];
			temp13 = data2[(I+5)*size+j+1]*data2[(I+5)*size+j+1];
			temp13 += data2[(I+13)*size+j+1]*data2[(I+13)*size+j+1];
			
			temp14 = data1[(I+6)*size+j]*data1[(I+6)*size+j];
			temp14 += data1[(I+14)*size+j]*data1[(I+14)*size+j];
			temp15 = data1[(I+6)*size+j+1]*data1[(I+6)*size+j+1];
			temp15 += data1[(I+14)*size+j+1]*data1[(I+14)*size+j+1];
			
			temp16 = data2[(I+6)*size+j]*data2[(I+6)*size+j];
			temp16 += data2[(I+14)*size+j]*data2[(I+14)*size+j];
			temp17 = data2[(I+6)*size+j+1]*data2[(I+6)*size+j+1];
			temp17 += data2[(I+14)*size+j+1]*data2[(I+14)*size+j+1];
			
			temp18 = data1[(I+7)*size+j]*data1[(I+7)*size+j];
			temp18 += data1[(I+15)*size+j]*data1[(I+15)*size+j];
			temp19 = data1[(I+7)*size+j+1]*data1[(I+7)*size+j+1];
			temp19 += data1[(I+15)*size+j+1]*data1[(I+15)*size+j+1];
			
			temp20 = data2[(I+7)*size+j]*data2[(I+7)*size+j];
			temp20 += data2[(I+15)*size+j]*data2[(I+15)*size+j];
			temp21 = data2[(I+7)*size+j+1]*data2[(I+7)*size+j+1];
			temp21 += data2[(I+15)*size+j+1]*data2[(I+15)*size+j+1];
			
			temp6 += temp14;
			temp7 += temp15;
			temp8 += temp16;
			temp9 += temp17;
			temp10 += temp18;
			temp11 += temp19;
			temp12 += temp20;
			temp13 += temp21;

			temp6 += temp10;
			temp7 += temp11;
			temp8 += temp12;
			temp9 += temp13;

			temp2 += temp6;
			temp3 += temp7;
			temp4 += temp8;
			temp5 += temp9;

			temp6 = 0.0;
			temp7 = 0.0;
			temp8 = 0.0;
			temp9 = 0.0;
			temp10 = 0.0;
			temp11 = 0.0;
			for(uint64_t k=0;k<extras;k++)
			{
				temp12 = data1[(16*howmany2+k)*size+j]*data2[(16*howmany2+k)*size+j];
				temp12 += data1[(16*howmany2+k)*size+j+1]*data2[(16*howmany2+k)*size+j+1];
				temp13 = data1[(16*howmany2+k)*size+j+1]*data2[(16*howmany2+k)*size+j];
				temp13 -= data1[(16*howmany2+k)*size+j]*data2[(16*howmany2+k)*size+j+1];

				temp14 = data1[(16*howmany2+k)*size+j]*data1[(16*howmany2+k)*size+j];
				temp15 = data1[(16*howmany2+k)*size+j+1]*data1[(16*howmany2+k)*size+j+1];
				temp16 = data2[(16*howmany2+k)*size+j]*data2[(16*howmany2+k)*size+j];
				temp17 = data2[(16*howmany2+k)*size+j+1]*data2[(16*howmany2+k)*size+j+1];
				
				temp6 += temp12;
				temp7 += temp13;
				temp8 += temp14;
				temp9 += temp15;
				temp10 += temp16;
				temp11 += temp17;
			}

			temp0 += temp6;
			temp1 += temp7;
			temp2 += temp8;
			temp3 += temp9;
			temp4 += temp10;
			temp5 += temp11;
			temp2 += temp3;
			temp4 += temp5;
			
			data1[i*size+j] = temp2;
			data1[i*size+j+1] = temp4;
			data2[i*size+j] = temp0;
			data2[i*size+j+1] = temp1;
		}
		extras = 0;
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
void reduceInPlaceBlockAVX(uint64_t N, uint64_t size, DataType* data){}

template<>
void reduceInPlaceBlockAVX<float>(uint64_t N, uint64_t size, float* data)
{
	__m256 ymm0,ymm1,ymm2,ymm3,ymm4,ymm5,ymm6,ymm7,ymm8,ymm9,ymm10,ymm11,ymm12,ymm13,ymm14,ymm15;
    float temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;
    uint64_t I,J,L, offset, registers, howmany, excess;
    uint64_t powers[16];
	
    howmany = N/size;
    registers = size/8;
	excess = size-(registers*8);

    if(howmany <= 1){return;}

    if(howmany < 16)
    {
        for(uint64_t i=0;i<size;i++)
        {
            for(uint64_t j=1;j<howmany;j++){data[i] += data[j*size+i];}
        }
        return;
    }

	for(int i=15;i>=0;i--)
    {
        powers[i]=(howmany>>(i<<2));
        howmany -= powers[i]<<(i<<2);
    }
    
    howmany = N/size;

    offset = powers[0]*16;

    for(uint64_t i=0;i<size;i++)
    {
        if(offset == 0){break;}
        for(uint64_t j=0;j<powers[0];j++)
        {
            data[i] += data[(howmany-powers[0]+j)*size+i];
        }
    }

    offset=0;

    for(uint64_t i=1;i<16;i++)
    {
        I = 1<<(i<<2);
        for(uint64_t j=0;j<i;j++)
        {
            if(powers[i]==0){break;}
            J = (1<<((i-j)<<2));
            for(uint64_t k=0;k<(powers[i]*J);k+=16)
            {
                for(uint64_t l=0;l<registers;l++)
                {
                    L = 8*l+offset;
                    ymm0 = _mm256_loadu_ps(data+k*size+L);
                    ymm1 = _mm256_loadu_ps(data+(k+1)*size+L);
                    ymm2 = _mm256_loadu_ps(data+(k+2)*size+L);
                    ymm3 = _mm256_loadu_ps(data+(k+3)*size+L);
                    ymm4 = _mm256_loadu_ps(data+(k+4)*size+L);
                    ymm5 = _mm256_loadu_ps(data+(k+5)*size+L);
                    ymm6 = _mm256_loadu_ps(data+(k+6)*size+L);
                    ymm7 = _mm256_loadu_ps(data+(k+7)*size+L);
                    ymm8 = _mm256_loadu_ps(data+(k+8)*size+L);
                    ymm9 = _mm256_loadu_ps(data+(k+9)*size+L);
                    ymm10 = _mm256_loadu_ps(data+(k+10)*size+L);
                    ymm11 = _mm256_loadu_ps(data+(k+11)*size+L);
                    ymm12 = _mm256_loadu_ps(data+(k+12)*size+L);
                    ymm13 = _mm256_loadu_ps(data+(k+13)*size+L);
                    ymm14 = _mm256_loadu_ps(data+(k+14)*size+L);
                    ymm15 = _mm256_loadu_ps(data+(k+15)*size+L);

                    ymm0 = _mm256_add_ps(ymm0,ymm8);
                    ymm1 = _mm256_add_ps(ymm1,ymm9);
                    ymm2 = _mm256_add_ps(ymm2,ymm10);
                    ymm3 = _mm256_add_ps(ymm3,ymm11);
                    ymm4 = _mm256_add_ps(ymm4,ymm12);
                    ymm5 = _mm256_add_ps(ymm5,ymm13);
                    ymm6 = _mm256_add_ps(ymm6,ymm14);
                    ymm7 = _mm256_add_ps(ymm7,ymm15);

                    ymm0 = _mm256_add_ps(ymm0,ymm4);
                    ymm1 = _mm256_add_ps(ymm1,ymm5);
                    ymm2 = _mm256_add_ps(ymm2,ymm6);
                    ymm3 = _mm256_add_ps(ymm3,ymm7);

                    ymm0 = _mm256_add_ps(ymm0,ymm2);
                    ymm1 = _mm256_add_ps(ymm1,ymm3);

                    ymm0 = _mm256_add_ps(ymm0,ymm1);
                    
                    _mm256_storeu_ps(data+k/16*size+L,ymm0);
                }
                for(uint64_t l=0;l<excess;l++)
				{
                    L = 8*registers+l+offset;
                    temp0 = data[k*size+L]+data[(k+1)*size+L];
                    temp1 = data[(k+2)*size+L]+data[(k+3)*size+L];
                    temp2 = data[(k+4)*size+L]+data[(k+5)*size+L];
                    temp3 = data[(k+6)*size+L]+data[(k+7)*size+L];
                    temp4 = data[(k+8)*size+L]+data[(k+9)*size+L];
                    temp5 = data[(k+10)*size+L]+data[(k+11)*size+L];
                    temp6 = data[(k+12)*size+L]+data[(k+13)*size+L];
                    temp7 = data[(k+14)*size+L]+data[(k+15)*size+L];

                    temp0 += temp4;
                    temp1 += temp5;
                    temp2 += temp6;
                    temp3 += temp7;

                    temp0 += temp3;
                    temp1 += temp2;

                    temp0 += temp1;
				    
                    data[(k/16)*size+L] = temp0;
				}
            }
        }
        for(uint64_t j=0;j<size;j++)
        {
            for(uint64_t k=(offset == 0);k<powers[i];k++){data[j] += data[j+size*k+offset];}
        }
        offset += I*powers[i]*size;
    }
}

template<>
void reduceInPlaceBlockAVX<double>(uint64_t N, uint64_t size, double* data)
{
	__m256d ymm0,ymm1,ymm2,ymm3,ymm4,ymm5,ymm6,ymm7,ymm8,ymm9,ymm10,ymm11,ymm12,ymm13,ymm14,ymm15;
    double temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;
    uint64_t I,J,L, offset, registers, howmany, excess;
    uint64_t powers[16];
	
    howmany = N/size;
    registers = size/4;
	excess = size-(registers*4);

    if(howmany <= 1){return;}

    if(howmany < 16)
    {
        for(uint64_t i=0;i<size;i++)
        {
            for(uint64_t j=1;j<howmany;j++){data[i] += data[j*size+i];}
        }
        return;
    }

	for(int i=15;i>=0;i--)
    {
        powers[i]=(howmany>>(i<<2));
        howmany -= powers[i]<<(i<<2);
    }
    
    howmany = N/size;

    offset = powers[0]*16;

    if(powers[0] != 0)
	{
    	for(uint64_t i=0;i<size;i++)
    	{
        	for(uint64_t j=0;j<powers[0];j++)
        	{
            	data[i+j*size] += data[(howmany-powers[0]+j)*size+i];
            	offset=0;
        	}
    	}
	}

    for(uint64_t i=1;i<16;i++)
    {
		I = 1<<(i<<2);
        for(uint64_t j=0;j<i;j++)
        {
            if(powers[i]==0){break;}
            J = (1<<((i-j)<<2));
            for(uint64_t k=0;k<(powers[i]*J);k+=16)
            {
                for(uint64_t l=0;l<registers;l++)
                {
                    L = 4*l+offset;
                    ymm0 = _mm256_loadu_pd(data+k*size+L);
                    ymm1 = _mm256_loadu_pd(data+(k+1)*size+L);
                    ymm2 = _mm256_loadu_pd(data+(k+2)*size+L);
                    ymm3 = _mm256_loadu_pd(data+(k+3)*size+L);
                    ymm4 = _mm256_loadu_pd(data+(k+4)*size+L);
                    ymm5 = _mm256_loadu_pd(data+(k+5)*size+L);
                    ymm6 = _mm256_loadu_pd(data+(k+6)*size+L);
                    ymm7 = _mm256_loadu_pd(data+(k+7)*size+L);
                    ymm8 = _mm256_loadu_pd(data+(k+8)*size+L);
                    ymm9 = _mm256_loadu_pd(data+(k+9)*size+L);
                    ymm10 = _mm256_loadu_pd(data+(k+10)*size+L);
                    ymm11 = _mm256_loadu_pd(data+(k+11)*size+L);
                    ymm12 = _mm256_loadu_pd(data+(k+12)*size+L);
                    ymm13 = _mm256_loadu_pd(data+(k+13)*size+L);
                    ymm14 = _mm256_loadu_pd(data+(k+14)*size+L);
                    ymm15 = _mm256_loadu_pd(data+(k+15)*size+L);

                    ymm0 = _mm256_add_pd(ymm0,ymm8);
                    ymm1 = _mm256_add_pd(ymm1,ymm9);
                    ymm2 = _mm256_add_pd(ymm2,ymm10);
                    ymm3 = _mm256_add_pd(ymm3,ymm11);
                    ymm4 = _mm256_add_pd(ymm4,ymm12);
                    ymm5 = _mm256_add_pd(ymm5,ymm13);
                    ymm6 = _mm256_add_pd(ymm6,ymm14);
                    ymm7 = _mm256_add_pd(ymm7,ymm15);

                    ymm0 = _mm256_add_pd(ymm0,ymm4);
                    ymm1 = _mm256_add_pd(ymm1,ymm5);
                    ymm2 = _mm256_add_pd(ymm2,ymm6);
                    ymm3 = _mm256_add_pd(ymm3,ymm7);

                    ymm0 = _mm256_add_pd(ymm0,ymm2);
                    ymm1 = _mm256_add_pd(ymm1,ymm3);

                    ymm0 = _mm256_add_pd(ymm0,ymm1);
                    
                    _mm256_storeu_pd(data+k/16*size+L,ymm0);
                }
                for(uint64_t l=0;l<excess;l++)
				{
                    L = 4*registers+l+offset;
                    temp0 = data[k*size+L]+data[(k+1)*size+L];
                    temp1 = data[(k+2)*size+L]+data[(k+3)*size+L];
                    temp2 = data[(k+4)*size+L]+data[(k+5)*size+L];
                    temp3 = data[(k+6)*size+L]+data[(k+7)*size+L];
                    temp4 = data[(k+8)*size+L]+data[(k+9)*size+L];
                    temp5 = data[(k+10)*size+L]+data[(k+11)*size+L];
                    temp6 = data[(k+12)*size+L]+data[(k+13)*size+L];
                    temp7 = data[(k+14)*size+L]+data[(k+15)*size+L];

                    temp0 += temp4;
                    temp1 += temp5;
                    temp2 += temp6;
                    temp3 += temp7;

                    temp0 += temp3;
                    temp1 += temp2;

                    temp0 += temp1;
				    
                    data[(k/16)*size+L] = temp0;
				}
            }
        }
        for(uint64_t j=0;j<size;j++)
        {
            for(uint64_t k=(offset == 0);k<powers[i];k++){data[j] += data[j+size*k+offset];}
        }
		offset += I*powers[i]*size;
    }
}
