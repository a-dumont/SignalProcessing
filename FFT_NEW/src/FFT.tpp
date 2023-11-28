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
///////////////////////////////////////////////////////////////////
//						 _____ _____ _____                       //
//						|  ___|  ___|_   _|                      //
//						| |_  | |_    | |                        //
//						|  _| |  _|   | |                        //
//						|_|   |_|     |_|                        //
///////////////////////////////////////////////////////////////////
template<class DataType>
void fft(uint64_t N, std::complex<DataType>* in, std::complex<DataType>* out){}

template<>
void fft<double>(uint64_t N, std::complex<double>* in, std::complex<double>* out)
{
	fftw_plan plan;
	plan = fftw_plan_dft_1d(
					N, 
					reinterpret_cast<fftw_complex*>(in), 
					reinterpret_cast<fftw_complex*>(out), 
					FFTW_FORWARD, 
					FFTW_ESTIMATE);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<>
void fft<float>(uint64_t N, std::complex<float>* in, std::complex<float>* out)
{
	fftwf_plan plan;
	plan = fftwf_plan_dft_1d(
					N, 
					reinterpret_cast<fftwf_complex*>(in), 
					reinterpret_cast<fftwf_complex*>(out), 
					FFTW_FORWARD, 
					FFTW_ESTIMATE);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
}

template<class DataType>
void fft_training(uint64_t N, std::complex<DataType>* in, std::complex<DataType>* out){}

template<>
void fft_training<double>(uint64_t N, std::complex<double>* in, std::complex<double>* out)
{
	fftw_plan plan;
	plan = fftw_plan_dft_1d(
					N, 
					reinterpret_cast<fftw_complex*>(in), 
					reinterpret_cast<fftw_complex*>(out), 
					FFTW_FORWARD, 
					FFTW_EXHAUSTIVE);
    fftw_export_wisdom_to_filename(&wisdom_path[0]);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<>
void fft_training<float>(uint64_t N, std::complex<float>* in, std::complex<float>* out)
{
	fftwf_plan plan;
	plan = fftwf_plan_dft_1d(
					N, 
					reinterpret_cast<fftwf_complex*>(in), 
					reinterpret_cast<fftwf_complex*>(out), 
					FFTW_FORWARD, 
					FFTW_EXHAUSTIVE);
    fftwf_export_wisdom_to_filename(&wisdom_path[0]);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
}

template<class DataType>
void fftBlock(int N, int size, std::complex<DataType>* in, std::complex<DataType>* out){}

template<>
void fftBlock<double>(int N, int size, std::complex<double>* in, std::complex<double>* out)
{

	int rank = 1;
	int length[] = {size};
	int howmany = N/size;
	int dist = size;
	int stride = 1;

	fftw_plan plan = fftw_plan_many_dft(
					rank,
					length,
					howmany,
					reinterpret_cast<fftw_complex*>(in),
					NULL,
					stride,
					dist,
					reinterpret_cast<fftw_complex*>(out),
					NULL,
					stride,
					dist,
					1,
					FFTW_ESTIMATE);

	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<>
void fftBlock<float>(int N, int size, std::complex<float>* in, std::complex<float>* out)
{

	int rank = 1;
	int length[] = {size};
	int howmany = N/size;
	int dist = size;
	int stride = 1;

	fftwf_plan plan = fftwf_plan_many_dft(
					rank,
					length,
					howmany,
					reinterpret_cast<fftwf_complex*>(in),
					NULL,
					stride,
					dist,
					reinterpret_cast<fftwf_complex*>(out),
					NULL,
					stride,
					dist,
					1,
					FFTW_ESTIMATE);

    //fftw_export_wisdom_to_filename(&wisdom_path[0]);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
}

template<class DataType>
void fftBlock_training(int N, int size, std::complex<DataType>* in, std::complex<DataType>* out){}

template<>
void 
fftBlock_training<double>(int N, int size, std::complex<double>* in, std::complex<double>* out)
{

	int rank = 1;
	int length[] = {size};
	int howmany = N/size;
	int dist = size;
	int stride = 1;

	fftw_plan plan = fftw_plan_many_dft(
					rank,
					length,
					howmany,
					reinterpret_cast<fftw_complex*>(in),
					NULL,
					stride,
					dist,
					reinterpret_cast<fftw_complex*>(out),
					NULL,
					stride,
					dist,
					1,
					FFTW_EXHAUSTIVE);

    fftw_export_wisdom_to_filename(&wisdom_path[0]);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<>
void fftBlock_training<float>(int N, int size, std::complex<float>* in, std::complex<float>* out)
{

	int rank = 1;
	int length[] = {size};
	int howmany = N/size;
	int dist = size;
	int stride = 1;

	fftwf_plan plan = fftwf_plan_many_dft(
					rank,
					length,
					howmany,
					reinterpret_cast<fftwf_complex*>(in),
					NULL,
					stride,
					dist,
					reinterpret_cast<fftwf_complex*>(out),
					NULL,
					stride,
					dist,
					1,
					FFTW_EXHAUSTIVE);

    fftwf_export_wisdom_to_filename(&wisdom_path[0]);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
}

///////////////////////////////////////////////////////////////////
//						      _____ _____ _____                  //
//						 _ __|  ___|  ___|_   _|                 //
//						| '__| |_  | |_    | |                   //
//						| |  |  _| |  _|   | |                   //
//						|_|  |_|   |_|     |_|                   //
///////////////////////////////////////////////////////////////////
template<class DataType>
void rfft(uint64_t N, DataType* in, std::complex<DataType>* out){}

template<>
void rfft<double>(uint64_t N, double* in, std::complex<double>* out)
{
	fftw_plan plan;
	plan = fftw_plan_dft_r2c_1d(
					N, 
					in, 
					reinterpret_cast<fftw_complex*>(out), 
					FFTW_ESTIMATE);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<>
void rfft<float>(uint64_t N, float* in, std::complex<float>* out)
{
	fftwf_plan plan;
	plan = fftwf_plan_dft_r2c_1d(
					N, 
					in, 
					reinterpret_cast<fftwf_complex*>(out), 
					FFTW_ESTIMATE);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
}

template<class DataType>
void rfft_training(uint64_t N, DataType* in, std::complex<DataType>* out){}

template<>
void rfft_training<double>(uint64_t N, double* in, std::complex<double>* out)
{
	fftw_plan plan;
	plan = fftw_plan_dft_r2c_1d(
					N, 
					in, 
					reinterpret_cast<fftw_complex*>(out), 
					FFTW_EXHAUSTIVE);
    fftw_export_wisdom_to_filename(&wisdom_path[0]);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<>
void rfft_training<float>(uint64_t N, float* in, std::complex<float>* out)
{
	fftwf_plan plan;
	plan = fftwf_plan_dft_r2c_1d(
					N, 
					in, 
					reinterpret_cast<fftwf_complex*>(out), 
					FFTW_EXHAUSTIVE);
    fftwf_export_wisdom_to_filename(&wisdom_path[0]);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
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

template<class DataType>
void rfftBlock_training(int N, int size, DataType* in, std::complex<DataType>* out){}

template<>
void rfftBlock_training<double>(int N, int size, double* in, std::complex<double>* out)
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
					FFTW_EXHAUSTIVE);

	fftw_export_wisdom_to_filename(&wisdom_path[0]);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<>
void rfftBlock_training<float>(int N, int size, float* in, std::complex<float>* out)
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
					FFTW_EXHAUSTIVE);

	fftwf_export_wisdom_to_filename(&wisdom_path[0]);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
}

///////////////////////////////////////////////////////////////////
//						 _ _____ _____ _____                     //
//						(_)  ___|  ___|_   _|                    //
//						| | |_  | |_    | |                      //
//						| |  _| |  _|   | |                      //
//						|_|_|   |_|     |_|                      //
///////////////////////////////////////////////////////////////////
template<class DataType>
void ifft(uint64_t N, std::complex<DataType>* in, std::complex<DataType>* out){}

template<>
void ifft<double>(uint64_t N, std::complex<double>* in, std::complex<double>* out)
{
	fftw_plan plan;
	plan = fftw_plan_dft_1d(
					N, 
					reinterpret_cast<fftw_complex*>(in), 
					reinterpret_cast<fftw_complex*>(out), 
					FFTW_BACKWARD, 
					FFTW_ESTIMATE);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<>
void ifft<float>(uint64_t N, std::complex<float>* in, std::complex<float>* out)
{
	fftwf_plan plan;
	plan = fftwf_plan_dft_1d(
					N, 
					reinterpret_cast<fftwf_complex*>(in), 
					reinterpret_cast<fftwf_complex*>(out), 
					FFTW_BACKWARD, 
					FFTW_ESTIMATE);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
}

template<class DataType>
void ifft_training(uint64_t N, std::complex<DataType>* in, std::complex<DataType>* out){}

template<>
void ifft_training<double>(uint64_t N, std::complex<double>* in, std::complex<double>* out)
{
	fftw_plan plan;
	plan = fftw_plan_dft_1d(
					N, 
					reinterpret_cast<fftw_complex*>(in), 
					reinterpret_cast<fftw_complex*>(out), 
					FFTW_BACKWARD, 
					FFTW_EXHAUSTIVE);
    fftw_export_wisdom_to_filename(&wisdom_path[0]);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<>
void ifft_training<float>(uint64_t N, std::complex<float>* in, std::complex<float>* out)
{
	fftwf_plan plan;
	plan = fftwf_plan_dft_1d(
					N, 
					reinterpret_cast<fftwf_complex*>(in), 
					reinterpret_cast<fftwf_complex*>(out), 
					FFTW_BACKWARD, 
					FFTW_EXHAUSTIVE);
    fftwf_export_wisdom_to_filename(&wisdom_path[0]);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
}

template<class DataType>
void ifftBlock(int N, int size, std::complex<DataType>* in, std::complex<DataType>* out){}

template<>
void ifftBlock<double>(int N, int size, std::complex<double>* in, std::complex<double>* out)
{

	int rank = 1;
	int length[] = {size};
	int howmany = N/size;
	int dist = size;
	int stride = 1;

	fftw_plan plan = fftw_plan_many_dft(
					rank,
					length,
					howmany,
					reinterpret_cast<fftw_complex*>(in),
					NULL,
					stride,
					dist,
					reinterpret_cast<fftw_complex*>(out),
					NULL,
					stride,
					dist,
					-1,
					FFTW_ESTIMATE);

	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<>
void ifftBlock<float>(int N, int size, std::complex<float>* in, std::complex<float>* out)
{

	int rank = 1;
	int length[] = {size};
	int howmany = N/size;
	int dist = size;
	int stride = 1;

	fftwf_plan plan = fftwf_plan_many_dft(
					rank,
					length,
					howmany,
					reinterpret_cast<fftwf_complex*>(in),
					NULL,
					stride,
					dist,
					reinterpret_cast<fftwf_complex*>(out),
					NULL,
					stride,
					dist,
					-1,
					FFTW_ESTIMATE);

	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
}

template<class DataType>
void ifftBlock_training(int N, int size, std::complex<DataType>* in, std::complex<DataType>* out){}

template<>
void 
ifftBlock_training<double>(int N, int size, std::complex<double>* in, std::complex<double>* out)
{

	int rank = 1;
	int length[] = {size};
	int howmany = N/size;
	int dist = size;
	int stride = 1;

	fftw_plan plan = fftw_plan_many_dft(
					rank,
					length,
					howmany,
					reinterpret_cast<fftw_complex*>(in),
					NULL,
					stride,
					dist,
					reinterpret_cast<fftw_complex*>(out),
					NULL,
					stride,
					dist,
					-1,
					FFTW_EXHAUSTIVE);

    fftw_export_wisdom_to_filename(&wisdom_path[0]);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<>
void ifftBlock_training<float>(int N, int size, std::complex<float>* in, std::complex<float>* out)
{

	int rank = 1;
	int length[] = {size};
	int howmany = N/size;
	int dist = size;
	int stride = 1;

	fftwf_plan plan = fftwf_plan_many_dft(
					rank,
					length,
					howmany,
					reinterpret_cast<fftwf_complex*>(in),
					NULL,
					stride,
					dist,
					reinterpret_cast<fftwf_complex*>(out),
					NULL,
					stride,
					dist,
					-1,
					FFTW_EXHAUSTIVE);

    fftwf_export_wisdom_to_filename(&wisdom_path[0]);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
}
///////////////////////////////////////////////////////////////////
//					 _      _____ _____ _____                    //
//					(_)_ __|  ___|  ___|_   _|                   //
//					| | '__| |_  | |_    | |                     //
//					| | |  |  _| |  _|   | |                     //
//					|_|_|  |_|   |_|     |_|                     //
///////////////////////////////////////////////////////////////////
template<class DataType>
void irfft(uint64_t N, std::complex<DataType>* in, DataType* out){}

template<>
void irfft<double>(uint64_t N, std::complex<double>* in, double* out)
{
	fftw_plan plan;
	plan = fftw_plan_dft_c2r_1d(
					N, 
					reinterpret_cast<fftw_complex*>(in), 
					out, 
					FFTW_ESTIMATE);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<>
void irfft<float>(uint64_t N, std::complex<float>* in, float* out)
{
	fftwf_plan plan;
	plan = fftwf_plan_dft_c2r_1d(
					N, 
					reinterpret_cast<fftwf_complex*>(in), 
					out, 
					FFTW_ESTIMATE);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
}

template<class DataType>
void irfft_training(uint64_t N, std::complex<DataType>* in, DataType* out){}

template<>
void irfft_training<double>(uint64_t N, std::complex<double>* in, double* out)
{
	fftw_plan plan;
	plan = fftw_plan_dft_c2r_1d(
					N, 
					reinterpret_cast<fftw_complex*>(in), 
					out, 
					FFTW_EXHAUSTIVE);
    fftw_export_wisdom_to_filename(&wisdom_path[0]);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<>
void irfft_training<float>(uint64_t N, std::complex<float>* in, float* out)
{
	fftwf_plan plan;
	plan = fftwf_plan_dft_c2r_1d(
					N, 
					reinterpret_cast<fftwf_complex*>(in), 
					out, 
					FFTW_EXHAUSTIVE);
    fftwf_export_wisdom_to_filename(&wisdom_path[0]);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
}

template<class DataType>
void irfftBlock(int N, int size, std::complex<DataType>* in, DataType* out){}

template<>
void irfftBlock<double>(int N, int size, std::complex<double>* in, double* out)
{

	int rank = 1;
	int length[] = {size};
	int howmany = N/size;
	int odist = size;
	int idist = size/2+1;
	int stride = 1;

	fftw_plan plan = fftw_plan_many_dft_c2r(
					rank,
					length,
					howmany,
					reinterpret_cast<fftw_complex*>(in),
					NULL,
					stride,
					idist,
					out,
					NULL,
					stride,
					odist,
					FFTW_ESTIMATE);

	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<>
void irfftBlock<float>(int N, int size, std::complex<float>* in, float* out)
{

	int rank = 1;
	int length[] = {size};
	int howmany = N/size;
	int odist = size;
	int idist = size/2+1;
	int stride = 1;

	fftwf_plan plan = fftwf_plan_many_dft_c2r(
					rank,
					length,
					howmany,
					reinterpret_cast<fftwf_complex*>(in),
					NULL,
					stride,
					idist,
					out,
					NULL,
					stride,
					odist,
					FFTW_ESTIMATE);

	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
}

template<class DataType>
void irfftBlock_training(int N, int size, std::complex<DataType>* in, DataType* out){}

template<>
void irfftBlock_training<double>(int N, int size, std::complex<double>* in, double* out)
{

	int rank = 1;
	int length[] = {size};
	int howmany = N/size;
	int odist = size;
	int idist = size/2+1;
	int stride = 1;

	fftw_plan plan = fftw_plan_many_dft_c2r(
					rank,
					length,
					howmany,
					reinterpret_cast<fftw_complex*>(in),
					NULL,
					stride,
					idist,
					out,
					NULL,
					stride,
					odist,
					FFTW_EXHAUSTIVE);

	fftw_export_wisdom_to_filename(&wisdom_path[0]);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<>
void irfftBlock_training<float>(int N, int size, std::complex<float>* in, float* out)
{

	int rank = 1;
	int length[] = {size};
	int howmany = N/size;
	int odist = size;
	int idist = size/2+1;
	int stride = 1;

	fftwf_plan plan = fftwf_plan_many_dft_c2r(
					rank,
					length,
					howmany,
					reinterpret_cast<fftwf_complex*>(in),
					NULL,
					stride,
					idist,
					out,
					NULL,
					stride,
					odist,
					FFTW_EXHAUSTIVE);

	fftwf_export_wisdom_to_filename(&wisdom_path[0]);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
}

///////////////////////////////////////////////////////////////////
//	  			 ___ _____ _   _ _____ ____  ____                //
//	 			/ _ \_   _| | | | ____|  _ \/ ___|               //
//			   | | | || | | |_| |  _| | |_) \___ \               //
//			   | |_| || | |  _  | |___|  _ < ___) |              //
//	 			\___/ |_| |_| |_|_____|_| \_\____/               //
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
		k = 2*(j%(Npad-1));
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
		k = 2*(j%(Npad-1));
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
		k = 2*(j%(Npad-1));
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
		k = 2*(j%(Npad-1));
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
