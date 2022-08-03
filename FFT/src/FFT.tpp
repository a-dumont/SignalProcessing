//Full size fft
template<class DataType>
void FFT(int n, DataType* in, DataType* out){}

template<>
void FFT<dbl_complex>(int n, dbl_complex* in, dbl_complex* out)
{
	fftw_plan plan;
	plan = fftw_plan_dft_1d(
					n, 
					reinterpret_cast<fftw_complex*>(in), 
					reinterpret_cast<fftw_complex*>(out), 
					FFTW_FORWARD, 
					FFTW_ESTIMATE);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<>
void FFT<flt_complex>(int n, flt_complex* in, flt_complex* out)
{
	fftwf_plan plan;
	plan = fftwf_plan_dft_1d(
					n, 
					reinterpret_cast<fftwf_complex*>(in), 
					reinterpret_cast<fftwf_complex*>(out), 
					FFTW_FORWARD, 
					FFTW_ESTIMATE);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
}

//Parallel full size fft
template<class DataType>
void FFT_Parallel(int n, DataType* in, DataType* out, int nthreads){}

template<>
void FFT_Parallel<dbl_complex>(int n, dbl_complex* in, dbl_complex* out, int nthreads) 
{
	fftw_plan plan;
	int threads_init = fftw_init_threads();
	if (threads_init == 0)
	{
		throw std::runtime_error("Cannot initialize threads.");
	}
	omp_set_num_threads(nthreads);
	fftw_plan_with_nthreads(omp_get_max_threads());
	plan = fftw_plan_dft_1d(
					n, 
					reinterpret_cast<fftw_complex*>(in), 
					reinterpret_cast<fftw_complex*>(out), 
					FFTW_FORWARD, 
					FFTW_ESTIMATE);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
	fftw_cleanup_threads();
}

template<>
void FFT_Parallel<flt_complex>(int n, flt_complex* in, flt_complex* out, int nthreads) 
{
	fftwf_plan plan;
	int threads_init = fftwf_init_threads();
	if (threads_init == 0)
	{
		throw std::runtime_error("Cannot initialize threads.");
	}
	omp_set_num_threads(nthreads);
	fftwf_plan_with_nthreads(omp_get_max_threads());
	plan = fftwf_plan_dft_1d(
					n, 
					reinterpret_cast<fftwf_complex*>(in), 
					reinterpret_cast<fftwf_complex*>(out), 
					FFTW_FORWARD, 
					FFTW_ESTIMATE);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
	fftwf_cleanup_threads();
}

//FFTs of size N
template<class DataType>
void FFT_Block(int n, int N, DataType* in, DataType* out){}

template<>
void FFT_Block<dbl_complex>(int n, int N, dbl_complex* in, dbl_complex* out)
{

	int rank = 1;
	int length[] = {N};
	int howmany = n/N;
	int dist = N;
	int stride = 1;

    fftw_import_wisdom_from_filename(&wisdom_path[0]);

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
					FFTW_MEASURE);

    fftw_export_wisdom_to_filename(&wisdom_path[0]);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
	fftw_forget_wisdom();
	fftw_cleanup();
}

template<>
void FFT_Block<flt_complex>(int n, int N, flt_complex* in, flt_complex* out)
{

	int rank = 1;
	int length[] = {N};
	int howmany = n/N;
	int dist = N;
	int stride = 1;

    fftwf_import_wisdom_from_filename(&wisdom_path[0]);

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
					FFTW_MEASURE);

    fftwf_export_wisdom_to_filename(&wisdom_path[0]);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
	fftwf_forget_wisdom();
	fftwf_cleanup();
}

template<class DataType>
void FFT_Block_Parallel(int n, int N, DataType* in, DataType* out, int nthreads){}

template<>
void FFT_Block_Parallel<dbl_complex>(int n, int N, dbl_complex* in, dbl_complex* out, int nthreads)
{

	int rank = 1;
	int length[] = {N};
	int howmany = n/N;
	int dist = N;
	int stride = 1;
	
	int threads_init = fftw_init_threads();
	if (threads_init == 0)
	{
		throw std::runtime_error("Cannot initialize threads.");
	}
	omp_set_num_threads(nthreads);
	fftw_plan_with_nthreads(omp_get_max_threads());
	
	fftw_import_wisdom_from_filename(&wisdom_parallel_path[0]);

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
					FFTW_MEASURE);

    fftw_export_wisdom_to_filename(&wisdom_parallel_path[0]);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
	fftw_forget_wisdom();
	fftw_cleanup_threads();
}

template<>
void FFT_Block_Parallel<flt_complex>(int n, int N, flt_complex* in, flt_complex* out, int nthreads)
{

	int rank = 1;
	int length[] = {N};
	int howmany = n/N;
	int dist = N;
	int stride = 1;
	
	int threads_init = fftwf_init_threads();
	if (threads_init == 0)
	{
		throw std::runtime_error("Cannot initialize threads.");
	}
	omp_set_num_threads(nthreads);
	fftwf_plan_with_nthreads(omp_get_max_threads());
	
	fftwf_import_wisdom_from_filename(&wisdom_parallel_path[0]);

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
					FFTW_MEASURE);

    fftwf_export_wisdom_to_filename(&wisdom_parallel_path[0]);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
	fftwf_forget_wisdom();
	fftwf_cleanup_threads();
}

template<class DataType>
void iFFT(int n, DataType* in, DataType* out){}

template<>
void iFFT<dbl_complex>(int n, dbl_complex* in, dbl_complex* out)
{
	fftw_plan plan;
	plan = fftw_plan_dft_1d(
					n, 
					reinterpret_cast<fftw_complex*>(in), 
					reinterpret_cast<fftw_complex*>(out), 
					FFTW_BACKWARD, 
					FFTW_ESTIMATE);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<>
void iFFT<flt_complex>(int n, flt_complex* in, flt_complex* out)
{
	fftwf_plan plan;
	plan = fftwf_plan_dft_1d(
					n, 
					reinterpret_cast<fftwf_complex*>(in), 
					reinterpret_cast<fftwf_complex*>(out), 
					FFTW_BACKWARD, 
					FFTW_ESTIMATE);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
}

template<class DataType>
void rFFT(int n, DataType* in, std::complex<DataType>* out){}

template<>
void rFFT<double>(int n, double* in, std::complex<double>* out)
{
	fftw_plan plan;
	plan = fftw_plan_dft_r2c_1d(
					n, 
					in, 
					reinterpret_cast<fftw_complex*>(out), 
					FFTW_ESTIMATE);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<>
void rFFT<float>(int n, float* in, std::complex<float>* out)
{
	fftwf_plan plan;
	plan = fftwf_plan_dft_r2c_1d(
					n, 
					in, 
					reinterpret_cast<fftwf_complex*>(out), 
					FFTW_ESTIMATE);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
}

template<class DataType>
void rFFT_Block(int n, int N, DataType* in, std::complex<DataType>* out){}

template<>
void rFFT_Block<double>(int n, int N, double* in, std::complex<double>* out)
{

	int rank = 1;
	int length[] = {N};
	int howmany = n/N;
	int idist = N;
	int odist = N/2+1;
	int stride = 1;

    fftw_import_wisdom_from_filename(&wisdom_path[0]);

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
void rFFT_Block<float>(int n, int N, float* in, std::complex<float>* out){}
{

	int rank = 1;
	int length[] = {N};
	int howmany = n/N;
	int idist = N;
	int odist = N/2+1;
	int stride = 1;

    fftwf_import_wisdom_from_filename(&wisdom_path[0]);

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

template<class DataType>
void irFFT(int n, std::complex<DataType>* in, DataType* out){}

template<>
void irFFT<double>(int n, std::complex<double>* in, double* out)
{
	fftw_plan plan;
	plan = fftw_plan_dft_c2r_1d(
					n, 
					reinterpret_cast<fftw_complex*>(in), 
					out, 
					FFTW_ESTIMATE|FFTW_PRESERVE_INPUT);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<>
void irFFT<float>(int n, std::complex<float>* in, float* out)
{
	fftwf_plan plan;
	plan = fftwf_plan_dft_c2r_1d(
					n, 
					reinterpret_cast<fftwf_complex*>(in), 
					out, 
					FFTW_ESTIMATE|FFTW_PRESERVE_INPUT);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
}
