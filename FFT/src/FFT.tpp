template<class DataType>
void FFT(int n, DataType* in, DataType* out)
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

template<class DataType>
void FFT_Parallel(int n, DataType* in, DataType* out, int nthreads)
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

template<class DataType>
void FFT_Block(int n, int N, DataType* in, DataType* out)
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

template<class DataType>
void FFT_Block_Parallel(int n, int N, DataType* in, DataType* out, int nthreads)
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

template<class DataType>
void FFT_Block_Parallel2(int n, int N, DataType* in, DataType* out)
{

	int howmany = n/N;
	fftw_import_wisdom_from_filename(&wisdom_path[0]);
	
	fftw_plan plan;
	plan = fftw_plan_dft_1d(
					N, 
					reinterpret_cast<fftw_complex*>(in), 
					reinterpret_cast<fftw_complex*>(out), 
					FFTW_FORWARD, 
					FFTW_ESTIMATE);

	#pragma omp parallel for
	for(int i=0;i<howmany;i++)
	{
		fftw_execute_dft(plan, 
						reinterpret_cast<fftw_complex*>(in+i*N), 
						reinterpret_cast<fftw_complex*>(out+i*N));
	}
	fftw_export_wisdom_to_filename(&wisdom_path[0]);
	fftw_destroy_plan(plan);
	fftw_forget_wisdom();
	fftw_cleanup();
}


template<class DataType>
void iFFT(int n, DataType* in, DataType* out)
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

template<class DataTypeIn, class DataTypeOut>
void rFFT(int n, DataTypeIn* in, DataTypeOut* out)
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

template<class DataTypeIn, class DataTypeOut>
void rFFT_Block(int n, int N, DataTypeIn* in, DataTypeOut* out)
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

template<class DataTypeIn, class DataTypeOut>
void irFFT(int n, DataTypeIn* in, DataTypeOut* out)
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



