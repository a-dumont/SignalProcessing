void execute(fftw_plan plan)
{
	fftw_execute(plan);
}

void destroy_plan(fftw_plan plan)
{
	fftw_destroy_plan(plan);
}

void import_wisdom(std::string path)
{
    fftw_import_wisdom_from_filename(&path[0]);
}

void export_wisdom(std::string path)
{
    fftw_export_wisdom_to_filename(&path[0]);
}

template<class DataType>
fftw_plan FFT_plan(int n, DataType* in, DataType* out)
{
	fftw_plan plan;
	plan = fftw_plan_dft_1d(
					n, 
					reinterpret_cast<fftw_complex*>(in), 
					reinterpret_cast<fftw_complex*>(out), 
					FFTW_FORWARD, 
					FFTW_ESTIMATE);
	return plan;
}

template<class DataType>
fftw_plan FFT_Block_plan(int n, int N, DataType* in, DataType* out)
{

	int rank = 1;
	int length[] = {N};
	int howmany = n/N;
	int dist = N;
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
	return plan;
}

template<class DataType>
fftw_plan iFFT_plan(int n, DataType* in, DataType* out)
{
	fftw_plan plan;
	plan = fftw_plan_dft_1d(
					n, 
					reinterpret_cast<fftw_complex*>(in), 
					reinterpret_cast<fftw_complex*>(out), 
					FFTW_BACKWARD, 
					FFTW_ESTIMATE);
	return plan;
}

template<class DataTypeIn, class DataTypeOut>
fftw_plan rFFT_plan(int n, DataTypeIn* in, DataTypeOut* out)
{
	fftw_plan plan;
	plan = fftw_plan_dft_r2c_1d(
					n, 
					in, 
					reinterpret_cast<fftw_complex*>(out), 
					FFTW_ESTIMATE);
	return plan;
}

template<class DataTypeIn, class DataTypeOut>
fftw_plan rFFT_Block_plan(int n, int N, DataTypeIn* in, DataTypeOut* out)
{

	int rank = 1;
	int length[] = {N};
	int howmany = n/N;
	int idist = N;
	int odist = N/2+1;
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
	return plan;
}

template<class DataTypeIn, class DataTypeOut>
fftw_plan irFFT_plan(int n, DataTypeIn* in, DataTypeOut* out)
{
	fftw_plan plan;
	plan = fftw_plan_dft_c2r_1d(
					n, 
					reinterpret_cast<fftw_complex*>(in), 
					out, 
					FFTW_ESTIMATE|FFTW_PRESERVE_INPUT);
	return plan;
}



