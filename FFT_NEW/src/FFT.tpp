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
fftBlock(uint64_t N, uint64_t size, std::complex<DataType>* in, std::complex<DataType>* out){}

template<>
fftBlock<double>(uint64_t N, uint64_t size, std::complex<double>* in, std::complex<double>* out)
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

    //fftw_export_wisdom_to_filename(&wisdom_path[0]);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<>
fftBlock<float>(uint64_t N, uint64_t size, std::complex<float>* in, std::complex<float>* out)
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
ifftBlock(uint64_t N, uint64_t size, std::complex<DataType>* in, std::complex<DataType>* out){}

template<>
ifftBlock<double>(uint64_t N, uint64_t size, std::complex<double>* in, std::complex<double>* out)
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

    //fftw_export_wisdom_to_filename(&wisdom_path[0]);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<>
ifftBlock<float>(uint64_t N, uint64_t size, std::complex<float>* in, std::complex<float>* out)
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

    //fftw_export_wisdom_to_filename(&wisdom_path[0]);
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

