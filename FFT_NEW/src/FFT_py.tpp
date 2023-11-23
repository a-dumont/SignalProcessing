///////////////////////////////////////////////////////////////////
//						 _____ _____ _____                       //
//						|  ___|  ___|_   _|                      //
//						| |_  | |_    | |                        //
//						|  _| |  _|   | |                        //
//						|_|   |_|     |_|                        //
///////////////////////////////////////////////////////////////////
#include <complex>
template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
fft_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	uint64_t N = buf_in.size;
	
	std::complex<DataType>* in = (std::complex<DataType>*) buf_in.ptr;
	std::complex<DataType>* out = (std::complex<DataType>*) fftw_malloc(2*N*sizeof(DataType));
	std::memcpy(out,in,2*N*sizeof(DataType));

	fft<DataType>(N, out, out);

	py::capsule free_when_done( out, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{N},
		{2*sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
fft_training_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	uint64_t N = buf_in.size;
	
	std::complex<DataType>* in = (std::complex<DataType>*) buf_in.ptr;
	std::complex<DataType>* out = (std::complex<DataType>*) fftw_malloc(2*N*sizeof(DataType));
	std::memcpy(out,in,2*N*sizeof(DataType));

	fft_training<DataType>(N, out, out);

	py::capsule free_when_done( out, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{N},
		{2*sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
fft_pad_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in, uint64_t size)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1){throw std::runtime_error("U dumbdumb dimension must be 1.");}		

	uint64_t N = std::min((uint64_t) buf_in.size, size);
	
	std::complex<DataType>* in = (std::complex<DataType>*) buf_in.ptr;
	std::complex<DataType>* out = (std::complex<DataType>*) fftw_malloc(2*size*sizeof(DataType));
	std::memset((void*) out,0,2*size*sizeof(DataType));
	std::memcpy(out,in,2*N*sizeof(DataType));

	fft<DataType>(size, out, out);

	py::capsule free_when_done( out, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{size},
		{2*sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style>
fftBlock_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in, uint64_t size)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	uint64_t N = buf_in.size;
	uint64_t howmany = N/size;
	if(howmany*size != N){howmany += 1;}
	uint64_t Npad = size*howmany;

	std::complex<DataType>* in = (std::complex<DataType>*) buf_in.ptr;
	std::complex<DataType>* out = (std::complex<DataType>*) fftw_malloc(2*Npad*sizeof(DataType));
	std::memset((DataType*)(out+N),0.0,2*(Npad-N)*sizeof(DataType));
	std::memcpy(out,in,2*N*sizeof(DataType));

	fftBlock((int) Npad,(int) size, out, out);

	py::capsule free_when_done( out, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{Npad},
		{2*sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style>
fftBlock_training_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in, uint64_t size)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	uint64_t N = buf_in.size;
	uint64_t howmany = N/size;
	if(howmany*size != N){howmany += 1;}
	uint64_t Npad = size*howmany;

	std::complex<DataType>* in = (std::complex<DataType>*) buf_in.ptr;
	std::complex<DataType>* out = (std::complex<DataType>*) fftw_malloc(2*Npad*sizeof(DataType));
	std::memset((DataType*)(out+N),0.0,2*(Npad-N)*sizeof(DataType));
	std::memcpy(out,in,2*N*sizeof(DataType));

	fftBlock_training((int) Npad,(int) size, out, out);

	py::capsule free_when_done( out, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{Npad},
		{2*sizeof(DataType)},
		out,
		free_when_done	
	);
}

class FFT_py
{
	public:

		fftw_plan plan;
		fftwf_plan planf;
		std::complex<double>* in;
		std::complex<float>* inf;
		uint64_t N;

		FFT_py(uint64_t N_in)
		{
			N = N_in;

			in = (std::complex<double>*) fftw_malloc(2*N*sizeof(double));
			inf = (std::complex<float>*) fftwf_malloc(2*N*sizeof(float));
			
			plan = fftw_plan_dft_1d(
							N,
							reinterpret_cast<fftw_complex*>(in),
							reinterpret_cast<fftw_complex*>(in),
							FFTW_FORWARD,
							FFTW_ESTIMATE);

			planf = fftwf_plan_dft_1d(
							N,
							reinterpret_cast<fftwf_complex*>(inf),
							reinterpret_cast<fftwf_complex*>(inf),
							FFTW_FORWARD,
							FFTW_ESTIMATE);
		}
		
		~FFT_py()
		{
			fftw_free(in);
			fftwf_free(inf);
			fftw_destroy_plan(plan);
			fftwf_destroy_plan(planf);
		}

		uint64_t getSize(){return N;}
		
		std::tuple<double,double> benchmark(uint64_t n)
		{
			auto time1 = Clock::now();
			for(uint64_t i=0;i<n;i++){fftw_execute(plan);}
			auto time2 = Clock::now();
			
			auto timef1 = Clock::now();
			for(uint64_t i=0;i<n;i++){fftwf_execute(planf);}
			auto timef2 = Clock::now();
			
			double time;
			time = std::chrono::duration_cast<std::chrono::microseconds>(time2-time1).count();

			double timef;
			timef = std::chrono::duration_cast<std::chrono::microseconds>(timef2-timef1).count();
			py::print("Time for double precision FFT of size ",N,": ",time/n," us","sep"_a="");
			py::print("Time for single precision FFT of size ",N,": ",timef/n," us","sep"_a="");
			return std::make_tuple<double,double>(time/n,timef/n);
		}

		py::array_t<std::complex<double>,1> 
		fft(py::array_t<std::complex<double>,1> py_in)
		{
			py::buffer_info buf_in = py_in.request();

			if (buf_in.ndim != 1)
			{
			throw std::runtime_error("U dumbdumb dimension must be 1.");
			}	

			std::memcpy(in,(std::complex<double>*) buf_in.ptr, 2*N*sizeof(double));
			fftw_execute(plan);
		
			std::complex<double>* out = (std::complex<double>*) malloc(2*N*sizeof(double));
			std::memcpy(out,in,2*N*sizeof(double));

			py::capsule free_when_done( out, free );
			return py::array_t<std::complex<double>,1>
			(
			{N},
			{2*sizeof(double)},
			out,
			free_when_done	
			);
		}

		py::array_t<std::complex<float>,1> 
		fftf(py::array_t<std::complex<float>,1> py_in)
		{
			py::buffer_info buf_in = py_in.request();

			if (buf_in.ndim != 1)
			{
			throw std::runtime_error("U dumbdumb dimension must be 1.");
			}	

			std::memcpy(inf,(std::complex<float>*) buf_in.ptr, 2*N*sizeof(float));
			fftwf_execute(planf);
		
			std::complex<float>* out = (std::complex<float>*) malloc(2*N*sizeof(float));
			std::memcpy(out,inf,2*N*sizeof(float));

			py::capsule free_when_done( out, free );
			return py::array_t<std::complex<float>,1>
			(
			{N},
			{2*sizeof(float)},
			out,
			free_when_done	
			);
		}
};

class FFT_Block_py
{
	public:

		fftw_plan plan, plan2;
		fftwf_plan planf, plan2f;
		std::complex<double> *in;
		std::complex<float> *inf;
		uint64_t N, size, howmany, Npad, threads;
		int length[1];
		uint64_t* transfer_size;

		FFT_Block_py(uint64_t N_in, uint64_t size_in)
		{
			N = N_in;
			size = size_in;
			howmany = N/size;
			if(howmany*size != N){howmany += 1;}
			Npad = size*howmany;
			length[0] = (int) size;

			#ifdef _WIN32_WINNT
				threads = (uint64_t) omp.omp_get_max_threads()*GetActiveProcessorGroupCount();
			#else
				threads = omp_get_max_threads();
			#endif

			threads = std::min(threads,(uint64_t) 64);

			in = (std::complex<double>*) fftw_malloc(2*Npad*sizeof(double));
			inf = (std::complex<float>*) fftwf_malloc(2*Npad*sizeof(float));
			
			plan = fftw_plan_many_dft(
							1,
							length,
							howmany/threads,
							reinterpret_cast<fftw_complex*>(in),
							NULL,
							1,
							(int) size,
							reinterpret_cast<fftw_complex*>(in),
							NULL,
							1,
							(int) size,
							1,
							FFTW_ESTIMATE);

			planf = fftwf_plan_many_dft(
							1,
							length,
							howmany/threads,
							reinterpret_cast<fftwf_complex*>(inf),
							NULL,
							1,
							(int) size,
							reinterpret_cast<fftwf_complex*>(inf),
							NULL,
							1,
							(int) size,
							1,
							FFTW_ESTIMATE);

			transfer_size = (uint64_t*) malloc(threads*sizeof(uint64_t));
			for(uint64_t i=0;i<(threads-1);i++){transfer_size[i]=size*(howmany/threads);}
			transfer_size[threads-1] = N-(threads-1)*transfer_size[0];
		}
		
		~FFT_Block_py()
		{
			free(transfer_size);
			fftw_free(in);
			fftwf_free(inf);
			fftw_destroy_plan(plan);
			fftwf_destroy_plan(planf);
		}

		uint64_t getSize(){return size;}
		uint64_t getN(){return Npad;}
		uint64_t getHowmany(){return Npad;}
		
		std::tuple<double,double> benchmark(uint64_t n)
		{
			auto time1 = Clock::now();
			for(uint64_t i=0;i<n;i++)
			{
				#pragma omp parallel for
				for(uint64_t j=0;j<threads;j++)
				{
					manage_thread_affinity();
					fftw_execute(plan);
				}
			}
			auto time2 = Clock::now();
			
			auto timef1 = Clock::now();
			for(uint64_t i=0;i<n;i++)
			{
				#pragma omp parallel for
				for(uint64_t j=0;j<threads;j++)
				{
					manage_thread_affinity();
					fftwf_execute(planf);
				}
			}
			auto timef2 = Clock::now();
			
			double time;
			time = std::chrono::duration_cast<std::chrono::microseconds>(time2-time1).count();

			double timef;
			timef = std::chrono::duration_cast<std::chrono::microseconds>(timef2-timef1).count();
			py::print("Time for ",howmany,
							" double precision FFT of size ",size,": ",time/n," us","sep"_a="");
			py::print("Time for ",howmany,
							" single precision FFT of size ",size,": ",timef/n," us","sep"_a="");
			return std::make_tuple<double,double>(time/n,timef/n);
		}

		py::array_t<std::complex<double>,1> 
		fftBlock(py::array_t<std::complex<double>,1> py_in)
		{
			py::buffer_info buf_in = py_in.request();
			std::complex<double>* py_ptr = (std::complex<double>*) buf_in.ptr;

			if (buf_in.ndim != 1){throw std::runtime_error("U dumbdumb dimension must be 1.");}	
			if ((uint64_t) buf_in.size > Npad)
			{throw std::runtime_error("U dumbdumb input too long.");}

			std::complex<double>* out;
			out = (std::complex<double>*) fftw_malloc(2*Npad*sizeof(double));
			std::memset((double*)(out+N),0.0,2*(Npad-N)*sizeof(double));

			#pragma omp parallel for
			for(uint64_t i=0;i<threads;i++)
			{
				manage_thread_affinity();
				std::memcpy(out+i*size*(howmany/threads),
								py_ptr+i*size*(howmany/threads), 
								2*transfer_size[i]*sizeof(double));
				fftw_execute_dft(plan,
								reinterpret_cast<fftw_complex*>(out+i*size*(howmany/threads)),
								reinterpret_cast<fftw_complex*>(out+i*size*howmany/threads));
			}

			if(N!=Npad){::fftBlock<double>(Npad-N,size,py_ptr,out);}

			py::capsule free_when_done( out, fftw_free );
			return py::array_t<std::complex<double>,py::array::c_style>
			(
			{Npad},
			{2*sizeof(double)},
			out,
			free_when_done	
			);
		}

		py::array_t<std::complex<float>,py::array::c_style> 
		fftBlockf(py::array_t<std::complex<float>,py::array::c_style> py_in)
		{
			py::buffer_info buf_in = py_in.request();
			std::complex<float>* py_ptr = (std::complex<float>*) buf_in.ptr;

			if (buf_in.ndim != 1){throw std::runtime_error("U dumbdumb dimension must be 1.");}	
			if ((uint64_t) buf_in.size > Npad)
			{throw std::runtime_error("U dumbdumb input too long.");}

			std::complex<float>* out;
			out = (std::complex<float>*) fftwf_malloc(2*Npad*sizeof(float));
			std::memset((float*)(out+N),0.0,2*(Npad-N)*sizeof(float));

			#pragma omp parallel for
			for(uint64_t i=0;i<threads;i++)
			{
				manage_thread_affinity();
				std::memcpy(out+i*size*(howmany/threads),
								py_ptr+i*size*(howmany/threads), 
								2*transfer_size[i]*sizeof(float));
				fftwf_execute_dft(planf,
								reinterpret_cast<fftwf_complex*>(out+i*size*(howmany/threads)),
								reinterpret_cast<fftwf_complex*>(out+i*size*howmany/threads));
			}

			if(N!=Npad)
			{
				::fftBlock<float>(Npad-N,size,py_ptr,out+threads*size*(howmany/threads));
			}

			py::capsule free_when_done( out, fftw_free );
			return py::array_t<std::complex<float>,py::array::c_style>
			(
			{Npad},
			{2*sizeof(float)},
			out,
			free_when_done	
			);
		}
};

///////////////////////////////////////////////////////////////////
//						      _____ _____ _____                  //
//						 _ __|  ___|  ___|_   _|                 //
//						| '__| |_  | |_    | |                   //
//						| |  |  _| |  _|   | |                   //
//						|_|  |_|   |_|     |_|                   //
///////////////////////////////////////////////////////////////////
template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
rfft_py(py::array_t<DataType,py::array::c_style> py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	uint64_t N = buf_in.size;
	
	DataType* in = (DataType*) buf_in.ptr;
	std::complex<DataType>* out = (std::complex<DataType>*) fftw_malloc((N+2)*sizeof(DataType));
	std::memcpy((void*) out,in,N*sizeof(DataType));
	out[N] = 0.0;

	rfft<DataType>(N, reinterpret_cast<DataType*>(out), out);

	py::capsule free_when_done( out, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{N/2+1},
		{2*sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
rfft_training_py(py::array_t<DataType,py::array::c_style> py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	uint64_t N = buf_in.size;
	
	DataType* in = (DataType*) buf_in.ptr;
	std::complex<DataType>* out = (std::complex<DataType>*) fftw_malloc(2*(N+2)*sizeof(DataType));
	std::memcpy((void*) out,in,2*N*sizeof(DataType));
	out[N] = 0.0;

	rfft_training<DataType>(N, reinterpret_cast<DataType*>(out), out);

	py::capsule free_when_done( out, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{N/2+1},
		{2*sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style>
rfftBlock_py(py::array_t<DataType,py::array::c_style> py_in, uint64_t size)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	uint64_t N = buf_in.size;
	uint64_t howmany = N/size;
	if(howmany*size != N){howmany += 1;}
	uint64_t Npad = size*howmany;
	uint64_t Nout = 2*(size/2+1)*howmany;

	DataType* py_ptr = (DataType*) buf_in.ptr;
	DataType* in = (DataType*) fftw_malloc(Npad*sizeof(DataType));
	std::complex<DataType>* out = (std::complex<DataType>*) fftw_malloc(Nout*sizeof(DataType));
	std::memset(in+N,0.0,(Npad-N)*sizeof(DataType));
	std::memcpy(in,py_ptr,N*sizeof(DataType));

	rfftBlock((int) Npad,(int) size, in, out);

	free(in);

	py::capsule free_when_done( out, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{Nout/2},
		{2*sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style>
rfftBlock_training_py(py::array_t<DataType,py::array::c_style> py_in, uint64_t size)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	uint64_t N = buf_in.size;
	uint64_t howmany = N/size;
	if(howmany*size != N){howmany += 1;}
	uint64_t Npad = size*howmany;
	uint64_t Nout = 2*(size/2+1)*howmany;

	DataType* py_ptr = (DataType*) buf_in.ptr;
	DataType* in = (DataType*) fftw_malloc(Npad*sizeof(DataType));
	std::complex<DataType>* out = (std::complex<DataType>*) fftw_malloc(Nout*sizeof(DataType));
	std::memset(in+N,0.0,(Npad-N)*sizeof(DataType));
	std::memcpy(in,py_ptr,N*sizeof(DataType));

	rfftBlock_training((int) Npad,(int) size, in, out);

	free(in);

	py::capsule free_when_done( out, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{Nout/2},
		{2*sizeof(DataType)},
		out,
		free_when_done	
	);
}

class RFFT_py
{
	public:

		fftw_plan plan;
		fftwf_plan planf;
		double* in;
		float* inf;
		uint64_t N;

		RFFT_py(uint64_t N_in)
		{
			N = N_in;

			in = (double*) fftw_malloc((N+2)*sizeof(double));
			inf = (float*) fftwf_malloc((N+2)*sizeof(float));
			
			plan = fftw_plan_dft_r2c_1d(N,in,reinterpret_cast<fftw_complex*>(in),FFTW_ESTIMATE);
			planf=fftwf_plan_dft_r2c_1d(N,inf,reinterpret_cast<fftwf_complex*>(inf),FFTW_ESTIMATE);
		}
		
		~RFFT_py()
		{
			fftw_free(in);
			fftwf_free(inf);
			fftw_destroy_plan(plan);
			fftwf_destroy_plan(planf);
		}

		uint64_t getSize(){return N;}
		
		std::tuple<double,double> benchmark(uint64_t n)
		{
			auto time1 = Clock::now();
			for(uint64_t i=0;i<n;i++){fftw_execute(plan);}
			auto time2 = Clock::now();
			
			auto timef1 = Clock::now();
			for(uint64_t i=0;i<n;i++){fftwf_execute(planf);}
			auto timef2 = Clock::now();
			
			double time;
			time = std::chrono::duration_cast<std::chrono::microseconds>(time2-time1).count();

			double timef;
			timef = std::chrono::duration_cast<std::chrono::microseconds>(timef2-timef1).count();
			py::print("Time for double precision rFFT of size ",N,": ",time/n," us","sep"_a="");
			py::print("Time for single precision rFFT of size ",N,": ",timef/n," us","sep"_a="");
			return std::make_tuple<double,double>(time/n,timef/n);
		}

		py::array_t<std::complex<double>,1> 
		rfft(py::array_t<double,1> py_in)
		{
			py::buffer_info buf_in = py_in.request();

			if (buf_in.ndim != 1)
			{
			throw std::runtime_error("U dumbdumb dimension must be 1.");
			}	

			std::memcpy(in,(double*) buf_in.ptr, N*sizeof(double));
			fftw_execute(plan);
		
			std::complex<double>* out = (std::complex<double>*) malloc((N+2)*sizeof(double));
			std::memcpy((void*) out,in,(N+2)*sizeof(double));

			py::capsule free_when_done( out, free );
			return py::array_t<std::complex<double>,1>
			(
			{N/2+1},
			{2*sizeof(double)},
			out,
			free_when_done	
			);
		}

		py::array_t<std::complex<float>,1> 
		rfftf(py::array_t<float,1> py_in)
		{
			py::buffer_info buf_in = py_in.request();

			if (buf_in.ndim != 1)
			{
			throw std::runtime_error("U dumbdumb dimension must be 1.");
			}	

			std::memcpy(inf,(float*) buf_in.ptr, N*sizeof(float));
			fftwf_execute(planf);
		
			std::complex<float>* out = (std::complex<float>*) malloc((N+2)*sizeof(float));
			std::memcpy((void*) out,inf,(N+2)*sizeof(float));

			py::capsule free_when_done( out, free );
			return py::array_t<std::complex<float>,1>
			(
			{N/2+1},
			{2*sizeof(float)},
			out,
			free_when_done	
			);
		}
};

class RFFT_Block_py
{
	public:

		fftw_plan plan, plan2;
		fftwf_plan planf, plan2f;
		double *in, *out;
		float *inf, *outf;
		uint64_t N, size, howmany, Npad, threads;
		int length[1];
		uint64_t* transfer_size;

		RFFT_Block_py(uint64_t N_in, uint64_t size_in)
		{
			N = N_in;
			size = size_in;
			howmany = N/size;
			if(howmany*size != N){howmany += 1;}
			Npad = size*howmany;
			length[0] = (int) size;

			#ifdef _WIN32_WINNT
				threads = (uint64_t) omp.omp_get_max_threads()*GetActiveProcessorGroupCount();
			#else
				threads = omp_get_max_threads();
			#endif

			threads = std::min(threads,(uint64_t) 64);

			in = (double*) fftw_malloc(size*howmany*sizeof(double));
			inf = (float*) fftwf_malloc(size*howmany*sizeof(float));
			
			out = (double*) fftw_malloc((size+2)*howmany*sizeof(double));
			outf = (float*) fftwf_malloc((size+2)*howmany*sizeof(float));
			
			plan = fftw_plan_many_dft_r2c(
							1,
							length,
							howmany/threads,
							in,
							NULL,
							1,
							(int) size,
							reinterpret_cast<fftw_complex*>(out),
							NULL,
							1,
							(int) size/2+1,
							FFTW_ESTIMATE);

			planf = fftwf_plan_many_dft_r2c(
							1,
							length,
							howmany/threads,
							inf,
							NULL,
							1,
							(int) size,
							reinterpret_cast<fftwf_complex*>(outf),
							NULL,
							1,
							(int) size/2+1,
							FFTW_ESTIMATE);

			transfer_size = (uint64_t*) malloc(threads*sizeof(uint64_t));
			for(uint64_t i=0;i<(threads-1);i++){transfer_size[i]=size*(howmany/threads);}
			transfer_size[threads-1] = N-(threads-1)*transfer_size[0];
		}
		
		~RFFT_Block_py()
		{
			free(transfer_size);
			fftw_free(in);
			fftwf_free(inf);
			fftw_destroy_plan(plan);
			fftwf_destroy_plan(planf);
		}

		uint64_t getSize(){return size;}
		uint64_t getN(){return Npad;}
		uint64_t getHowmany(){return Npad;}
		
		std::tuple<double,double> benchmark(uint64_t n)
		{
			auto time1 = Clock::now();
			for(uint64_t i=0;i<n;i++)
			{
				#pragma omp parallel for
				for(uint64_t j=0;j<threads;j++)
				{
					manage_thread_affinity();
					fftw_execute(plan);
				}
			}
			auto time2 = Clock::now();
			
			auto timef1 = Clock::now();
			for(uint64_t i=0;i<n;i++)
			{
				#pragma omp parallel for
				for(uint64_t j=0;j<threads;j++)
				{
					manage_thread_affinity();
					fftwf_execute(planf);
				}
			}
			auto timef2 = Clock::now();
			
			double time;
			time = std::chrono::duration_cast<std::chrono::microseconds>(time2-time1).count();

			double timef;
			timef = std::chrono::duration_cast<std::chrono::microseconds>(timef2-timef1).count();
			py::print("Time for ",howmany,
							" double precision rFFT of size ",size,": ",time/n," us","sep"_a="");
			py::print("Time for ",howmany,
							" single precision rFFT of size ",size,": ",timef/n," us","sep"_a="");
			return std::make_tuple<double,double>(time/n,timef/n);
		}

		py::array_t<std::complex<double>,1> 
		rfftBlock(py::array_t<double,1> py_in)
		{
			py::buffer_info buf_in = py_in.request();
			double* py_ptr = (double*) buf_in.ptr;

			if (buf_in.ndim != 1){throw std::runtime_error("U dumbdumb dimension must be 1.");}	
			if ((uint64_t) buf_in.size > Npad)
			{throw std::runtime_error("U dumbdumb input too long.");}

			std::complex<double>* out;
			out = (std::complex<double>*) fftw_malloc(howmany*(size+2)*sizeof(double));
			std::memset((double*)(out+N),0.0,(howmany*(size+2)-N)*sizeof(double));

			#pragma omp parallel for
			for(uint64_t i=0;i<threads;i++)
			{
				manage_thread_affinity();
				//double* temp_out = ((double*) out)+i*(size+2)*(transfer_size[0]/size);
				//for(uint64_t j=0;j<transfer_size[i];j++)
				//{
					//temp_out[j+2*(j/size)] = py_ptr[j+i*transfer_size[0]];
				//}
				std::memcpy(in,py_ptr,transfer_size[i]*sizeof(double));
				fftw_execute_dft_r2c(plan,
								in,
								reinterpret_cast<fftw_complex*>(out));
			}

			if(N!=Npad){::rfftBlock<double>(Npad-N,size,py_ptr,out);}

			py::capsule free_when_done( out, fftw_free );
			return py::array_t<std::complex<double>,py::array::c_style>
			(
			{howmany*(size/2+1)},
			{2*sizeof(double)},
			out,
			free_when_done	
			);
		}

		py::array_t<std::complex<float>,1> 
		rfftBlockf(py::array_t<float,1> py_in)
		{
			py::buffer_info buf_in = py_in.request();
			float* py_ptr = (float*) buf_in.ptr;

			if (buf_in.ndim != 1){throw std::runtime_error("U dumbdumb dimension must be 1.");}	
			if ((uint64_t) buf_in.size > Npad)
			{throw std::runtime_error("U dumbdumb input too long.");}

			std::complex<float>* out;
			out = (std::complex<float>*) fftwf_malloc(howmany*(size+2)*sizeof(float));
			std::memset((float*)(out+N),0.0,(howmany*(size+2)-N)*sizeof(float));

			#pragma omp parallel for
			for(uint64_t i=0;i<threads;i++)
			{
				manage_thread_affinity();
				float* temp_out = ((float*) out)+i*(size+2)*(transfer_size[0]/size);
				for(uint64_t j=0;j<transfer_size[i];j++)
				{
					temp_out[j+2*(j/size)] = py_ptr[j+i*transfer_size[0]];
				}
				fftwf_execute_dft_r2c(planf,
								temp_out,
								reinterpret_cast<fftwf_complex*>(temp_out));
			}

			if(N!=Npad){::rfftBlock<float>(Npad-N,size,py_ptr,out);}

			py::capsule free_when_done( out, fftwf_free );
			return py::array_t<std::complex<float>,py::array::c_style>
			(
			{howmany*(size/2+1)},
			{2*sizeof(float)},
			out,
			free_when_done	
			);
		}
};

///////////////////////////////////////////////////////////////////
//						 _ _____ _____ _____                     //
//						(_)  ___|  ___|_   _|                    //
//						| | |_  | |_    | |                      //
//						| |  _| |  _|   | |                      //
//						|_|_|   |_|     |_|                      //
///////////////////////////////////////////////////////////////////
template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
ifft_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	uint64_t N = buf_in.size;
	DataType norm = 1.0/N;
	
	std::complex<DataType>* in = (std::complex<DataType>*) buf_in.ptr;
	std::complex<DataType>* out = (std::complex<DataType>*) fftw_malloc(2*N*sizeof(DataType));
	for(uint64_t i=0;i<N;i++){out[i] = in[i]*norm;}

	ifft<DataType>(N, out, out);

	py::capsule free_when_done( out, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{N},
		{2*sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
ifft_training_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	uint64_t N = buf_in.size;
	DataType norm = 1.0/N;
	
	std::complex<DataType>* in = (std::complex<DataType>*) buf_in.ptr;
	std::complex<DataType>* out = (std::complex<DataType>*) fftw_malloc(2*N*sizeof(DataType));
	for(uint64_t i=0;i<N;i++){out[i] = in[i]*norm;}

	ifft_training<DataType>(N, out, out);

	py::capsule free_when_done( out, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{N},
		{2*sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style>
ifftBlock_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in, uint64_t size)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	uint64_t N = buf_in.size;
	uint64_t howmany = N/size;
	if(howmany*size != N){howmany += 1;}
	uint64_t Npad = size*howmany;
	DataType norm = 1.0/size;

	std::complex<DataType>* in = (std::complex<DataType>*) buf_in.ptr;
	std::complex<DataType>* out = (std::complex<DataType>*) fftw_malloc(2*Npad*sizeof(DataType));
	std::memset((DataType*)(out+N),0.0,2*(Npad-N)*sizeof(DataType));
	for(uint64_t i=0;i<N;i++){out[i]=in[i]*norm;}

	ifftBlock((int) Npad,(int) size, out, out);

	py::capsule free_when_done( out, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{Npad},
		{2*sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style>
ifftBlock_training_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in, uint64_t size)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	uint64_t N = buf_in.size;
	uint64_t howmany = N/size;
	if(howmany*size != N){howmany += 1;}
	uint64_t Npad = size*howmany;
	DataType norm = 1.0/size;

	std::complex<DataType>* in = (std::complex<DataType>*) buf_in.ptr;
	std::complex<DataType>* out = (std::complex<DataType>*) fftw_malloc(2*Npad*sizeof(DataType));
	std::memset((DataType*)(out+N),0.0,2*(Npad-N)*sizeof(DataType));
	for(uint64_t i=0;i<N;i++){out[i]=in[i]*norm;}

	ifftBlock_training((int) Npad,(int) size, out, out);

	py::capsule free_when_done( out, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{Npad},
		{2*sizeof(DataType)},
		out,
		free_when_done	
	);
}

class IFFT_py
{
	public:

		fftw_plan plan;
		fftwf_plan planf;
		std::complex<double>* in;
		std::complex<float>* inf;
		uint64_t N;
		double norm;
		float normf;

		IFFT_py(uint64_t N_in)
		{
			N = N_in;
			norm = 1.0/N;
			normf = 1.0/N;

			in = (std::complex<double>*) fftw_malloc(2*N*sizeof(double));
			inf = (std::complex<float>*) fftwf_malloc(2*N*sizeof(float));
			
			plan = fftw_plan_dft_1d(
							N,
							reinterpret_cast<fftw_complex*>(in),
							reinterpret_cast<fftw_complex*>(in),
							FFTW_BACKWARD,
							FFTW_ESTIMATE);

			planf = fftwf_plan_dft_1d(
							N,
							reinterpret_cast<fftwf_complex*>(inf),
							reinterpret_cast<fftwf_complex*>(inf),
							FFTW_BACKWARD,
							FFTW_ESTIMATE);
		}
		
		~IFFT_py()
		{
			fftw_free(in);
			fftwf_free(inf);
			fftw_destroy_plan(plan);
			fftwf_destroy_plan(planf);
		}

		uint64_t getSize(){return N;}
		
		std::tuple<double,double> benchmark(uint64_t n)
		{
			auto time1 = Clock::now();
			for(uint64_t i=0;i<n;i++){fftw_execute(plan);}
			auto time2 = Clock::now();
			
			auto timef1 = Clock::now();
			for(uint64_t i=0;i<n;i++){fftwf_execute(planf);}
			auto timef2 = Clock::now();
			
			double time;
			time = std::chrono::duration_cast<std::chrono::microseconds>(time2-time1).count();

			double timef;
			timef = std::chrono::duration_cast<std::chrono::microseconds>(timef2-timef1).count();
			py::print("Time for double precision iFFT of size ",N,": ",time/n," us","sep"_a="");
			py::print("Time for single precision iFFT of size ",N,": ",timef/n," us","sep"_a="");
			return std::make_tuple<double,double>(time/n,timef/n);
		}

		py::array_t<std::complex<double>,1> 
		ifft(py::array_t<std::complex<double>,1> py_in)
		{
			py::buffer_info buf_in = py_in.request();

			if (buf_in.ndim != 1)
			{
			throw std::runtime_error("U dumbdumb dimension must be 1.");
			}	

			std::complex<double>* py_ptr = (std::complex<double>*) buf_in.ptr;
			for(uint64_t i=0;i<N;i++){in[i] = py_ptr[i]*norm;}
			
			fftw_execute(plan);
		
			std::complex<double>* out = (std::complex<double>*) malloc(2*N*sizeof(double));
			std::memcpy(out,in,2*N*sizeof(double));

			py::capsule free_when_done( out, free );
			return py::array_t<std::complex<double>,1>
			(
			{N},
			{2*sizeof(double)},
			out,
			free_when_done	
			);
		}

		py::array_t<std::complex<float>,1> 
		ifftf(py::array_t<std::complex<float>,1> py_in)
		{
			py::buffer_info buf_in = py_in.request();

			if (buf_in.ndim != 1)
			{
			throw std::runtime_error("U dumbdumb dimension must be 1.");
			}	

			std::complex<float>* py_ptr = (std::complex<float>*) buf_in.ptr;
			for(uint64_t i=0;i<N;i++){inf[i] = py_ptr[i]*normf;}

			fftwf_execute(planf);
		
			std::complex<float>* out = (std::complex<float>*) malloc(2*N*sizeof(float));
			std::memcpy(out,inf,2*N*sizeof(float));

			py::capsule free_when_done( out, free );
			return py::array_t<std::complex<float>,1>
			(
			{N},
			{2*sizeof(float)},
			out,
			free_when_done	
			);
		}
};

class iFFT_Block_py
{
	public:

		fftw_plan plan, plan2;
		fftwf_plan planf, plan2f;
		fftw_plan** plans;
		fftwf_plan** plansf;
		std::complex<double> *in, *out_temp;
		std::complex<float> *inf, *out_tempf;
		uint64_t N, size, howmany, Npad, threads;
		int length[1];
		uint64_t* transfer_size;

		iFFT_Block_py(uint64_t N_in, uint64_t size_in)
		{
			N = N_in;
			size = size_in;
			howmany = N/size;
			if(howmany*size != N){howmany += 1;}
			Npad = size*howmany;
			length[0] = (int) size;

			#ifdef _WIN32_WINNT
				threads = (uint64_t) omp.omp_get_max_threads()*GetActiveProcessorGroupCount();
			#else
				threads = omp_get_max_threads();
			#endif

			threads = std::min(threads,(uint64_t) 64);

			in = (std::complex<double>*) fftw_malloc(2*Npad*sizeof(double));
			inf = (std::complex<float>*) fftwf_malloc(2*Npad*sizeof(float));
			
			plan = fftw_plan_many_dft(
							1,
							length,
							howmany/threads,
							reinterpret_cast<fftw_complex*>(in),
							NULL,
							1,
							(int) size,
							reinterpret_cast<fftw_complex*>(in),
							NULL,
							1,
							(int) size,
							-1,
							FFTW_ESTIMATE);

			planf = fftwf_plan_many_dft(
							1,
							length,
							howmany/threads,
							reinterpret_cast<fftwf_complex*>(inf),
							NULL,
							1,
							(int) size,
							reinterpret_cast<fftwf_complex*>(inf),
							NULL,
							1,
							(int) size,
							-1,
							FFTW_ESTIMATE);

			transfer_size = (uint64_t*) malloc(threads*sizeof(uint64_t));
			for(uint64_t i=0;i<(threads-1);i++){transfer_size[i]=size*(howmany/threads);}
			transfer_size[threads-1] = N-(threads-1)*transfer_size[0];
		}
		
		~iFFT_Block_py()
		{
			free(transfer_size);
			fftw_free(in);
			fftwf_free(inf);
			fftw_destroy_plan(plan);
			fftwf_destroy_plan(planf);
		}

		uint64_t getSize(){return size;}
		uint64_t getN(){return Npad;}
		uint64_t getHowmany(){return Npad;}
		
		std::tuple<double,double> benchmark(uint64_t n)
		{
			auto time1 = Clock::now();
			for(uint64_t i=0;i<n;i++)
			{
				#pragma omp parallel for
				for(uint64_t j=0;j<threads;j++)
				{
					manage_thread_affinity();
					fftw_execute(plan);
				}
			}
			auto time2 = Clock::now();
			
			auto timef1 = Clock::now();
			for(uint64_t i=0;i<n;i++)
			{
				#pragma omp parallel for
				for(uint64_t j=0;j<threads;j++)
				{
					manage_thread_affinity();
					fftwf_execute(planf);
				}
			}
			auto timef2 = Clock::now();
			
			double time;
			time = std::chrono::duration_cast<std::chrono::microseconds>(time2-time1).count();

			double timef;
			timef = std::chrono::duration_cast<std::chrono::microseconds>(timef2-timef1).count();
			py::print("Time for ",howmany,
							" double precision iFFT of size ",size,": ",time/n," us","sep"_a="");
			py::print("Time for ",howmany,
							" single precision iFFT of size ",size,": ",timef/n," us","sep"_a="");
			return std::make_tuple<double,double>(time/n,timef/n);
		}

		py::array_t<std::complex<double>,1> 
		ifftBlock(py::array_t<std::complex<double>,1> py_in)
		{
			py::buffer_info buf_in = py_in.request();
			std::complex<double>* py_ptr = (std::complex<double>*) buf_in.ptr;

			if (buf_in.ndim != 1){throw std::runtime_error("U dumbdumb dimension must be 1.");}	
			if ((uint64_t) buf_in.size > Npad)
			{throw std::runtime_error("U dumbdumb input too long.");}

			std::complex<double>* out;
			out = (std::complex<double>*) fftw_malloc(2*Npad*sizeof(double));
			std::memset((double*)(out+N),0.0,2*(Npad-N)*sizeof(double));

			#pragma omp parallel for
			for(uint64_t i=0;i<threads;i++)
			{
				manage_thread_affinity();
				std::memcpy(out+i*size*(howmany/threads),
								py_ptr+i*size*(howmany/threads), 
								2*transfer_size[i]*sizeof(double));
				fftw_execute_dft(plan,
								reinterpret_cast<fftw_complex*>(out+i*size*(howmany/threads)),
								reinterpret_cast<fftw_complex*>(out+i*size*howmany/threads));
			}

			if(N!=Npad)
			{
				::ifftBlock<double>(Npad-N,size,py_ptr,out+threads*size*(howmany/threads));
			}

			py::capsule free_when_done( out, fftw_free );
			return py::array_t<std::complex<double>,py::array::c_style>
			(
			{Npad},
			{2*sizeof(double)},
			out,
			free_when_done	
			);
		}

		py::array_t<std::complex<float>,py::array::c_style> 
		ifftBlockf(py::array_t<std::complex<float>,py::array::c_style> py_in)
		{
			py::buffer_info buf_in = py_in.request();
			std::complex<float>* py_ptr = (std::complex<float>*) buf_in.ptr;

			if (buf_in.ndim != 1){throw std::runtime_error("U dumbdumb dimension must be 1.");}	
			if ((uint64_t) buf_in.size > Npad)
			{throw std::runtime_error("U dumbdumb input too long.");}

			std::complex<float>* out;
			out = (std::complex<float>*) fftwf_malloc(2*Npad*sizeof(float));
			std::memset((float*)(out+N),0.0,2*(Npad-N)*sizeof(float));

			#pragma omp parallel for
			for(uint64_t i=0;i<threads;i++)
			{
				manage_thread_affinity();
				std::memcpy(out+i*size*(howmany/threads),
								py_ptr+i*size*(howmany/threads), 
								2*transfer_size[i]*sizeof(float));
				fftwf_execute_dft(planf,
								reinterpret_cast<fftwf_complex*>(out+i*size*(howmany/threads)),
								reinterpret_cast<fftwf_complex*>(out+i*size*howmany/threads));
			}

			if(N!=Npad)
			{
				::ifftBlock<float>(Npad-N,size,py_ptr,out+threads*size*(howmany/threads));
			}

			py::capsule free_when_done( out, fftw_free );
			return py::array_t<std::complex<float>,py::array::c_style>
			(
			{Npad},
			{2*sizeof(float)},
			out,
			free_when_done	
			);
		}
};

///////////////////////////////////////////////////////////////////
//					 _      _____ _____ _____                    //
//					(_)_ __|  ___|  ___|_   _|                   //
//					| | '__| |_  | |_    | |                     //
//					| | |  |  _| |  _|   | |                     //
//					|_|_|  |_|   |_|     |_|                     //
///////////////////////////////////////////////////////////////////
template<class DataType>
py::array_t<DataType,py::array::c_style> 
irfft_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	uint64_t N = buf_in.size;
	DataType norm = 1.0/(2*N-2);
	
	DataType* in = (DataType*) buf_in.ptr;
	DataType* out = (DataType*) fftw_malloc((2*N)*sizeof(DataType));

	for(uint64_t i=0;i<(2*N);i++){out[i]=in[i]*norm;}

	irfft<DataType>(2*N-2, reinterpret_cast<std::complex<DataType>*>(out), out);

	py::capsule free_when_done( out, fftw_free );
	return py::array_t<DataType, py::array::c_style> 
	(
		{2*N-2},
		{sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<DataType,py::array::c_style> 
irfft_training_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	uint64_t N = buf_in.size;
	DataType norm = 1.0/(2*N-2);
	
	DataType* in = (DataType*) buf_in.ptr;
	DataType* out = (DataType*) fftw_malloc((2*N)*sizeof(DataType));

	for(uint64_t i=0;i<(2*N);i++){out[i]=in[i]*norm;}

	irfft_training<DataType>(2*N-2, reinterpret_cast<std::complex<DataType>*>(out), out);

	py::capsule free_when_done( out, fftw_free );
	return py::array_t<DataType, py::array::c_style> 
	(
		{2*N-2},
		{sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<DataType,py::array::c_style>
irfftBlock_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in, uint64_t size)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	uint64_t N = buf_in.size;
	uint64_t howmany = N/(size/2+1);
	if(howmany*(size/2+1) != N){howmany += 1;}
	uint64_t Nout = size*howmany;
	uint64_t Npad = howmany*(size/2+1);

	std::complex<DataType>* py_ptr = (std::complex<DataType>*) buf_in.ptr;
	std::complex<DataType>* in = (std::complex<DataType>*) fftw_malloc(2*Npad*sizeof(DataType));
	DataType* out = (DataType*) fftw_malloc(Nout*sizeof(DataType));
	std::memset((void*)(in+N),0.0,2*(Npad-N)*sizeof(DataType));
	std::memcpy(in,py_ptr,2*N*sizeof(DataType));

	irfftBlock((int) Nout,(int) size, in, out);

	free(in);

	py::capsule free_when_done( out, fftw_free );
	return py::array_t<DataType, py::array::c_style> 
	(
		{Nout},
		{sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<DataType,py::array::c_style>
irfftBlock_training_py(py::array_t<std::complex<DataType>,1> py_in, uint64_t size)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	uint64_t N = buf_in.size;
	uint64_t howmany = N/(size/2+1);
	if(howmany*(size/2+1) != N){howmany += 1;}
	uint64_t Nout = size*howmany;
	uint64_t Npad = howmany*(size/2+1);

	std::complex<DataType>* py_ptr = (std::complex<DataType>*) buf_in.ptr;
	std::complex<DataType>* in = (std::complex<DataType>*) fftw_malloc(2*Npad*sizeof(DataType));
	DataType* out = (DataType*) fftw_malloc(Nout*sizeof(DataType));
	std::memset((void*)(in+N),0.0,2*(Npad-N)*sizeof(DataType));
	std::memcpy(in,py_ptr,2*N*sizeof(DataType));

	irfftBlock_training((int) Nout,(int) size, in, out);

	free(in);

	py::capsule free_when_done( out, fftw_free );
	return py::array_t<DataType, py::array::c_style> 
	(
		{Nout},
		{sizeof(DataType)},
		out,
		free_when_done	
	);
}

class IRFFT_py
{
	public:

		fftw_plan plan;
		fftwf_plan planf;
		std::complex<double>* in;
		std::complex<float>* inf;
		uint64_t N;
		double norm;
		float normf;

		IRFFT_py(uint64_t N_in)
		{
			N = N_in;
			norm = 1.0/N;
			normf = 1.0/N;

			in = (std::complex<double>*) fftw_malloc((N+2)*sizeof(double));
			inf = (std::complex<float>*) fftwf_malloc((N+2)*sizeof(float));
			
			plan = fftw_plan_dft_c2r_1d(N,
							reinterpret_cast<fftw_complex*>(in),
							reinterpret_cast<double*>(in),
							FFTW_ESTIMATE);
			planf=fftwf_plan_dft_c2r_1d(N,
							reinterpret_cast<fftwf_complex*>(inf),
							reinterpret_cast<float*>(inf),
							FFTW_ESTIMATE);
		}
		
		~IRFFT_py()
		{
			fftw_free(in);
			fftwf_free(inf);
			fftw_destroy_plan(plan);
			fftwf_destroy_plan(planf);
		}

		uint64_t getSize(){return N;}
		
		std::tuple<double,double> benchmark(uint64_t n)
		{
			auto time1 = Clock::now();
			for(uint64_t i=0;i<n;i++){fftw_execute(plan);}
			auto time2 = Clock::now();
			
			auto timef1 = Clock::now();
			for(uint64_t i=0;i<n;i++){fftwf_execute(planf);}
			auto timef2 = Clock::now();
			
			double time;
			time = std::chrono::duration_cast<std::chrono::microseconds>(time2-time1).count();

			double timef;
			timef = std::chrono::duration_cast<std::chrono::microseconds>(timef2-timef1).count();
			py::print("Time for double precision irFFT of size ",N,": ",time/n," us","sep"_a="");
			py::print("Time for single precision irFFT of size ",N,": ",timef/n," us","sep"_a="");
			return std::make_tuple<double,double>(time/n,timef/n);
		}

		py::array_t<double,1> 
		irfft(py::array_t<std::complex<double>,1> py_in)
		{
			py::buffer_info buf_in = py_in.request();

			if (buf_in.ndim != 1)
			{
			throw std::runtime_error("U dumbdumb dimension must be 1.");
			}	
			
			//std::memcpy(in,(std::complex<double>*) buf_in.ptr, (N+2)*sizeof(double));
			std::complex<double>* py_ptr = (std::complex<double>*) buf_in.ptr;
			for(uint64_t i=0; i<(N/2+1);i++){in[i] = py_ptr[i]*norm;}
			
			fftw_execute(plan);
		
			double* out = (double*) malloc(N*sizeof(double));
			std::memcpy(out,(void*) in,N*sizeof(double));

			py::capsule free_when_done( out, free );
			return py::array_t<double,1>
			(
			{N},
			{sizeof(double)},
			out,
			free_when_done	
			);
		}

		py::array_t<float,1> 
		irfftf(py::array_t<std::complex<float>,1> py_in)
		{
			py::buffer_info buf_in = py_in.request();

			if (buf_in.ndim != 1)
			{
			throw std::runtime_error("U dumbdumb dimension must be 1.");
			}	

			//std::memcpy(inf,(std::complex<float>*) buf_in.ptr, (N+2)*sizeof(float));
			std::complex<float>* py_ptr = (std::complex<float>*) buf_in.ptr;
			for(uint64_t i=0; i<(N/2+1);i++){inf[i] = py_ptr[i]*normf;}

			fftwf_execute(planf);
		
			float* out = (float*) malloc(N*sizeof(float));
			std::memcpy(out,(void*) inf,N*sizeof(float));

			py::capsule free_when_done( out, free );
			return py::array_t<float,1>
			(
			{N},
			{sizeof(float)},
			out,
			free_when_done	
			);
		}
};



