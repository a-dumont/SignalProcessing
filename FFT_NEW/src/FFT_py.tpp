///////////////////////////////////////////////////////////////////
//						 _____ _____ _____                       //
//						|  ___|  ___|_   _|                      //
//						| |_  | |_    | |                        //
//						|  _| |  _|   | |                        //
//						|_|   |_|     |_|                        //
///////////////////////////////////////////////////////////////////
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
			fftw_destroy_plan(plan);
			fftwf_destroy_plan(planf);
		}

		uint64_t getN(){return N;}
		
		std::tuple<double,double> benchmark(uint64_t n)
		{
			auto time1 = Clock::now();
			for(uint64_t i=0;i<n;i++){fftw_execute(plan);}
			auto time2 = Clock::now();
			
			auto timef1 = Clock::now();
			for(uint64_t i=0;i<n;i++){fftwf_execute(planf);}
			auto timef2 = Clock::now();
			
			double time = std::chrono::duration_cast<std::chrono::microseconds>(time2-time1).count();
			double timef = std::chrono::duration_cast<std::chrono::microseconds>(time2f-time1f).count();
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
			fftw_destroy_plan(plan);
			fftwf_destroy_plan(planf);
		}

		uint64_t getN(){return N;}
		
		std::tuple<double,double> benchmark(uint64_t n)
		{
			auto time1 = Clock::now();
			for(uint64_t i=0;i<n;i++){fftw_execute(plan);}
			auto time2 = Clock::now();
			
			auto timef1 = Clock::now();
			for(uint64_t i=0;i<n;i++){fftwf_execute(planf);}
			auto timef2 = Clock::now();
			
			double time = std::chrono::duration_cast<std::chrono::microseconds>(time2-time1).count();
			double timef = std::chrono::duration_cast<std::chrono::microseconds>(time2f-time1f).count();
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
	//std::memcpy(out,in,2*N*sizeof(DataType));
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
	//std::memcpy(out,in,2*N*sizeof(DataType));
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
			fftw_destroy_plan(plan);
			fftwf_destroy_plan(planf);
		}

		uint64_t getN(){return N;}
		
		std::tuple<double,double> benchmark(uint64_t n)
		{
			auto time1 = Clock::now();
			for(uint64_t i=0;i<n;i++){fftw_execute(plan);}
			auto time2 = Clock::now();
			
			auto timef1 = Clock::now();
			for(uint64_t i=0;i<n;i++){fftwf_execute(planf);}
			auto timef2 = Clock::now();
			
			double time = std::chrono::duration_cast<std::chrono::microseconds>(time2-time1).count();
			double timef = std::chrono::duration_cast<std::chrono::microseconds>(time2f-time1f).count();
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

			//std::memcpy(in,(std::complex<double>*) buf_in.ptr, 2*N*sizeof(double));
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

			//std::memcpy(inf,(std::complex<float>*) buf_in.ptr, 2*N*sizeof(float));
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
			fftw_destroy_plan(plan);
			fftwf_destroy_plan(planf);
		}

		uint64_t getN(){return N;}
		
		std::tuple<double,double> benchmark(uint64_t n)
		{
			auto time1 = Clock::now();
			for(uint64_t i=0;i<n;i++){fftw_execute(plan);}
			auto time2 = Clock::now();
			
			auto timef1 = Clock::now();
			for(uint64_t i=0;i<n;i++){fftwf_execute(planf);}
			auto timef2 = Clock::now();
			
			double time = std::chrono::duration_cast<std::chrono::microseconds>(time2-time1).count();
			double timef = std::chrono::duration_cast<std::chrono::microseconds>(time2f-time1f).count();
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



