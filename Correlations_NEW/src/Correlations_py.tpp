///////////////////////////////////////////////////////////////////
//                       _    ____                               //
//                      / \  / ___|___  _ __ _ __                //
//                     / _ \| |   / _ \| '__| '__|               //
//                    / ___ \ |__| (_) | |  | |                  //
//                   /_/   \_\____\___/|_|  |_|                  //
///////////////////////////////////////////////////////////////////
template<class DataType>
py::array_t<DataType,py::array::c_style>
aCorrCircularFreqAVX_py(py::array_t<DataType,py::array::c_style> py_in, uint64_t size)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	uint64_t N = buf_in.size;
	uint64_t howmany = N/size;
	if(size*howmany != N){howmany+=1;}

	// Retreive all pointers
	DataType* in = (DataType*) buf_in.ptr;
	
	DataType* out;
	out = (DataType*) fftw_malloc(2*(howmany+1)*(size/2+1)*sizeof(DataType));
	std::memset((void*) out, 0, 2*(howmany+1)*(size/2+1)*sizeof(DataType));
	
	DataType* result;
   	result = (DataType*) malloc((size/2+1)*sizeof(DataType));

	// Compute rFFT blocks
	rfftBlock<DataType>((int) N, (int) size, in, 
					reinterpret_cast<std::complex<DataType>*>(out)+(size/2+1));
	
	// Compute product
	aCorrCircularFreqAVX<DataType>(2*howmany*(size/2+1),out+2*(size/2+1),out+2*(size/2+1));
	
	// Sum all blocks
	reduceBlockAVX<DataType>(2*howmany*(size/2+1),2*(size/2+1),
					out+2*(size/2+1),
					out);

	// Divide the sum by the number of blocks
	for(uint64_t i=0;i<(size/2+1);i++){result[i]=(out[2*i]+out[2*i+1])/howmany;}
	
	// Free intermediate buffer
	fftw_free(out);

	py::capsule free_when_done( result, free );
	return py::array_t<DataType, py::array::c_style>
	(
		{(size/2+1)},
		{sizeof(DataType)},
		result,
		free_when_done
	);
}

class ACorrCircularFreqAVX_py
{
	public:

		fftw_plan plan, plan2;
		fftwf_plan planf, plan2f;
		double *in, *out_temp;
		float *inf, *out_tempf;
		uint64_t N, size, howmany, Npad, threads;
		int length[1];
		uint64_t* transfer_size;

		ACorrCircularFreqAVX_py(uint64_t N_in, uint64_t size_in)
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
			
			out_temp = (double*) fftw_malloc((size+2)*howmany*sizeof(double));
			out_tempf = (float*) fftwf_malloc((size+2)*howmany*sizeof(float));
			
			plan = fftw_plan_many_dft_r2c(1, length, howmany/threads, in, NULL,
							1, (int) size, reinterpret_cast<fftw_complex*>(out_temp),
							NULL, 1, (int) size/2+1, FFTW_ESTIMATE);

			planf = fftwf_plan_many_dft_r2c(1, length, howmany/threads, inf,
							NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(out_tempf),
							NULL, 1, (int) size/2+1, FFTW_ESTIMATE);

			transfer_size = (uint64_t*) malloc(threads*sizeof(uint64_t));
			for(uint64_t i=0;i<(threads-1);i++){transfer_size[i]=size*(howmany/threads);}
			transfer_size[threads-1] = N-(threads-1)*transfer_size[0];
		}
		
		~ACorrCircularFreqAVX_py()
		{
			free(transfer_size);
			fftw_free(in);
			fftwf_free(inf);
			fftw_free(out_temp);
			fftwf_free(out_tempf);
			fftw_destroy_plan(plan);
			fftwf_destroy_plan(planf);
		}

		uint64_t getSize(){return size;}
		uint64_t getN(){return Npad;}
		uint64_t getHowmany(){return Npad;}

		void train()
		{
			fftw_destroy_plan(plan);
			plan = fftw_plan_many_dft_r2c(1, length, howmany/threads, in, NULL,
							1, (int) size, reinterpret_cast<fftw_complex*>(out_temp),
							NULL, 1, (int) size/2+1, FFTW_EXHAUSTIVE);

			fftwf_destroy_plan(planf);
			planf = fftwf_plan_many_dft_r2c(1, length, howmany/threads, inf,
							NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(out_tempf),
							NULL, 1, (int) size/2+1, FFTW_EXHAUSTIVE);

			py::print("Training double precision.");
			fftw_execute(plan);
			fftw_export_wisdom_to_filename(&wisdom_path[0]);
			py::print("Training single precision.");
			fftwf_execute(planf);
			fftw_export_wisdom_to_filename(&wisdom_path[0]);
		}
		
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
			timef=std::chrono::duration_cast<std::chrono::microseconds>(timef2-timef1).count();
			
			py::print("Time for ",howmany,
							" double precision rFFT of size ",
							size,": ",time/n," us","sep"_a="");
			py::print("Time for ",howmany,
							" single precision rFFT of size ",
							size,": ",timef/n," us","sep"_a="");

			return std::make_tuple<double,double>(time/n,timef/n);
		}

		py::array_t<double,1> 
		aCorrCircularFreqAVX(py::array_t<double,1> py_in)
		{
			py::buffer_info buf_in = py_in.request();
			double* py_ptr = (double*) buf_in.ptr;

			if (buf_in.ndim != 1){throw std::runtime_error("U dumbdumb dimension must be 1.");}	
			if ((uint64_t) buf_in.size > Npad)
			{throw std::runtime_error("U dumbdumb input too long.");}

			double* out;
			out = (double*) malloc((size/2+1)*sizeof(double));
			std::memset(out,0.0,(size/2+1)*sizeof(double));
			std::memset(in,0.0,(Npad-N)*sizeof(double));

			#pragma omp parallel for
			for(uint64_t i=0;i<threads;i++)
			{
				manage_thread_affinity();
				std::memcpy(in+i*transfer_size[0],
								py_ptr+i*transfer_size[0],transfer_size[i]*sizeof(double));
				fftw_execute_dft_r2c(plan,
								in+i*transfer_size[0],
								reinterpret_cast<fftw_complex*>
								(out_temp+i*(transfer_size[0]/size)*(size/2+1)));
				std::memset((void*) (in+i*transfer_size[0]), 0, transfer_size[i]*sizeof(double));
				::aCorrCircularFreqAVX(2*(size/2+1)*(howmany/threads),
								out_temp+i*(transfer_size[0]/size)*(size/2+1),
								out_temp+i*(transfer_size[0]/size)*(size/2+1));
				::reduceBlockAVX(2*(size/2+1)*(howmany/threads),2*(size/2+1),
								out_temp+i*(transfer_size[0]/size)*(size/2+1).
								in+i*transfer_size[0]);
			}
			for(uint64_t i=0;i<threads;i++)
			{
				for(uint64_t j=0;j<(size/2+1);j++)
				{
					out[j]+=((in+i*transfer_size[0])[2*j]+(in+i*transfer_size[0])[2*j+1])/howmany;
				}
			}

			if(howmany != threads*(howmany/threads))
			{
				::rfftBlock<double>(size*(howmany-threads*(howmany/threads)),size,
								py_ptr+threads*(howmany/threads)*size,
								reinterpret_cast<std::complex<double>*>(
								out_temp+threads*(howmany/threads)*(size/2+1)));
				::aCorrCircularFreqAVX(size*(howmany-threads*(howmany/threads)),
								2*(size/2+1),
								out_temp+threads*(howmany/threads)*(size/2+1),
								out_temp+threads*(howmany/threads)*(size/2+1));
				std::memset((void*) in, 0, transfer_size[0]*sizeof(double));
				::reduceBlockAVX(2*(size/2+1)*(howmany/threads),2*(size/2+1),
								out_temp+threads*(howmany/threads)*(size/2+1),in);
				for(uint64_t j=0;j<(size/2+1);j++){out[j]+=(in[2*j]+in[2*j+1])/howmany;}
			}

			py::capsule free_when_done( out, free );
			return py::array_t<double,py::array::c_style>
			(
			{(size/2+1)},
			{sizeof(double)},
			out,
			free_when_done	
			);
		}
		/*
		py::array_t<std::complex<float>,py::array::c_style> 
		rfftBlockf(py::array_t<float,py::array::c_style> py_in)
		{
			py::buffer_info buf_in = py_in.request();
			float* py_ptr = (float*) buf_in.ptr;

			if (buf_in.ndim != 1){throw std::runtime_error("U dumbdumb dimension must be 1.");}	
			if ((uint64_t) buf_in.size > Npad)
			{throw std::runtime_error("U dumbdumb input too long.");}

			std::complex<float>* outf;
			outf = (std::complex<float>*) fftwf_malloc(howmany*(size+2)*sizeof(float));
			std::memset(inf,0.0,(Npad-N)*sizeof(float));

			#pragma omp parallel for
			for(uint64_t i=0;i<threads;i++)
			{
				manage_thread_affinity();
				std::memcpy(inf+i*transfer_size[0],
								py_ptr+i*transfer_size[0],transfer_size[i]*sizeof(float));
				fftwf_execute_dft_r2c(planf,
								inf+i*transfer_size[0],
								reinterpret_cast<fftwf_complex*>
								(outf+i*(transfer_size[0]/size)*(size/2+1)));
			}

			if(howmany != threads*(howmany/threads))
			{
				::rfftBlock<float>(size*(howmany-threads*(howmany/threads)),size,
								py_ptr+threads*(howmany/threads)*size,
								outf+threads*(howmany/threads)*(size/2+1));
			}

			py::capsule free_when_done2( outf, fftwf_free );
			return py::array_t<std::complex<float>,py::array::c_style>
			(
			{howmany*(size/2+1)},
			{2*sizeof(float)},
			outf,
			free_when_done2	
			);
		}*/
};

///////////////////////////////////////////////////////////////////
//                      __  ______                               //
//                      \ \/ / ___|___  _ __ _ __                //
//                       \  / |   / _ \| '__| '__|               //
//                       /  \ |__| (_) | |  | |                  //
//                      /_/\_\____\___/|_|  |_|                  //
///////////////////////////////////////////////////////////////////
template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style>
xCorrCircularFreqAVX_py(py::array_t<DataType,py::array::c_style> py_in1, 
				py::array_t<DataType,py::array::c_style> py_in2, uint64_t size)
{
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	if (buf_in1.ndim != 1 || buf_in2.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	uint64_t N = std::min(buf_in1.size,buf_in2.size);
	uint64_t howmany = N/size;
	if(size*howmany != N){howmany+=1;}

	// Retreive all pointers
	DataType* in1 = (DataType*) buf_in1.ptr;
	DataType* in2 = (DataType*) buf_in2.ptr;
	
	DataType *out1, *out2;
	out1 = (DataType*) fftw_malloc(2*(howmany+1)*(size/2+1)*sizeof(DataType));
	out2 = (DataType*) fftw_malloc(2*(howmany+1)*(size/2+1)*sizeof(DataType));
	std::memset((void*) out1, 0, 2*(howmany+1)*(size/2+1)*sizeof(DataType));
	std::memset((void*) out2, 0, 2*(howmany+1)*(size/2+1)*sizeof(DataType));
	
	DataType* result;
   	result = (DataType*) malloc(2*(size/2+1)*sizeof(DataType));

	// Compute rFFT blocks
	rfftBlock<DataType>((int) N, (int) size, in1, 
					reinterpret_cast<std::complex<DataType>*>(out1)+(size/2+1));
	rfftBlock<DataType>((int) N, (int) size, in2, 
					reinterpret_cast<std::complex<DataType>*>(out2)+(size/2+1));

	// Compute product
	xCorrCircularFreqAVX<DataType>(2*howmany*(size/2+1),out1+2*(size/2+1),
					out2+2*(size/2+1), out1+2*(size/2+1));
	
	// Sum all blocks
	reduceBlockAVX<DataType>(2*howmany*(size/2+1),2*(size/2+1),
					out1+2*(size/2+1),
					out1);

	// Divide the sum by the number of blocks
	for(uint64_t i=0;i<(2*(size/2+1));i++)
	{
		result[i]=out1[i]/howmany;
	}
	
	// Free intermediate buffer
	fftw_free(out1);
	fftw_free(out2);

	py::capsule free_when_done(result, free);
	return py::array_t<std::complex<DataType>, py::array::c_style>
	(
		{size/2+1},
		{2*sizeof(DataType)},
		reinterpret_cast<std::complex<DataType>*>(result),
		free_when_done
	);
}

///////////////////////////////////////////////////////////////////
//                       _____ ____                              //
//                      |  ___/ ___|___  _ __ _ __               //
//                      | |_ | |   / _ \| '__| '__|              //
//                      |  _|| |__| (_) | |  | |                 //
//                      |_|   \____\___/|_|  |_|                 //
///////////////////////////////////////////////////////////////////
template<class DataType>
std::tuple<
py::array_t<DataType,py::array::c_style>,
py::array_t<DataType,py::array::c_style>,
py::array_t<std::complex<DataType>,py::array::c_style>
>
axCorrCircularFreqAVX_py(py::array_t<DataType,py::array::c_style> py_in1, 
				py::array_t<DataType,py::array::c_style> py_in2, uint64_t size)
{
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	if (buf_in1.ndim != 1 || buf_in2.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	uint64_t N = std::min(buf_in1.size,buf_in2.size);
	uint64_t howmany = N/size;
	if(size*howmany != N){howmany+=1;}

	// Retreive all pointers
	DataType* in1 = (DataType*) buf_in1.ptr;
	DataType* in2 = (DataType*) buf_in2.ptr;
	
	DataType *out1, *out2, *out3;
	out1 = (DataType*) fftw_malloc(2*(howmany+1)*(size/2+1)*sizeof(DataType));
	out2 = (DataType*) fftw_malloc(2*(howmany+1)*(size/2+1)*sizeof(DataType));
	out3 = (DataType*) malloc(2*(howmany+1)*(size/2+1)*sizeof(DataType));
	std::memset((void*) out1, 0, 2*(howmany+1)*(size/2+1)*sizeof(DataType));
	std::memset((void*) out2, 0, 2*(howmany+1)*(size/2+1)*sizeof(DataType));
	std::memset((void*) out3, 0, 2*(howmany+1)*(size/2+1)*sizeof(DataType));
	
	DataType *result1, *result2, *result3;
   	result1 = (DataType*) malloc((size/2+1)*sizeof(DataType));
   	result2 = (DataType*) malloc((size/2+1)*sizeof(DataType));
   	result3 = (DataType*) malloc(2*(size/2+1)*sizeof(DataType));

	// Compute rFFT blocks
	rfftBlock<DataType>((int) N, (int) size, in1, 
					reinterpret_cast<std::complex<DataType>*>(out1)+(size/2+1));
	rfftBlock<DataType>((int) N, (int) size, in2, 
					reinterpret_cast<std::complex<DataType>*>(out2)+(size/2+1));

	// Compute product
	axCorrCircularFreqAVX<DataType>(2*howmany*(size/2+1),out1+2*(size/2+1),
					out2+2*(size/2+1), out1+2*(size/2+1), out2+2*(size/2+1), out3+2*(size/2+1));
	
	// Sum all blocks
	reduceBlockAVX<DataType>(2*howmany*(size/2+1),2*(size/2+1), out1+2*(size/2+1), out1);
	reduceBlockAVX<DataType>(2*howmany*(size/2+1),2*(size/2+1), out2+2*(size/2+1), out2);
	reduceBlockAVX<DataType>(2*howmany*(size/2+1),2*(size/2+1), out3+2*(size/2+1), out3);

	// Divide the sum by the number of blocks
	for(uint64_t i=0;i<(size/2+1);i++)
	{
		result1[i]=(out1[2*i]+out1[2*i+1])/howmany;
		result2[i]=(out2[2*i]+out2[2*i+1])/howmany;
		result3[2*i]=out3[2*i]/howmany;
		result3[2*i+1]=out3[2*i+1]/howmany;
	}
	
	// Free intermediate buffer
	fftw_free(out1);
	fftw_free(out2);
	free(out3);

	py::capsule free_when_done1(result1, free);
	py::capsule free_when_done2(result2, free);
	py::capsule free_when_done3(result3, free);
	return std::make_tuple(
	py::array_t<DataType, py::array::c_style>
	(
		{size/2+1},
		{sizeof(DataType)},
		result1,
		free_when_done1
	),
	py::array_t<DataType, py::array::c_style>
	(
		{size/2+1},
		{sizeof(DataType)},
		result2,
		free_when_done2
	),
	py::array_t<std::complex<DataType>, py::array::c_style>
	(
		{size/2+1},
		{2*sizeof(DataType)},
		reinterpret_cast<std::complex<DataType>*>(result3),
		free_when_done3
	));
}

///////////////////////////////////////////////////////////////////
//               ___ _____ _   _ _____ ____  ____                //
//				/ _ \_   _| | | | ____|  _ \/ ___|               //
//			   | | | || | | |_| |  _| | |_) \___ \               //
//			   | |_| || | |  _  | |___|  _ < ___) |              //
//				\___/ |_| |_| |_|_____|_| \_\____/               //
///////////////////////////////////////////////////////////////////

template<class DataType>
DataType reduceAVX_py(py::array_t<DataType,py::array::c_style> py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	uint64_t N = buf_in.size;
	uint64_t size = std::max((uint64_t) 256, N*sizeof(DataType)/32);

	DataType* in = (DataType*) buf_in.ptr;
	DataType* out = (DataType*) malloc(size*sizeof(DataType));

	reduceAVX<DataType>(N, in, out);
	DataType result = out[0];
	free(out);

	return result;
}

template<class DataType>
py::array_t<DataType,py::array::c_style>
reduceBlockAVX_py(py::array_t<DataType,py::array::c_style> py_in, uint64_t size)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	uint64_t N = buf_in.size;

	DataType* in = (DataType*) buf_in.ptr;
	DataType* out = (DataType*) malloc(N*sizeof(DataType));

	std::memset((void*) out,0,N*sizeof(DataType));
	
	reduceBlockAVX<DataType>(N, size, in, out);

	py::capsule free_when_done( out, free );
	return py::array_t<DataType, py::array::c_style>
	(
		{size},
		{sizeof(DataType)},
		out,
		free_when_done
	);
}