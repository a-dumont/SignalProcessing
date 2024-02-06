///////////////////////////////////////////////////////////////////
//                       _    ____                               //
//                      / \  / ___|___  _ __ _ __                //
//                     / _ \| |   / _ \| '__| '__|               //
//                    / ___ \ |__| (_) | |  | |                  //
//                   /_/   \_\____\___/|_|  |_|                  //
///////////////////////////////////////////////////////////////////
#include <type_traits>
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

ACorrCircularFreqAVX_py::ACorrCircularFreqAVX_py(uint64_t N_in, uint64_t size_in)
{
	N = N_in;
	size = size_in;
	cSize = size/2+1;
	howmany = N/size;
	length[0] = (int) size;

	#ifdef _WIN32_WINNT
		threads = (uint64_t) omp.omp_get_max_threads()*GetActiveProcessorGroupCount();
	#else
		threads = omp_get_max_threads();
	#endif

	threads = std::min(threads,(uint64_t) 64);
	if(threads > howmany){threads=1;}
	howmanyPerThread = howmany/threads;
			
	transferSize=size*howmanyPerThread;

	in = (double*) fftw_malloc(2*cSize*threads*howmanyPerThread*sizeof(double));
	out = (double*) fftw_malloc(2*cSize*threads*howmanyPerThread*sizeof(double));
	inf = (float*) in; outf = (float*) out;

	inThreads = (double**) malloc(threads*sizeof(double*));
	outThreads = (double**) malloc(threads*sizeof(double*));
	inThreadsf = (float**) malloc(threads*sizeof(float*));
	outThreadsf = (float**) malloc(threads*sizeof(float*));
	for(uint64_t i=0;i<threads;i++)
	{
		inThreads[i] = in+2*i*howmanyPerThread*cSize;
		outThreads[i] = out+2*i*howmanyPerThread*cSize;
		inThreadsf[i] = inf+2*i*howmanyPerThread*cSize;
		outThreadsf[i] = outf+2*i*howmanyPerThread*cSize;
	}

	plan = fftw_plan_many_dft_r2c(1, length, howmanyPerThread, in, 
					NULL, 1, (int) size, reinterpret_cast<fftw_complex*>(out),
					NULL, 1, (int) cSize, FFTW_ESTIMATE);

	planf = fftwf_plan_many_dft_r2c(1, length, howmanyPerThread, inf,
					NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(outf),
					NULL, 1, (int) cSize, FFTW_ESTIMATE);

	if(howmany-threads*howmanyPerThread != 0)
	{
		plan2 = fftw_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread, in, 
						NULL, 1, (int) size, reinterpret_cast<fftw_complex*>(out),
						NULL, 1, (int) cSize, FFTW_ESTIMATE);
		plan2f = fftwf_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread,inf,
						NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(outf),
						NULL, 1, (int) cSize, FFTW_ESTIMATE);
	}	
}
		
ACorrCircularFreqAVX_py::~ACorrCircularFreqAVX_py()
{
	fftw_free(in);
	fftw_free(out);
	free(inThreads);
	free(outThreads);
	free(inThreadsf);
	free(outThreadsf);
	fftw_destroy_plan(plan);
	fftwf_destroy_plan(planf);
			
	if(howmany-threads*howmanyPerThread != 0)
	{
		fftw_destroy_plan(plan2);
		fftwf_destroy_plan(plan2f);
	}
}

uint64_t ACorrCircularFreqAVX_py::getSize(){return size;}
uint64_t ACorrCircularFreqAVX_py::getN(){return Npad;}
uint64_t ACorrCircularFreqAVX_py::getHowmany(){return howmany;}

void ACorrCircularFreqAVX_py::train()
{	
	fftw_import_wisdom_from_filename(&wisdom_path[0]);
	fftwf_import_wisdom_from_filename(&wisdom_pathf[0]);
			
	fftw_destroy_plan(plan);
	plan = fftw_plan_many_dft_r2c(1, length, howmany/threads, in, NULL,
					1, (int) size, reinterpret_cast<fftw_complex*>(out),
					NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
	fftw_execute(plan);

	fftwf_destroy_plan(planf);
	planf = fftwf_plan_many_dft_r2c(1, length, howmany/threads, inf,
					NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(outf),
					NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
	fftwf_execute(planf);

	if(howmany-threads*howmanyPerThread != 0)
	{
		fftw_destroy_plan(plan2);
		plan2 = fftw_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread, in, 
						NULL, 1, (int) size, reinterpret_cast<fftw_complex*>(out),
						NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
		fftw_execute(plan2);
		
		fftwf_destroy_plan(plan2f);
		plan2f = fftwf_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread,inf,
						NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(outf),
						NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
		fftwf_execute(plan2f);
	}

	fftw_export_wisdom_to_filename(&wisdom_path[0]);
	fftwf_export_wisdom_to_filename(&wisdom_pathf[0]);
	py::print("Training done.");
}
		
std::tuple<double,double> ACorrCircularFreqAVX_py::benchmark(uint64_t n)
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
			
	py::print("Time for ",howmany, " double precision rFFT of size ", 
				size,": ",time/n," us","sep"_a="");
	py::print("Time for ",howmany, " single precision rFFT of size ",
				size,": ",timef/n," us","sep"_a="");

	return std::make_tuple<double,double>(time/n,timef/n);
}

py::array_t<double,1> 
ACorrCircularFreqAVX_py::aCorrCircularFreqAVX(py::array_t<double,1> py_in)
{
	py::buffer_info buf_in = py_in.request();
	double* py_ptr = (double*) buf_in.ptr;

	if (buf_in.ndim != 1){throw std::runtime_error("U dumbdumb dimension must be 1.");}	
	if ((uint64_t) buf_in.size > Npad){throw std::runtime_error("U dumbdumb input too long.");}

	double* result;
	result = (double*) malloc(cSize*sizeof(double));
	std::memset(result,0.0,cSize*sizeof(double));
			
	#pragma omp parallel for
	for(uint64_t i=0;i<threads;i++)
	{
		manage_thread_affinity();
		std::memcpy(inThreads[i],py_ptr+i*transferSize,transferSize*sizeof(double));
		fftw_execute_dft_r2c(plan,inThreads[i],reinterpret_cast<fftw_complex*>(outThreads[i]));
		::aCorrCircularFreqAVX(2*cSize*howmanyPerThread,outThreads[i],outThreads[i]);
		::reduceInPlaceBlockAVX(2*cSize*howmanyPerThread,2*cSize,outThreads[i]);
					
		for(uint64_t j=0;j<cSize;j++)
		{
			#pragma omp atomic
			result[j] += (outThreads[i][2*j]+outThreads[i][2*j+1])/howmany;
		}
	}
			
	if(howmany-threads*howmanyPerThread != 0)
	{
		std::memcpy(inThreads[0],py_ptr+threads*transferSize,
						size*(howmany-threads*howmanyPerThread)*sizeof(double));
		fftw_execute_dft_r2c(plan2,inThreads[0],
						reinterpret_cast<fftw_complex*>(outThreads[0]));
		::aCorrCircularFreqAVX(2*cSize*(howmany-threads*howmanyPerThread),
						outThreads[0],outThreads[0]);
		::reduceInPlaceBlockAVX(2*cSize*(howmany-threads*howmanyPerThread),
						2*cSize,outThreads[0]);
		for(uint64_t j=0;j<cSize;j++)
		{
			result[j] += (outThreads[0][2*j]+outThreads[0][2*j+1])/howmany;
		}
	}

	py::capsule free_when_done( result, free );
	return py::array_t<double,py::array::c_style>
	(
	{cSize},
	{sizeof(double)},
	result,
	free_when_done	
	);
}

py::array_t<float,1> 
ACorrCircularFreqAVX_py::aCorrCircularFreqAVXf(py::array_t<float,1> py_in)
{			
	py::buffer_info buf_in = py_in.request();
	float* py_ptr = (float*) buf_in.ptr;

	if (buf_in.ndim != 1){throw std::runtime_error("U dumbdumb dimension must be 1.");}	
	if ((uint64_t) buf_in.size > Npad){throw std::runtime_error("U dumbdumb input too long.");}

	float* result;
	result = (float*) malloc(cSize*sizeof(float));
	std::memset(result,0.0,cSize*sizeof(float));
			
	#pragma omp parallel for
	for(uint64_t i=0;i<threads;i++)
	{
		manage_thread_affinity();
		std::memcpy(inThreadsf[i],py_ptr+i*transferSize,transferSize*sizeof(float));
		fftwf_execute_dft_r2c(planf,inThreadsf[i],
						reinterpret_cast<fftwf_complex*>(outThreadsf[i]));
		::aCorrCircularFreqAVX(2*cSize*howmanyPerThread,outThreadsf[i],outThreadsf[i]);
		::reduceInPlaceBlockAVX(2*cSize*howmanyPerThread,2*cSize,outThreadsf[i]);
					
		for(uint64_t j=0;j<cSize;j++)
		{
			#pragma omp atomic
			result[j] += (outThreadsf[i][2*j]+outThreadsf[i][2*j+1])/howmany;
		}
	}
			
	if(howmany-threads*howmanyPerThread != 0)
	{
		std::memcpy(inThreadsf[0],py_ptr+threads*transferSize,
						size*(howmany-threads*howmanyPerThread)*sizeof(float));
		fftwf_execute_dft_r2c(plan2f,inThreadsf[0],
						reinterpret_cast<fftwf_complex*>(outThreadsf[0]));
		::aCorrCircularFreqAVX(2*cSize*(howmany-threads*howmanyPerThread),
						outThreadsf[0],outThreadsf[0]);
		::reduceInPlaceBlockAVX(2*cSize*(howmany-threads*howmanyPerThread),
						2*cSize,outThreadsf[0]);
		for(uint64_t j=0;j<cSize;j++)
		{
			result[j] += (outThreadsf[0][2*j]+outThreadsf[0][2*j+1])/howmany;
		}
	}

	py::capsule free_when_done( result, free );
	return py::array_t<float,py::array::c_style>
	(
	{cSize},
	{sizeof(float)},
	result,
	free_when_done	
	);
}

DigitizerACorrCircularFreqAVX_py::DigitizerACorrCircularFreqAVX_py(uint64_t N_in,uint64_t size_in)
{
	N = N_in;
	size = size_in;
	cSize = size/2+1;
	howmany = N/size;
	length[0] = (int) size;

	#ifdef _WIN32_WINNT
		threads = (uint64_t) omp.omp_get_max_threads()*GetActiveProcessorGroupCount();
	#else
		threads = omp_get_max_threads();
	#endif

	threads = std::min(threads,(uint64_t) 64);
	if(threads > howmany){threads=1;}
	howmanyPerThread = howmany/threads;
			
	transferSize=size*howmanyPerThread;

	in = (double*) fftw_malloc(2*cSize*threads*howmanyPerThread*sizeof(double));
	out = (double*) fftw_malloc(2*cSize*threads*howmanyPerThread*sizeof(double));
	inf = (float*) in; outf = (float*) out;

	inThreads = (double**) malloc(threads*sizeof(double*));
	outThreads = (double**) malloc(threads*sizeof(double*));
	inThreadsf = (float**) malloc(threads*sizeof(float*));
	outThreadsf = (float**) malloc(threads*sizeof(float*));
	for(uint64_t i=0;i<threads;i++)
	{
		inThreads[i] = in+2*i*howmanyPerThread*cSize;
		outThreads[i] = out+2*i*howmanyPerThread*cSize;
		inThreadsf[i] = inf+2*i*howmanyPerThread*cSize;
		outThreadsf[i] = outf+2*i*howmanyPerThread*cSize;
	}

	plan = fftw_plan_many_dft_r2c(1, length, howmanyPerThread, in, 
					NULL, 1, (int) size, reinterpret_cast<fftw_complex*>(out),
					NULL, 1, (int) cSize, FFTW_ESTIMATE);

	planf = fftwf_plan_many_dft_r2c(1, length, howmanyPerThread, inf,
					NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(outf),
					NULL, 1, (int) cSize, FFTW_ESTIMATE);

	if(howmany-threads*howmanyPerThread != 0)
	{
		plan2 = fftw_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread, in, 
						NULL, 1, (int) size, reinterpret_cast<fftw_complex*>(out),
						NULL, 1, (int) cSize, FFTW_ESTIMATE);
		plan2f = fftwf_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread,inf,
						NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(outf),
						NULL, 1, (int) cSize, FFTW_ESTIMATE);
	}	
}
		
DigitizerACorrCircularFreqAVX_py::~DigitizerACorrCircularFreqAVX_py()
{
	fftw_free(in);
	fftw_free(out);
	free(inThreads);
	free(outThreads);
	free(inThreadsf);
	free(outThreadsf);
	fftw_destroy_plan(plan);
	fftwf_destroy_plan(planf);
	
	if(howmany-threads*howmanyPerThread != 0)
	{
		fftw_destroy_plan(plan2);
		fftwf_destroy_plan(plan2f);
	}
}

uint64_t DigitizerACorrCircularFreqAVX_py::getSize(){return size;}
uint64_t DigitizerACorrCircularFreqAVX_py::getN(){return Npad;}
uint64_t DigitizerACorrCircularFreqAVX_py::getHowmany(){return howmany;}

void DigitizerACorrCircularFreqAVX_py::train()
{	
	fftw_import_wisdom_from_filename(&wisdom_path[0]);
	fftwf_import_wisdom_from_filename(&wisdom_pathf[0]);
			
	fftw_destroy_plan(plan);
	plan = fftw_plan_many_dft_r2c(1, length, howmany/threads, in, NULL,
					1, (int) size, reinterpret_cast<fftw_complex*>(out),
					NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
	fftw_execute(plan);

	fftwf_destroy_plan(planf);
	planf = fftwf_plan_many_dft_r2c(1, length, howmany/threads, inf,
					NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(outf),
					NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
	fftwf_execute(planf);

	if(howmany-threads*howmanyPerThread != 0)
	{
		fftw_destroy_plan(plan2);
		plan2 = fftw_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread, in, 
						NULL, 1, (int) size, reinterpret_cast<fftw_complex*>(out),
						NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
		fftw_execute(plan2);
				
		fftwf_destroy_plan(plan2f);
		plan2f = fftwf_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread,inf,
						NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(outf),
						NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
		fftwf_execute(plan2f);
	}

	fftw_export_wisdom_to_filename(&wisdom_path[0]);
	fftwf_export_wisdom_to_filename(&wisdom_pathf[0]);
	py::print("Training done.");
}
		
std::tuple<double,double> DigitizerACorrCircularFreqAVX_py::benchmark(uint64_t n)
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

template<class DataType>
py::array_t<double,1> 
DigitizerACorrCircularFreqAVX_py::aCorrCircularFreqAVX
(py::array_t<DataType,1> py_in, double conv, DataType offset)
{			
	py::buffer_info buf_in = py_in.request();
	DataType* py_ptr = (DataType*) buf_in.ptr;

	if (buf_in.ndim != 1){throw std::runtime_error("U dumbdumb dimension must be 1.");}	
	if ((uint64_t) buf_in.size > Npad){throw std::runtime_error("U dumbdumb input too long.");}

	double* result;
	result = (double*) malloc(cSize*sizeof(double));
	std::memset(result,0.0,cSize*sizeof(double));
			
	#pragma omp parallel for
	for(uint64_t i=0;i<threads;i++)
	{
		manage_thread_affinity();
		convertAVX(transferSize,py_ptr+i*transferSize,inThreads[i],conv,offset);
		fftw_execute_dft_r2c(plan,inThreads[i],
						reinterpret_cast<fftw_complex*>(outThreads[i]));
		std::memset(inThreads[i],0.0,cSize*howmanyPerThread*sizeof(double));
		::aCorrCircularFreqAVX(2*cSize*howmanyPerThread,outThreads[i],outThreads[i]);
		::reduceBlockAVX(2*cSize*howmanyPerThread,2*cSize,outThreads[i],inThreads[i]);
					
		for(uint64_t j=0;j<cSize;j++)
		{
			#pragma omp atomic
			result[j] += (inThreads[i][2*j]+inThreads[i][2*j+1])/howmany;
		}
	}
	
	if(howmany-threads*howmanyPerThread != 0)
	{
		convertAVX(size*(howmany-threads*howmanyPerThread),
						py_ptr+threads*transferSize,inThreads[0],conv,offset);
		fftw_execute_dft_r2c(plan2,inThreads[0],
						reinterpret_cast<fftw_complex*>(outThreads[0]));
		::aCorrCircularFreqAVX(2*cSize*(howmany-threads*howmanyPerThread),
						outThreads[0],outThreads[0]);
		::reduceInPlaceBlockAVX(2*cSize*(howmany-threads*howmanyPerThread),
						2*cSize,outThreads[0]);
		for(uint64_t j=0;j<cSize;j++)
		{
			result[j] += (outThreads[0][2*j]+outThreads[0][2*j+1])/howmany;
		}
	}

	py::capsule free_when_done( result, free );
	return py::array_t<double,py::array::c_style>
	(
	{cSize},
	{sizeof(double)},
	result,
	free_when_done	
	);
}

template<class DataType>
py::array_t<float,1> 
DigitizerACorrCircularFreqAVX_py::aCorrCircularFreqAVXf
(py::array_t<DataType,1> py_in, float conv, DataType offset)
{			
	py::buffer_info buf_in = py_in.request();
	DataType* py_ptr = (DataType*) buf_in.ptr;

	if (buf_in.ndim != 1){throw std::runtime_error("U dumbdumb dimension must be 1.");}	
	if ((uint64_t) buf_in.size > Npad){throw std::runtime_error("U dumbdumb input too long.");}

	float* result;
	result = (float*) malloc(cSize*sizeof(float));
	std::memset(result,0.0,cSize*sizeof(float));
			
	#pragma omp parallel for
	for(uint64_t i=0;i<threads;i++)
	{
		manage_thread_affinity();
		convertAVX(transferSize,py_ptr+i*transferSize,inThreadsf[i],conv,offset);
		fftwf_execute_dft_r2c(planf,inThreadsf[i],
						reinterpret_cast<fftwf_complex*>(outThreadsf[i]));
		::aCorrCircularFreqAVX(2*cSize*howmanyPerThread,outThreadsf[i],outThreadsf[i]);
		::reduceInPlaceBlockAVX(2*cSize*howmanyPerThread,2*cSize,outThreadsf[i]);
			
		for(uint64_t j=0;j<cSize;j++)
		{
			#pragma omp atomic
			result[j] += (outThreadsf[i][2*j]+outThreadsf[i][2*j+1])/howmany;
		}
	}
	
	if(howmany-threads*howmanyPerThread != 0)
	{
		convertAVX(size*(howmany-threads*howmanyPerThread),
						py_ptr+threads*transferSize,inThreadsf[0],conv,offset);
		std::memcpy(inThreadsf[0],py_ptr+threads*transferSize,
						size*(howmany-threads*howmanyPerThread)*sizeof(float));
		fftwf_execute_dft_r2c(plan2f,inThreadsf[0],
						reinterpret_cast<fftwf_complex*>(outThreadsf[0]));
		::aCorrCircularFreqAVX(2*cSize*(howmany-threads*howmanyPerThread),
						outThreadsf[0],outThreadsf[0]);
		::reduceInPlaceBlockAVX(2*cSize*(howmany-threads*howmanyPerThread),
						2*cSize,outThreadsf[0]);
		for(uint64_t j=0;j<cSize;j++)
		{
			result[j] += (outThreadsf[0][2*j]+outThreadsf[0][2*j+1])/howmany;
		}
	}
	
	py::capsule free_when_done( result, free );
	return py::array_t<float,py::array::c_style>
	(
	{cSize},
	{sizeof(float)},
	result,
	free_when_done	
	);
}

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

XCorrCircularFreqAVX_py::XCorrCircularFreqAVX_py(uint64_t N_in, uint64_t size_in)
{
	N = N_in;
	size = size_in;
	cSize = size/2+1;
	howmany = N/size;
	length[0] = (int) size;

	#ifdef _WIN32_WINNT
		threads = (uint64_t) omp.omp_get_max_threads()*GetActiveProcessorGroupCount();
	#else
		threads = omp_get_max_threads();
	#endif

	threads = std::min(threads,(uint64_t) 64);
	if(threads > howmany){threads=1;}
	howmanyPerThread = howmany/threads;
			
	transferSize=size*howmanyPerThread;

	in = (double*) fftw_malloc(2*cSize*threads*howmanyPerThread*sizeof(double));
	out1 = (double*) fftw_malloc(2*cSize*threads*howmanyPerThread*sizeof(double));
	out2 = (double*) fftw_malloc(2*cSize*threads*howmanyPerThread*sizeof(double));
	inf = (float*) in; out1f = (float*) out1; out2f = (float*) out2;

	inThreads = (double**) malloc(threads*sizeof(double*));
	outThreads1 = (double**) malloc(threads*sizeof(double*));
	outThreads2 = (double**) malloc(threads*sizeof(double*));
	inThreadsf = (float**) malloc(threads*sizeof(float*));
	outThreads1f = (float**) malloc(threads*sizeof(float*));
	outThreads2f = (float**) malloc(threads*sizeof(float*));
	for(uint64_t i=0;i<threads;i++)
	{
		inThreads[i] = in+2*i*howmanyPerThread*cSize;
		outThreads1[i] = out1+2*i*howmanyPerThread*cSize;
		outThreads2[i] = out2+2*i*howmanyPerThread*cSize;
		inThreadsf[i] = inf+2*i*howmanyPerThread*cSize;
		outThreads1f[i] = out1f+2*i*howmanyPerThread*cSize;
		outThreads2f[i] = out2f+2*i*howmanyPerThread*cSize;
	}

	plan = fftw_plan_many_dft_r2c(1, length, howmanyPerThread, in, 
					NULL, 1, (int) size, reinterpret_cast<fftw_complex*>(out1),
					NULL, 1, (int) cSize, FFTW_ESTIMATE);

	planf = fftwf_plan_many_dft_r2c(1, length, howmanyPerThread, inf,
					NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(out1f),
					NULL, 1, (int) cSize, FFTW_ESTIMATE);

	if(howmany-threads*howmanyPerThread != 0)
	{
		plan2 = fftw_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread, in, 
						NULL, 1, (int) size, reinterpret_cast<fftw_complex*>(out1),
						NULL, 1, (int) cSize, FFTW_ESTIMATE);
		plan2f = fftwf_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread,inf,
						NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(out1f),
						NULL, 1, (int) cSize, FFTW_ESTIMATE);
	}	
}
		
XCorrCircularFreqAVX_py::~XCorrCircularFreqAVX_py()
{
	fftw_free(in);
	fftw_free(out1);
	fftw_free(out2);
	free(inThreads);
	free(outThreads1);
	free(outThreads2);
	free(inThreadsf);
	free(outThreads1f);
	free(outThreads2f);
	fftw_destroy_plan(plan);
	fftwf_destroy_plan(planf);
			
	if(howmany-threads*howmanyPerThread != 0)
	{
		fftw_destroy_plan(plan2);
		fftwf_destroy_plan(plan2f);
	}
}

uint64_t XCorrCircularFreqAVX_py::getSize(){return size;}
uint64_t XCorrCircularFreqAVX_py::getN(){return Npad;}
uint64_t XCorrCircularFreqAVX_py::getHowmany(){return howmany;}

void XCorrCircularFreqAVX_py::train()
{	
	fftw_import_wisdom_from_filename(&wisdom_path[0]);
	fftwf_import_wisdom_from_filename(&wisdom_pathf[0]);
			
	fftw_destroy_plan(plan);
	plan = fftw_plan_many_dft_r2c(1, length, howmany/threads, in, NULL,
					1, (int) size, reinterpret_cast<fftw_complex*>(out1),
					NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
	fftw_execute(plan);

	fftwf_destroy_plan(planf);
	planf = fftwf_plan_many_dft_r2c(1, length, howmany/threads, inf,
					NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(out1f),
					NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
	fftwf_execute(planf);

	if(howmany-threads*howmanyPerThread != 0)
	{
		fftw_destroy_plan(plan2);
		plan2 = fftw_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread, in, 
						NULL, 1, (int) size, reinterpret_cast<fftw_complex*>(out1),
						NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
		fftw_execute(plan2);
				
		fftwf_destroy_plan(plan2f);
		plan2f = fftwf_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread,inf,
						NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(out1f),
						NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
		fftwf_execute(plan2f);
	}

	fftw_export_wisdom_to_filename(&wisdom_path[0]);
	fftwf_export_wisdom_to_filename(&wisdom_pathf[0]);
	py::print("Training done.");
}
		
std::tuple<double,double> XCorrCircularFreqAVX_py::benchmark(uint64_t n)
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

py::array_t<std::complex<double>,1> 
XCorrCircularFreqAVX_py::xCorrCircularFreqAVX
(py::array_t<double,1> py_in1, py::array_t<double,1> py_in2)
{			
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();
	double* py_ptr1 = (double*) buf_in1.ptr;
	double* py_ptr2 = (double*) buf_in2.ptr;

	if (buf_in1.ndim != 1 || buf_in2.ndim != 1 )
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	
	if ((uint64_t) buf_in1.size > Npad || (uint64_t) buf_in2.size > Npad)
	{
		throw std::runtime_error("U dumbdumb input too long.");
	}

	double* result;
	result = (double*) malloc(2*cSize*sizeof(double));
	std::memset(result,0.0,2*cSize*sizeof(double));
			
	#pragma omp parallel for
	for(uint64_t i=0;i<threads;i++)
	{
		manage_thread_affinity();
		std::memcpy(inThreads[i],py_ptr1+i*transferSize,transferSize*sizeof(double));
		fftw_execute_dft_r2c(plan,inThreads[i],reinterpret_cast<fftw_complex*>(outThreads1[i]));
		std::memcpy(inThreads[i],py_ptr2+i*transferSize,transferSize*sizeof(double));
		fftw_execute_dft_r2c(plan,inThreads[i],reinterpret_cast<fftw_complex*>(outThreads2[i]));
		::xCorrCircularFreqAVX(2*cSize*howmanyPerThread,
						outThreads1[i],outThreads2[i],outThreads1[i]);
		::reduceInPlaceBlockAVX(2*cSize*howmanyPerThread,2*cSize,outThreads1[i]);
					
		for(uint64_t j=0;j<(2*cSize);j++)
		{
			#pragma omp atomic
			result[j] += outThreads1[i][j]/howmany;
		}
	}
			
	if(howmany-threads*howmanyPerThread != 0)
	{
		std::memcpy(inThreads[0],py_ptr1+threads*transferSize,
						size*(howmany-threads*howmanyPerThread)*sizeof(double));
		fftw_execute_dft_r2c(plan2,inThreads[0],
						reinterpret_cast<fftw_complex*>(outThreads1[0]));
		std::memcpy(inThreads[0],py_ptr2+threads*transferSize,
						size*(howmany-threads*howmanyPerThread)*sizeof(double));
		fftw_execute_dft_r2c(plan2,inThreads[0],
						reinterpret_cast<fftw_complex*>(outThreads2[0]));
		::xCorrCircularFreqAVX(2*cSize*(howmany-threads*howmanyPerThread),
						outThreads1[0],outThreads2[0],outThreads1[0]);
		::reduceInPlaceBlockAVX(2*cSize*(howmany-threads*howmanyPerThread),
						2*cSize,outThreads1[0]);
		for(uint64_t j=0;j<(2*cSize);j++)
		{
			result[j] += outThreads1[0][j]/howmany;
		}
	}
		py::capsule free_when_done( result, free );
	return py::array_t<std::complex<double>,py::array::c_style>
	(
	{cSize},
	{2*sizeof(double)},
	reinterpret_cast<std::complex<double>*>(result),
	free_when_done	
	);
}

py::array_t<std::complex<float>,1> 
XCorrCircularFreqAVX_py::xCorrCircularFreqAVXf
(py::array_t<float,1> py_in1, py::array_t<float,1> py_in2)
{			
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();
	float* py_ptr1 = (float*) buf_in1.ptr;
	float* py_ptr2 = (float*) buf_in2.ptr;

	if (buf_in1.ndim != 1 || buf_in2.ndim != 1 )
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	
	if ((uint64_t) buf_in1.size > Npad || (uint64_t) buf_in2.size > Npad)
	{
		throw std::runtime_error("U dumbdumb input too long.");
	}

	float* result;
	result = (float*) malloc(2*cSize*sizeof(float));
	std::memset(result,0.0,2*cSize*sizeof(float));
			
	#pragma omp parallel for
	for(uint64_t i=0;i<threads;i++)
	{
		manage_thread_affinity();
		std::memcpy(inThreadsf[i],py_ptr1+i*transferSize,transferSize*sizeof(float));
		fftwf_execute_dft_r2c(planf,inThreadsf[i],
						reinterpret_cast<fftwf_complex*>(outThreads1f[i]));
		std::memcpy(inThreadsf[i],py_ptr2+i*transferSize,transferSize*sizeof(float));
		fftwf_execute_dft_r2c(planf,inThreadsf[i],
						reinterpret_cast<fftwf_complex*>(outThreads2f[i]));
		::xCorrCircularFreqAVX(2*cSize*howmanyPerThread,
						outThreads1f[i],outThreads2f[i],outThreads1f[i]);
		::reduceInPlaceBlockAVX(2*cSize*howmanyPerThread,2*cSize,outThreads1f[i]);
			
		for(uint64_t j=0;j<(2*cSize);j++)
		{
			#pragma omp atomic
			result[j] += outThreads1f[i][j]/howmany;
		}
	}
			
	if(howmany-threads*howmanyPerThread != 0)
	{
		std::memcpy(inThreadsf[0],py_ptr1+threads*transferSize,
						size*(howmany-threads*howmanyPerThread)*sizeof(float));
		fftwf_execute_dft_r2c(plan2f,inThreadsf[0],
						reinterpret_cast<fftwf_complex*>(outThreads1f[0]));
		std::memcpy(inThreadsf[0],py_ptr2+threads*transferSize,
						size*(howmany-threads*howmanyPerThread)*sizeof(float));
		fftwf_execute_dft_r2c(plan2f,inThreadsf[0],
						reinterpret_cast<fftwf_complex*>(outThreads2f[0]));
		::xCorrCircularFreqAVX(2*cSize*(howmany-threads*howmanyPerThread),
						outThreads1f[0],outThreads2f[0],outThreads1f[0]);
		::reduceInPlaceBlockAVX(2*cSize*(howmany-threads*howmanyPerThread),
						2*cSize,outThreads1f[0]);
		for(uint64_t j=0;j<(2*cSize);j++)
		{
			result[j] += outThreads1f[0][j]/howmany;
		}
	}

	py::capsule free_when_done( result, free );
	return py::array_t<std::complex<float>,py::array::c_style>
	(
	{cSize},
	{2*sizeof(float)},
	reinterpret_cast<std::complex<float>*>(result),
	free_when_done	
	);
}	

DigitizerXCorrCircularFreqAVX_py::DigitizerXCorrCircularFreqAVX_py
(uint64_t N_in, uint64_t size_in)
{
	N = N_in;
	size = size_in;
	cSize = size/2+1;
	howmany = N/size;
	length[0] = (int) size;

	#ifdef _WIN32_WINNT
		threads = (uint64_t) omp.omp_get_max_threads()*GetActiveProcessorGroupCount();
	#else
		threads = omp_get_max_threads();
	#endif

	threads = std::min(threads,(uint64_t) 64);
	if(threads > howmany){threads=1;}
	howmanyPerThread = howmany/threads;
			
	transferSize=size*howmanyPerThread;

	in = (double*) fftw_malloc(2*cSize*threads*howmanyPerThread*sizeof(double));
	out1 = (double*) fftw_malloc(2*cSize*threads*howmanyPerThread*sizeof(double));
	out2 = (double*) fftw_malloc(2*cSize*threads*howmanyPerThread*sizeof(double));
	inf = (float*) in; out1f = (float*) out1; out2f = (float*) out2;

	inThreads = (double**) malloc(threads*sizeof(double*));
	outThreads1 = (double**) malloc(threads*sizeof(double*));
	outThreads2 = (double**) malloc(threads*sizeof(double*));
	inThreadsf = (float**) malloc(threads*sizeof(float*));
	outThreads1f = (float**) malloc(threads*sizeof(float*));
	outThreads2f = (float**) malloc(threads*sizeof(float*));
	for(uint64_t i=0;i<threads;i++)
	{
		inThreads[i] = in+2*i*howmanyPerThread*cSize;
		outThreads1[i] = out1+2*i*howmanyPerThread*cSize;
		outThreads2[i] = out2+2*i*howmanyPerThread*cSize;
		inThreadsf[i] = inf+2*i*howmanyPerThread*cSize;
		outThreads1f[i] = out1f+2*i*howmanyPerThread*cSize;
		outThreads2f[i] = out2f+2*i*howmanyPerThread*cSize;
	}

	plan = fftw_plan_many_dft_r2c(1, length, howmanyPerThread, in, 
					NULL, 1, (int) size, reinterpret_cast<fftw_complex*>(out1),
					NULL, 1, (int) cSize, FFTW_ESTIMATE);

	planf = fftwf_plan_many_dft_r2c(1, length, howmanyPerThread, inf,
					NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(out1f),
					NULL, 1, (int) cSize, FFTW_ESTIMATE);

	if(howmany-threads*howmanyPerThread != 0)
	{
		plan2 = fftw_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread, in, 
						NULL, 1, (int) size, reinterpret_cast<fftw_complex*>(out1),
						NULL, 1, (int) cSize, FFTW_ESTIMATE);
		plan2f = fftwf_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread, inf, 
						NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(out1f),
						NULL, 1, (int) cSize, FFTW_ESTIMATE);
	}	
}
		
DigitizerXCorrCircularFreqAVX_py::~DigitizerXCorrCircularFreqAVX_py()
{
	fftw_free(in);
	fftw_free(out1);
	fftw_free(out2);
	free(inThreads);
	free(outThreads1);
	free(outThreads2);
	free(inThreadsf);
	free(outThreads1f);
	free(outThreads2f);
	fftw_destroy_plan(plan);
	fftwf_destroy_plan(planf);
			
	if(howmany-threads*howmanyPerThread != 0)
	{
		fftw_destroy_plan(plan2);
		fftwf_destroy_plan(plan2f);
	}
}

uint64_t DigitizerXCorrCircularFreqAVX_py::getSize(){return size;}
uint64_t DigitizerXCorrCircularFreqAVX_py::getN(){return Npad;}
uint64_t DigitizerXCorrCircularFreqAVX_py::getHowmany(){return howmany;}

void DigitizerXCorrCircularFreqAVX_py::train()
{	
	fftw_import_wisdom_from_filename(&wisdom_path[0]);
	fftwf_import_wisdom_from_filename(&wisdom_pathf[0]);
			
	fftw_destroy_plan(plan);
	plan = fftw_plan_many_dft_r2c(1, length, howmany/threads, in, NULL,
					1, (int) size, reinterpret_cast<fftw_complex*>(out1),
					NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
	fftw_execute(plan);

	fftwf_destroy_plan(planf);
	planf = fftwf_plan_many_dft_r2c(1, length, howmany/threads, inf,
					NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(out1f),
					NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
	fftwf_execute(planf);

	if(howmany-threads*howmanyPerThread != 0)
	{
		fftw_destroy_plan(plan2);
		plan2 = fftw_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread, in, 
						NULL, 1, (int) size, reinterpret_cast<fftw_complex*>(out1),
						NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
		fftw_execute(plan2);		
		fftwf_destroy_plan(plan2f);
				
		plan2f = fftwf_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread, inf, 
						NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(out1f),
						NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
		fftwf_execute(plan2f);
	}

	fftw_export_wisdom_to_filename(&wisdom_path[0]);
	fftwf_export_wisdom_to_filename(&wisdom_pathf[0]);
	py::print("Training done.");
}
		
std::tuple<double,double> DigitizerXCorrCircularFreqAVX_py::benchmark(uint64_t n)
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
		
template<class DataType>
py::array_t<std::complex<double>,1> 
DigitizerXCorrCircularFreqAVX_py::xCorrCircularFreqAVX
(py::array_t<DataType,1> py_in1, py::array_t<DataType,1> py_in2, double conv, DataType offset)
{			
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();
	DataType* py_ptr1 = (DataType*) buf_in1.ptr;
	DataType* py_ptr2 = (DataType*) buf_in2.ptr;

	if (buf_in1.ndim != 1 || buf_in2.ndim != 1 )
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	
	if ((uint64_t) buf_in1.size > Npad || (uint64_t) buf_in2.size > Npad)
	{
		throw std::runtime_error("U dumbdumb input too long.");
	}

	double* result;
	result = (double*) malloc(2*cSize*sizeof(double));
	std::memset(result,0.0,2*cSize*sizeof(double));
			
	#pragma omp parallel for
	for(uint64_t i=0;i<threads;i++)
	{
		manage_thread_affinity();
		convertAVX(transferSize, py_ptr1+i*transferSize,inThreads[i],conv,offset);
		fftw_execute_dft_r2c(plan,inThreads[i], reinterpret_cast<fftw_complex*>(outThreads1[i]));
		convertAVX(transferSize, py_ptr2+i*transferSize,inThreads[i],conv,offset);
		fftw_execute_dft_r2c(plan,inThreads[i], reinterpret_cast<fftw_complex*>(outThreads2[i]));
		::xCorrCircularFreqAVX(2*cSize*howmanyPerThread,
						outThreads1[i],outThreads2[i],outThreads1[i]);
		::reduceInPlaceBlockAVX(2*cSize*howmanyPerThread,2*cSize,outThreads1[i]);
					
		for(uint64_t j=0;j<(2*cSize);j++)
		{
			#pragma omp atomic
			result[j] += outThreads1[i][j]/howmany;
		}
	}
			
	if(howmany-threads*howmanyPerThread != 0)
	{
		convertAVX(size*(howmany-threads*howmanyPerThread), 
						py_ptr1+threads*transferSize,inThreads[0],conv,offset);
		fftw_execute_dft_r2c(plan2,inThreads[0],
						reinterpret_cast<fftw_complex*>(outThreads1[0]));
		convertAVX(size*(howmany-threads*howmanyPerThread), 
						py_ptr2+threads*transferSize,inThreads[0],conv,offset);
		fftw_execute_dft_r2c(plan2,inThreads[0],
						reinterpret_cast<fftw_complex*>(outThreads2[0]));
		::xCorrCircularFreqAVX(2*cSize*(howmany-threads*howmanyPerThread),
						outThreads1[0],outThreads2[0],outThreads1[0]);
		::reduceInPlaceBlockAVX(2*cSize*(howmany-threads*howmanyPerThread),
						2*cSize,outThreads1[0]);
		for(uint64_t j=0;j<(2*cSize);j++)
		{
			result[j] += outThreads1[0][j]/howmany;
		}
	}

	py::capsule free_when_done( result, free );
	return py::array_t<std::complex<double>,py::array::c_style>
	(
	{cSize},
	{2*sizeof(double)},
	reinterpret_cast<std::complex<double>*>(result),
	free_when_done	
	);
}

template<class DataType>
py::array_t<std::complex<float>,1> 
DigitizerXCorrCircularFreqAVX_py::xCorrCircularFreqAVXf
(py::array_t<DataType,1> py_in1, py::array_t<DataType,1> py_in2, float conv, DataType offset)
{			
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();
	DataType* py_ptr1 = (DataType*) buf_in1.ptr;
	DataType* py_ptr2 = (DataType*) buf_in2.ptr;

	if (buf_in1.ndim != 1 || buf_in2.ndim != 1 )
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	
	if ((uint64_t) buf_in1.size > Npad || (uint64_t) buf_in2.size > Npad)
	{
		throw std::runtime_error("U dumbdumb input too long.");
	}

	float* result;
	result = (float*) malloc(2*cSize*sizeof(float));
	std::memset(result,0.0,2*cSize*sizeof(float));
			
	#pragma omp parallel for
	for(uint64_t i=0;i<threads;i++)
	{
		manage_thread_affinity();
		convertAVX(transferSize, py_ptr1+i*transferSize,inThreadsf[i],conv,offset);
		fftwf_execute_dft_r2c(planf,inThreadsf[i],
						reinterpret_cast<fftwf_complex*>(outThreads1f[i]));
		convertAVX(transferSize, py_ptr2+i*transferSize,inThreadsf[i],conv,offset);
		fftwf_execute_dft_r2c(planf,inThreadsf[i],
						reinterpret_cast<fftwf_complex*>(outThreads2f[i]));
		::xCorrCircularFreqAVX(2*cSize*howmanyPerThread,
						outThreads1f[i],outThreads2f[i],outThreads1f[i]);
		::reduceInPlaceBlockAVX(2*cSize*howmanyPerThread,2*cSize,outThreads1f[i]);
					
		for(uint64_t j=0;j<(2*cSize);j++)
		{
			#pragma omp atomic
			result[j] += outThreads1f[i][j]/howmany;
		}
	}
		
	if(howmany-threads*howmanyPerThread != 0)
	{
		convertAVX(size*(howmany-threads*howmanyPerThread), 
						py_ptr1+threads*transferSize,inThreadsf[0],conv,offset);
		fftwf_execute_dft_r2c(plan2f,inThreadsf[0],
						reinterpret_cast<fftwf_complex*>(outThreads1f[0]));
		convertAVX(size*(howmany-threads*howmanyPerThread), 
						py_ptr2+threads*transferSize,inThreadsf[0],conv,offset);
		fftwf_execute_dft_r2c(plan2f,inThreadsf[0],
						reinterpret_cast<fftwf_complex*>(outThreads2f[0]));
		::xCorrCircularFreqAVX(2*cSize*(howmany-threads*howmanyPerThread),
						outThreads1f[0],outThreads2f[0],outThreads1f[0]);
		::reduceInPlaceBlockAVX(2*cSize*(howmany-threads*howmanyPerThread),
						2*cSize,outThreads1f[0]);
		for(uint64_t j=0;j<(2*cSize);j++)
		{
			result[j] += outThreads1f[0][j]/howmany;
		}
	}

	py::capsule free_when_done( result, free );
	return py::array_t<std::complex<float>,py::array::c_style>
	(
	{cSize},
	{2*sizeof(float)},
	reinterpret_cast<std::complex<float>*>(result),
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
fCorrCircularFreqAVX_py(py::array_t<DataType,py::array::c_style> py_in1, 
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
	fCorrCircularFreqAVX<DataType>(2*howmany*(size/2+1),out1+2*(size/2+1),
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

FCorrCircularFreqAVX_py::FCorrCircularFreqAVX_py(uint64_t N_in, uint64_t size_in)
{
	N = N_in;
	size = size_in;
	cSize = size/2+1;
	howmany = N/size;
	length[0] = (int) size;

	#ifdef _WIN32_WINNT
		threads = (uint64_t) omp.omp_get_max_threads()*GetActiveProcessorGroupCount();
	#else
		threads = omp_get_max_threads();
	#endif

	threads = std::min(threads,(uint64_t) 64);
	if(threads > howmany){threads=1;}
	howmanyPerThread = howmany/threads;
			
	transferSize=size*howmanyPerThread;

	in = (double*) fftw_malloc(2*cSize*threads*howmanyPerThread*sizeof(double));
	out1 = (double*) fftw_malloc(2*cSize*threads*howmanyPerThread*sizeof(double));
	out2 = (double*) fftw_malloc(2*cSize*threads*howmanyPerThread*sizeof(double));
	out3 = (double*) fftw_malloc(2*cSize*threads*howmanyPerThread*sizeof(double));
	inf = (float*) in; out1f = (float*) out1; out2f = (float*) out2; out3f = (float*) out3;

	inThreads = (double**) malloc(threads*sizeof(double*));
	outThreads1 = (double**) malloc(threads*sizeof(double*));
	outThreads2 = (double**) malloc(threads*sizeof(double*));
	outThreads3 = (double**) malloc(threads*sizeof(double*));
	inThreadsf = (float**) malloc(threads*sizeof(float*));
	outThreads1f = (float**) malloc(threads*sizeof(float*));
	outThreads2f = (float**) malloc(threads*sizeof(float*));
	outThreads3f = (float**) malloc(threads*sizeof(float*));
	for(uint64_t i=0;i<threads;i++)
	{
		inThreads[i] = in+2*i*howmanyPerThread*cSize;
		outThreads1[i] = out1+2*i*howmanyPerThread*cSize;
		outThreads2[i] = out2+2*i*howmanyPerThread*cSize;
		outThreads3[i] = out3+2*i*howmanyPerThread*cSize;
		inThreadsf[i] = inf+2*i*howmanyPerThread*cSize;
		outThreads1f[i] = out1f+2*i*howmanyPerThread*cSize;
		outThreads2f[i] = out2f+2*i*howmanyPerThread*cSize;
		outThreads3f[i] = out3f+2*i*howmanyPerThread*cSize;
	}

	plan = fftw_plan_many_dft_r2c(1, length, howmanyPerThread, in, 
					NULL, 1, (int) size, reinterpret_cast<fftw_complex*>(out1),
					NULL, 1, (int) cSize, FFTW_ESTIMATE);

	planf = fftwf_plan_many_dft_r2c(1, length, howmanyPerThread, inf,
					NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(out1f),
					NULL, 1, (int) cSize, FFTW_ESTIMATE);

	if(howmany-threads*howmanyPerThread != 0)
	{
		plan2 = fftw_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread, in, 
						NULL, 1, (int) size, reinterpret_cast<fftw_complex*>(out1),
						NULL, 1, (int) cSize, FFTW_ESTIMATE);
		plan2f = fftwf_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread,inf,
						NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(out1f),
						NULL, 1, (int) cSize, FFTW_ESTIMATE);
	}	
}
		
FCorrCircularFreqAVX_py::~FCorrCircularFreqAVX_py()
{
	fftw_free(in);
	fftw_free(out1);
	fftw_free(out2);
	fftw_free(out3);
	free(inThreads);
	free(outThreads1);
	free(outThreads2);
	free(outThreads3);
	free(inThreadsf);
	free(outThreads1f);
	free(outThreads2f);
	free(outThreads3f);
	fftw_destroy_plan(plan);
	fftwf_destroy_plan(planf);
			
	if(howmany-threads*howmanyPerThread != 0)
	{
		fftw_destroy_plan(plan2);
		fftwf_destroy_plan(plan2f);
	}
}

uint64_t FCorrCircularFreqAVX_py::getSize(){return size;}
uint64_t FCorrCircularFreqAVX_py::getN(){return Npad;}
uint64_t FCorrCircularFreqAVX_py::getHowmany(){return howmany;}

void FCorrCircularFreqAVX_py::train()
{	
	fftw_import_wisdom_from_filename(&wisdom_path[0]);
	fftwf_import_wisdom_from_filename(&wisdom_pathf[0]);
			
	fftw_destroy_plan(plan);
	plan = fftw_plan_many_dft_r2c(1, length, howmany/threads, in, NULL,
					1, (int) size, reinterpret_cast<fftw_complex*>(out1),
					NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
	fftw_execute(plan);

	fftwf_destroy_plan(planf);
	planf = fftwf_plan_many_dft_r2c(1, length, howmany/threads, inf,
					NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(out1f),
					NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
	fftwf_execute(planf);

	if(howmany-threads*howmanyPerThread != 0)
	{
		fftw_destroy_plan(plan2);
		plan2 = fftw_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread, in, 
						NULL, 1, (int) size, reinterpret_cast<fftw_complex*>(out1),
						NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
		fftw_execute(plan2);
				
		fftwf_destroy_plan(plan2f);
		plan2f = fftwf_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread,inf,
						NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(out1f),
						NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
		fftwf_execute(plan2f);
	}

	fftw_export_wisdom_to_filename(&wisdom_path[0]);
	fftwf_export_wisdom_to_filename(&wisdom_pathf[0]);
	py::print("Training done.");
}
		
std::tuple<double,double> FCorrCircularFreqAVX_py::benchmark(uint64_t n)
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

std::tuple<py::array_t<double,1>,py::array_t<double,1>,py::array_t<std::complex<double>,1>>
FCorrCircularFreqAVX_py::fCorrCircularFreqAVX
(py::array_t<double,1> py_in1, py::array_t<double,1> py_in2)
{			
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();
	double* py_ptr1 = (double*) buf_in1.ptr;
	double* py_ptr2 = (double*) buf_in2.ptr;

	if (buf_in1.ndim != 1 || buf_in2.ndim != 1 )
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	
	if ((uint64_t) buf_in1.size > Npad || (uint64_t) buf_in2.size > Npad)
	{
		throw std::runtime_error("U dumbdumb input too long.");
	}

	double* result;
	result = (double*) malloc(4*cSize*sizeof(double));
	std::memset(result,0.0,4*cSize*sizeof(double));
			
	#pragma omp parallel for
	for(uint64_t i=0;i<threads;i++)
	{
		manage_thread_affinity();
		std::memcpy(inThreads[i],py_ptr1+i*transferSize,transferSize*sizeof(double));
		fftw_execute_dft_r2c(plan,inThreads[i],reinterpret_cast<fftw_complex*>(outThreads1[i]));
		std::memcpy(inThreads[i],py_ptr2+i*transferSize,transferSize*sizeof(double));
		fftw_execute_dft_r2c(plan,inThreads[i],reinterpret_cast<fftw_complex*>(outThreads2[i]));
		::fCorrCircularFreqAVX(2*cSize*howmanyPerThread,outThreads1[i],outThreads2[i],
						outThreads1[i],outThreads2[i],outThreads3[i]);
		
		::reduceInPlaceBlockAVX(2*cSize*howmanyPerThread,2*cSize,outThreads1[i]);		
		::reduceInPlaceBlockAVX(2*cSize*howmanyPerThread,2*cSize,outThreads2[i]);		
		::reduceInPlaceBlockAVX(2*cSize*howmanyPerThread,2*cSize,outThreads3[i]);		

		for(uint64_t j=0;j<cSize;j++)
		{
			#pragma omp atomic
			result[j] += (outThreads1[i][2*j]+outThreads1[i][2*j+1])/howmany;
			result[j+cSize] += (outThreads2[i][2*j]+outThreads2[i][2*j+1])/howmany;
			result[2*j+2*cSize] += outThreads3[i][2*j]/howmany;
			result[2*j+2*cSize+1] += outThreads3[i][2*j+1]/howmany;
		}
	}		
		
	if(howmany-threads*howmanyPerThread != 0)
	{
		std::memcpy(inThreads[0],py_ptr1+threads*transferSize,
						size*(howmany-threads*howmanyPerThread)*sizeof(double));
		fftw_execute_dft_r2c(plan2,inThreads[0],
						reinterpret_cast<fftw_complex*>(outThreads1[0]));
		std::memcpy(inThreads[0],py_ptr2+threads*transferSize,
						size*(howmany-threads*howmanyPerThread)*sizeof(double));
		fftw_execute_dft_r2c(plan2,inThreads[0],
						reinterpret_cast<fftw_complex*>(outThreads2[0]));
		::fCorrCircularFreqAVX(2*cSize*(howmany-threads*howmanyPerThread),outThreads1[0],
						outThreads2[0],outThreads1[0],outThreads2[0],outThreads3[0]);
		
		::reduceInPlaceBlockAVX(2*cSize*(howmany-threads*howmanyPerThread),
						2*cSize,outThreads1[0]);
		::reduceInPlaceBlockAVX(2*cSize*(howmany-threads*howmanyPerThread),
						2*cSize,outThreads2[0]);
		::reduceInPlaceBlockAVX(2*cSize*(howmany-threads*howmanyPerThread),
						2*cSize,outThreads3[0]);
		for(uint64_t j=0;j<cSize;j++)
		{
			result[j] += (outThreads1[0][2*j]+outThreads1[0][2*j+1])/howmany;
			result[j+cSize] += (outThreads2[0][2*j]+outThreads2[0][2*j+1])/howmany;
			result[2*j+2*cSize] += outThreads3[0][2*j]/howmany;
			result[2*j+2*cSize+1] += outThreads3[0][2*j+1]/howmany;
		}
	}
	
	py::capsule free_when_done1( result, free );
	py::capsule free_when_done2( result+cSize );
	py::capsule free_when_done3( result+2*cSize );
	return std::make_tuple(
	py::array_t<double,1>({cSize},{sizeof(double)},result,free_when_done1),
	py::array_t<double,1>({cSize},{sizeof(double)},result+cSize,free_when_done2),
	py::array_t<std::complex<double>,py::array::c_style>
	({cSize},{2*sizeof(double)},
	reinterpret_cast<std::complex<double>*>(result+2*cSize),free_when_done3)
	);
}

std::tuple<py::array_t<float,1>,py::array_t<float,1>,py::array_t<std::complex<float>,1>>
FCorrCircularFreqAVX_py::fCorrCircularFreqAVXf
(py::array_t<float,1> py_in1, py::array_t<float,1> py_in2)
{			
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();
	float* py_ptr1 = (float*) buf_in1.ptr;
	float* py_ptr2 = (float*) buf_in2.ptr;

	if (buf_in1.ndim != 1 || buf_in2.ndim != 1 )
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	
	if ((uint64_t) buf_in1.size > Npad || (uint64_t) buf_in2.size > Npad)
	{
		throw std::runtime_error("U dumbdumb input too long.");
	}

	float* result;
	result = (float*) malloc(4*cSize*sizeof(float));
	std::memset(result,0.0,4*cSize*sizeof(float));
			
	#pragma omp parallel for
	for(uint64_t i=0;i<threads;i++)
	{
		manage_thread_affinity();
		std::memcpy(inThreadsf[i],py_ptr1+i*transferSize,transferSize*sizeof(float));
		fftwf_execute_dft_r2c(planf,inThreadsf[i],
						reinterpret_cast<fftwf_complex*>(outThreads1f[i]));
		std::memcpy(inThreadsf[i],py_ptr2+i*transferSize,transferSize*sizeof(float));
		fftwf_execute_dft_r2c(planf,inThreadsf[i],
						reinterpret_cast<fftwf_complex*>(outThreads2f[i]));
		::fCorrCircularFreqAVX(2*cSize*howmanyPerThread,outThreads1f[i],outThreads2f[i],
						outThreads1f[i],outThreads2f[i],outThreads3f[i]);
		
		::reduceInPlaceBlockAVX(2*cSize*howmanyPerThread,2*cSize,outThreads1f[i]);		
		::reduceInPlaceBlockAVX(2*cSize*howmanyPerThread,2*cSize,outThreads2f[i]);		
		::reduceInPlaceBlockAVX(2*cSize*howmanyPerThread,2*cSize,outThreads3f[i]);		

		for(uint64_t j=0;j<cSize;j++)
		{
			#pragma omp atomic
			result[j] += (outThreads1f[i][2*j]+outThreads1f[i][2*j+1])/howmany;
			result[j+cSize] += (outThreads2f[i][2*j]+outThreads2f[i][2*j+1])/howmany;
			result[2*j+2*cSize] += outThreads3f[i][2*j]/howmany;
			result[2*j+2*cSize+1] += outThreads3f[i][2*j+1]/howmany;
		}
	}
			
	if(howmany-threads*howmanyPerThread != 0)
	{
		std::memcpy(inThreadsf[0],py_ptr1+threads*transferSize,
						size*(howmany-threads*howmanyPerThread)*sizeof(float));
		fftwf_execute_dft_r2c(plan2f,inThreadsf[0],
						reinterpret_cast<fftwf_complex*>(outThreads1f[0]));
		std::memcpy(inThreadsf[0],py_ptr2+threads*transferSize,
						size*(howmany-threads*howmanyPerThread)*sizeof(float));
		fftwf_execute_dft_r2c(plan2f,inThreadsf[0],
						reinterpret_cast<fftwf_complex*>(outThreads2f[0]));
		::fCorrCircularFreqAVX(2*cSize*(howmany-threads*howmanyPerThread),outThreads1f[0],
						outThreads2f[0],outThreads1f[0],outThreads2f[0],outThreads3f[0]);
		
		::reduceInPlaceBlockAVX(2*cSize*(howmany-threads*howmanyPerThread),
						2*cSize,outThreads1f[0]);
		::reduceInPlaceBlockAVX(2*cSize*(howmany-threads*howmanyPerThread),
						2*cSize,outThreads2f[0]);
		::reduceInPlaceBlockAVX(2*cSize*(howmany-threads*howmanyPerThread),
						2*cSize,outThreads3f[0]);
		for(uint64_t j=0;j<cSize;j++)
		{
			result[j] += (outThreads1f[0][2*j]+outThreads1f[0][2*j+1])/howmany;
			result[j+cSize] += (outThreads2f[0][2*j]+outThreads2f[0][2*j+1])/howmany;
			result[2*j+2*cSize] += outThreads3f[0][2*j]/howmany;
			result[2*j+2*cSize+1] += outThreads3f[0][2*j+1]/howmany;
		}
	}
	
	py::capsule free_when_done1( result, free );
	py::capsule free_when_done2( result+cSize );
	py::capsule free_when_done3( result+2*cSize );
	return std::make_tuple(
	py::array_t<float,1>({cSize},{sizeof(float)},result,free_when_done1),
	py::array_t<float,1>({cSize},{sizeof(float)},result+cSize,free_when_done2),
	py::array_t<std::complex<float>,py::array::c_style>
	({cSize},{2*sizeof(float)},
	reinterpret_cast<std::complex<float>*>(result+2*cSize),free_when_done3)
	);
}	

DigitizerFCorrCircularFreqAVX_py::DigitizerFCorrCircularFreqAVX_py
(uint64_t N_in, uint64_t size_in)
{
	N = N_in;
	size = size_in;
	cSize = size/2+1;
	howmany = N/size;
	length[0] = (int) size;

	#ifdef _WIN32_WINNT
		threads = (uint64_t) omp.omp_get_max_threads()*GetActiveProcessorGroupCount();
	#else
		threads = omp_get_max_threads();
	#endif

	threads = std::min(threads,(uint64_t) 64);
	if(threads > howmany){threads=1;}
	howmanyPerThread = howmany/threads;
			
	transferSize=size*howmanyPerThread;

	in = (double*) fftw_malloc(2*cSize*threads*howmanyPerThread*sizeof(double));
	out1 = (double*) fftw_malloc(2*cSize*threads*howmanyPerThread*sizeof(double));
	out2 = (double*) fftw_malloc(2*cSize*threads*howmanyPerThread*sizeof(double));
	out3 = (double*) fftw_malloc(2*cSize*threads*howmanyPerThread*sizeof(double));
	inf = (float*) in; out1f = (float*) out1; out2f = (float*) out2; out3f = (float*) out3;

	inThreads = (double**) malloc(threads*sizeof(double*));
	outThreads1 = (double**) malloc(threads*sizeof(double*));
	outThreads2 = (double**) malloc(threads*sizeof(double*));
	outThreads3 = (double**) malloc(threads*sizeof(double*));
	inThreadsf = (float**) malloc(threads*sizeof(float*));
	outThreads1f = (float**) malloc(threads*sizeof(float*));
	outThreads2f = (float**) malloc(threads*sizeof(float*));
	outThreads3f = (float**) malloc(threads*sizeof(float*));
	for(uint64_t i=0;i<threads;i++)
	{
		inThreads[i] = in+2*i*howmanyPerThread*cSize;
		outThreads1[i] = out1+2*i*howmanyPerThread*cSize;
		outThreads2[i] = out2+2*i*howmanyPerThread*cSize;
		outThreads3[i] = out3+2*i*howmanyPerThread*cSize;
		inThreadsf[i] = inf+2*i*howmanyPerThread*cSize;
		outThreads1f[i] = out1f+2*i*howmanyPerThread*cSize;
		outThreads2f[i] = out2f+2*i*howmanyPerThread*cSize;
		outThreads3f[i] = out3f+2*i*howmanyPerThread*cSize;
	}

	plan = fftw_plan_many_dft_r2c(1, length, howmanyPerThread, in, 
					NULL, 1, (int) size, reinterpret_cast<fftw_complex*>(out1),
					NULL, 1, (int) cSize, FFTW_ESTIMATE);

	planf = fftwf_plan_many_dft_r2c(1, length, howmanyPerThread, inf,
					NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(out1f),
					NULL, 1, (int) cSize, FFTW_ESTIMATE);

	if(howmany-threads*howmanyPerThread != 0)
	{
		plan2 = fftw_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread, in, 
						NULL, 1, (int) size, reinterpret_cast<fftw_complex*>(out1),
						NULL, 1, (int) cSize, FFTW_ESTIMATE);
		plan2f = fftwf_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread,inf,
						NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(out1f),
						NULL, 1, (int) cSize, FFTW_ESTIMATE);
	}	
}
		
DigitizerFCorrCircularFreqAVX_py::~DigitizerFCorrCircularFreqAVX_py()
{
	fftw_free(in);
	fftw_free(out1);
	fftw_free(out2);
	fftw_free(out3);
	free(inThreads);
	free(outThreads1);
	free(outThreads2);
	free(outThreads3);
	free(inThreadsf);
	free(outThreads1f);
	free(outThreads2f);
	free(outThreads3f);
	fftw_destroy_plan(plan);
	fftwf_destroy_plan(planf);
			
	if(howmany-threads*howmanyPerThread != 0)
	{
		fftw_destroy_plan(plan2);
		fftwf_destroy_plan(plan2f);
	}
}

uint64_t DigitizerFCorrCircularFreqAVX_py::getSize(){return size;}
uint64_t DigitizerFCorrCircularFreqAVX_py::getN(){return Npad;}
uint64_t DigitizerFCorrCircularFreqAVX_py::getHowmany(){return howmany;}

void DigitizerFCorrCircularFreqAVX_py::train()
{	
	fftw_import_wisdom_from_filename(&wisdom_path[0]);
	fftwf_import_wisdom_from_filename(&wisdom_pathf[0]);
			
	fftw_destroy_plan(plan);
	plan = fftw_plan_many_dft_r2c(1, length, howmany/threads, in, NULL,
					1, (int) size, reinterpret_cast<fftw_complex*>(out1),
					NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
	fftw_execute(plan);

	fftwf_destroy_plan(planf);
	planf = fftwf_plan_many_dft_r2c(1, length, howmany/threads, inf,
					NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(out1f),
					NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
	fftwf_execute(planf);

	if(howmany-threads*howmanyPerThread != 0)
	{
		fftw_destroy_plan(plan2);
		plan2 = fftw_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread, in, 
						NULL, 1, (int) size, reinterpret_cast<fftw_complex*>(out1),
						NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
		fftw_execute(plan2);
				
		fftwf_destroy_plan(plan2f);
		plan2f = fftwf_plan_many_dft_r2c(1, length, howmany-threads*howmanyPerThread,inf,
						NULL, 1, (int) size, reinterpret_cast<fftwf_complex*>(out1f),
						NULL, 1, (int) cSize, FFTW_EXHAUSTIVE);
		fftwf_execute(plan2f);
	}

	fftw_export_wisdom_to_filename(&wisdom_path[0]);
	fftwf_export_wisdom_to_filename(&wisdom_pathf[0]);
	py::print("Training done.");
}
		
std::tuple<double,double> DigitizerFCorrCircularFreqAVX_py::benchmark(uint64_t n)
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

template<class DataType>
std::tuple<py::array_t<double,1>,py::array_t<double,1>,py::array_t<std::complex<double>,1>>
DigitizerFCorrCircularFreqAVX_py::fCorrCircularFreqAVX
(py::array_t<DataType,1> py_in1, py::array_t<DataType,1> py_in2, double conv, DataType offset)
{			
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();
	DataType* py_ptr1 = (DataType*) buf_in1.ptr;
	DataType* py_ptr2 = (DataType*) buf_in2.ptr;

	if (buf_in1.ndim != 1 || buf_in2.ndim != 1 )
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	
	if ((uint64_t) buf_in1.size > Npad || (uint64_t) buf_in2.size > Npad)
	{
		throw std::runtime_error("U dumbdumb input too long.");
	}

	double* result;
	result = (double*) malloc(4*cSize*sizeof(double));
	std::memset(result,0.0,4*cSize*sizeof(double));
			
	#pragma omp parallel for
	for(uint64_t i=0;i<threads;i++)
	{
		manage_thread_affinity();
		convertAVX(transferSize, py_ptr1+i*transferSize,inThreads[i],conv,offset);
		fftw_execute_dft_r2c(plan,inThreads[i],reinterpret_cast<fftw_complex*>(outThreads1[i]));
		convertAVX(transferSize, py_ptr2+i*transferSize,inThreads[i],conv,offset);
		fftw_execute_dft_r2c(plan,inThreads[i],reinterpret_cast<fftw_complex*>(outThreads2[i]));
		::fCorrCircularFreqAVX(2*cSize*howmanyPerThread,outThreads1[i],outThreads2[i],
						outThreads1[i],outThreads2[i],outThreads3[i]);
		
		::reduceInPlaceBlockAVX(2*cSize*howmanyPerThread,2*cSize,outThreads1[i]);		
		::reduceInPlaceBlockAVX(2*cSize*howmanyPerThread,2*cSize,outThreads2[i]);		
		::reduceInPlaceBlockAVX(2*cSize*howmanyPerThread,2*cSize,outThreads3[i]);		

		for(uint64_t j=0;j<cSize;j++)
		{
			#pragma omp atomic
			result[j] += (outThreads1[i][2*j]+outThreads1[i][2*j+1])/howmany;
			result[j+cSize] += (outThreads2[i][2*j]+outThreads2[i][2*j+1])/howmany;
			result[2*j+2*cSize] += outThreads3[i][2*j]/howmany;
			result[2*j+2*cSize+1] += outThreads3[i][2*j+1]/howmany;
		}
	}
		
		
	if(howmany-threads*howmanyPerThread != 0)
	{
		convertAVX(size*(howmany-threads*howmanyPerThread), 
						py_ptr1+threads*transferSize,inThreads[0],conv,offset);
		fftw_execute_dft_r2c(plan2,inThreads[0],
						reinterpret_cast<fftw_complex*>(outThreads1[0]));
		convertAVX(size*(howmany-threads*howmanyPerThread), 
						py_ptr2+threads*transferSize,inThreads[0],conv,offset);
		fftw_execute_dft_r2c(plan2,inThreads[0],
						reinterpret_cast<fftw_complex*>(outThreads2[0]));
		::fCorrCircularFreqAVX(2*cSize*(howmany-threads*howmanyPerThread),outThreads1[0],
						outThreads2[0],outThreads1[0],outThreads2[0],outThreads3[0]);
		
		::reduceInPlaceBlockAVX(2*cSize*(howmany-threads*howmanyPerThread),
						2*cSize,outThreads1[0]);
		::reduceInPlaceBlockAVX(2*cSize*(howmany-threads*howmanyPerThread),
						2*cSize,outThreads2[0]);
		::reduceInPlaceBlockAVX(2*cSize*(howmany-threads*howmanyPerThread),
						2*cSize,outThreads3[0]);
		for(uint64_t j=0;j<cSize;j++)
		{
			result[j] += (outThreads1[0][2*j]+outThreads1[0][2*j+1])/howmany;
			result[j+cSize] += (outThreads2[0][2*j]+outThreads2[0][2*j+1])/howmany;
			result[2*j+2*cSize] += outThreads3[0][2*j]/howmany;
			result[2*j+2*cSize+1] += outThreads3[0][2*j+1]/howmany;
		}
	}
	
	py::capsule free_when_done1( result, free );
	py::capsule free_when_done2( result+cSize );
	py::capsule free_when_done3( result+2*cSize );
	return std::make_tuple(
	py::array_t<double,1>({cSize},{sizeof(double)},result,free_when_done1),
	py::array_t<double,1>({cSize},{sizeof(double)},result+cSize,free_when_done2),
	py::array_t<std::complex<double>,py::array::c_style>
	({cSize},{2*sizeof(double)},
	reinterpret_cast<std::complex<double>*>(result+2*cSize),free_when_done3)
	);
}

template<class DataType>
std::tuple<py::array_t<float,1>,py::array_t<float,1>,py::array_t<std::complex<float>,1>>
DigitizerFCorrCircularFreqAVX_py::fCorrCircularFreqAVXf
(py::array_t<DataType,1> py_in1, py::array_t<DataType,1> py_in2, float conv, DataType offset)
{			
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();
	DataType* py_ptr1 = (DataType*) buf_in1.ptr;
	DataType* py_ptr2 = (DataType*) buf_in2.ptr;

	if (buf_in1.ndim != 1 || buf_in2.ndim != 1 )
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	
	if ((uint64_t) buf_in1.size > Npad || (uint64_t) buf_in2.size > Npad)
	{
		throw std::runtime_error("U dumbdumb input too long.");
	}

	float* result;
	result = (float*) malloc(4*cSize*sizeof(float));
	std::memset(result,0.0,4*cSize*sizeof(float));
			
	#pragma omp parallel for
	for(uint64_t i=0;i<threads;i++)
	{
		manage_thread_affinity();
		convertAVX(transferSize, py_ptr1+i*transferSize,inThreadsf[i],conv,offset);
		fftwf_execute_dft_r2c(planf,inThreadsf[i],
						reinterpret_cast<fftwf_complex*>(outThreads1f[i]));
		convertAVX(transferSize, py_ptr2+i*transferSize,inThreadsf[i],conv,offset);
		fftwf_execute_dft_r2c(planf,inThreadsf[i],
						reinterpret_cast<fftwf_complex*>(outThreads2f[i]));
		::fCorrCircularFreqAVX(2*cSize*howmanyPerThread,outThreads1f[i],outThreads2f[i],
						outThreads1f[i],outThreads2f[i],outThreads3f[i]);
		

		::reduceInPlaceBlockAVX(2*cSize*howmanyPerThread,2*cSize,outThreads1f[i]);		
		::reduceInPlaceBlockAVX(2*cSize*howmanyPerThread,2*cSize,outThreads2f[i]);		
		::reduceInPlaceBlockAVX(2*cSize*howmanyPerThread,2*cSize,outThreads3f[i]);		

		for(uint64_t j=0;j<cSize;j++)
		{
			#pragma omp atomic
			result[j] += (outThreads1f[i][2*j]+outThreads1f[i][2*j+1])/howmany;
			result[j+cSize] += (outThreads2f[i][2*j]+outThreads2f[i][2*j+1])/howmany;
			result[2*j+2*cSize] += outThreads3f[i][2*j]/howmany;
			result[2*j+2*cSize+1] += outThreads3f[i][2*j+1]/howmany;
		}

	}
			
	if(howmany-threads*howmanyPerThread != 0)
	{
		convertAVX(size*(howmany-threads*howmanyPerThread), 
						py_ptr1+threads*transferSize,inThreadsf[0],conv,offset);
		fftwf_execute_dft_r2c(plan2f,inThreadsf[0],
						reinterpret_cast<fftwf_complex*>(outThreads1f[0]));
		convertAVX(size*(howmany-threads*howmanyPerThread), 
						py_ptr2+threads*transferSize,inThreadsf[0],conv,offset);
		fftwf_execute_dft_r2c(plan2f,inThreadsf[0],
						reinterpret_cast<fftwf_complex*>(outThreads2f[0]));
		::fCorrCircularFreqAVX(2*cSize*(howmany-threads*howmanyPerThread),outThreads1f[0],
						outThreads2f[0],outThreads1f[0],outThreads2f[0],outThreads3f[0]);
		
		::reduceInPlaceBlockAVX(2*cSize*(howmany-threads*howmanyPerThread),
						2*cSize,outThreads1f[0]);
		::reduceInPlaceBlockAVX(2*cSize*(howmany-threads*howmanyPerThread),
						2*cSize,outThreads2f[0]);
		::reduceInPlaceBlockAVX(2*cSize*(howmany-threads*howmanyPerThread),
						2*cSize,outThreads3f[0]);
		for(uint64_t j=0;j<cSize;j++)
		{
			result[j] += (outThreads1f[0][2*j]+outThreads1f[0][2*j+1])/howmany;
			result[j+cSize] += (outThreads2f[0][2*j]+outThreads2f[0][2*j+1])/howmany;
			result[2*j+2*cSize] += outThreads3f[0][2*j]/howmany;
			result[2*j+2*cSize+1] += outThreads3f[0][2*j+1]/howmany;
		}
	}
	
	py::capsule free_when_done1( result, free );
	py::capsule free_when_done2( result+cSize );
	py::capsule free_when_done3( result+2*cSize );
	return std::make_tuple(
	py::array_t<float,1>({cSize},{sizeof(float)},result,free_when_done1),
	py::array_t<float,1>({cSize},{sizeof(float)},result+cSize,free_when_done2),
	py::array_t<std::complex<float>,py::array::c_style>
	({cSize},{2*sizeof(float)},
	reinterpret_cast<std::complex<float>*>(result+2*cSize),free_when_done3)
	);
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
