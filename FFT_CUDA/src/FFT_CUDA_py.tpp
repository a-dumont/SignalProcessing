template< class DataType >
DataType FFT_CUDA_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	dbl_complex* ptr_py_in = (dbl_complex*) buf_in.ptr;
	dbl_complex* out;

	cudaMallocManaged((void**)&out, n*sizeof(dbl_complex));
	
	cudaMemcpy(out,ptr_py_in,sizeof(dbl_complex)*n,cudaMemcpyHostToDevice);
	
	FFT_CUDA(n,out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{n},
		{sizeof(dbl_complex)},
		out,
		free_when_done	
	);
}

template< class DataType >
DataType fFFT_CUDA_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	flt_complex* ptr_py_in = (flt_complex*) buf_in.ptr;
	flt_complex* out;

	cudaMallocManaged((void**)&out, n*sizeof(flt_complex));
	
	cudaMemcpy(out,ptr_py_in,sizeof(flt_complex)*n,cudaMemcpyHostToDevice);
	
	fFFT_CUDA(n,out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<flt_complex, py::array::c_style> 
	(
		{n},
		{sizeof(flt_complex)},
		out,
		free_when_done	
	);
}

template< class DataType ,class DataType2>
DataType FFT_Block_CUDA_py(DataType py_in,long long int N)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = (long long int) buf_in.size;
	long long int howmany = (long long int) n/N;

	dbl_complex* ptr_py_in = (dbl_complex*) buf_in.ptr;
	dbl_complex* out;

	cudaMallocManaged((void**)&out, sizeof(dbl_complex)*N*howmany);

	cudaMemcpy(out,ptr_py_in,sizeof(dbl_complex)*N*howmany,cudaMemcpyHostToDevice);

	FFT_Block_CUDA(n, N, out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{N*howmany},
		{sizeof(dbl_complex)},
		out,
		free_when_done	
	);
}

template< class DataType ,class DataType2>
DataType fFFT_Block_CUDA_py(DataType py_in,long long int N)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = (long long int) buf_in.size;
	long long int howmany = (long long int) n/N;

	flt_complex* ptr_py_in = (flt_complex*) buf_in.ptr;
	flt_complex* out;

	cudaMallocManaged((void**)&out, sizeof(flt_complex)*N*howmany);

	cudaMemcpy(out,ptr_py_in,sizeof(flt_complex)*N*howmany,cudaMemcpyHostToDevice);

	fFFT_Block_CUDA(n, N, out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<flt_complex, py::array::c_style> 
	(
		{N*howmany},
		{sizeof(flt_complex)},
		out,
		free_when_done	
	);
}

template< class DataType >
DataType iFFT_CUDA_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	dbl_complex* ptr_py_in = (dbl_complex*) buf_in.ptr;
	dbl_complex* out;

	cudaMallocManaged((void**)&out, n*sizeof(dbl_complex));
	
	cudaMemcpy(out,ptr_py_in,sizeof(dbl_complex)*n,cudaMemcpyHostToDevice);
	
	iFFT_CUDA(n,out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{n},
		{sizeof(dbl_complex)},
		out,
		free_when_done	
	);
}

template< class DataType >
DataType ifFFT_CUDA_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	flt_complex* ptr_py_in = (flt_complex*) buf_in.ptr;
	flt_complex* out;

	cudaMallocManaged((void**)&out, n*sizeof(dbl_complex));
	
	cudaMemcpy(out,ptr_py_in,sizeof(flt_complex)*n,cudaMemcpyHostToDevice);
	
	ifFFT_CUDA(n,out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<flt_complex, py::array::c_style> 
	(
		{n},
		{sizeof(flt_complex)},
		out,
		free_when_done	
	);
}


template< class DataType >
np_complex rFFT_CUDA_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	double* ptr_py_in = (double*) buf_in.ptr;
	dbl_complex* out;

	cudaMallocManaged((void**)&out, (n/2+1)*sizeof(dbl_complex));
	
	cudaMemcpy(out,ptr_py_in,sizeof(double)*n,cudaMemcpyHostToDevice);
	
	rFFT_CUDA(n,out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{n/2+1},
		{sizeof(dbl_complex)},
		out,
		free_when_done	
	);
}

template< class DataType >
np_fcomplex rfFFT_CUDA_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	float* ptr_py_in = (float*) buf_in.ptr;
	flt_complex* out;

	cudaMallocManaged((void**)&out, (n/2+1)*sizeof(flt_complex));
	
	cudaMemcpy(out,ptr_py_in,sizeof(float)*n,cudaMemcpyHostToDevice);
	
	rfFFT_CUDA(n,out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<flt_complex, py::array::c_style> 
	(
		{n/2+1},
		{sizeof(flt_complex)},
		out,
		free_when_done	
	);
}


template< class DataType ,class DataType2>
np_complex rFFT_Block_CUDA_py(DataType py_in,int N)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = (long long int) buf_in.size;
	long long int howmany = (long long int) n/N;

	dbl_complex* ptr_py_in = (dbl_complex*) buf_in.ptr;
	dbl_complex* out;

	cudaMallocManaged((void**)&out, sizeof(dbl_complex)*(N/2+1)*howmany);
	cudaMemcpy2D(out,(N+2)*sizeof(double),ptr_py_in,N*sizeof(double),
					sizeof(double)*N,howmany,cudaMemcpyHostToDevice);

	rFFT_Block_CUDA(n, N, out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{(N/2+1)*howmany},
		{sizeof(dbl_complex)},
		out,
		free_when_done	
	);
}

template< class DataType ,class DataType2>
np_fcomplex rfFFT_Block_CUDA_py(DataType py_in,int N)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = (long long int) buf_in.size;
	long long int howmany = (long long int) n/N;

	flt_complex* ptr_py_in = (flt_complex*) buf_in.ptr;
	flt_complex* out;

	cudaMallocManaged((void**)&out, sizeof(flt_complex)*(N/2+1)*howmany);
	cudaMemcpy2D(out,(N+2)*sizeof(float),ptr_py_in,N*sizeof(float),
					sizeof(float)*N,howmany,cudaMemcpyHostToDevice);

	rfFFT_Block_CUDA(n, N, out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<flt_complex, py::array::c_style> 
	(
		{(N/2+1)*howmany},
		{sizeof(flt_complex)},
		out,
		free_when_done	
	);
}


template< class DataType >
np_double irFFT_CUDA_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	dbl_complex* ptr_py_in = (dbl_complex*) buf_in.ptr;
	double* out;

	cudaMallocManaged((void**)&out, 2*n*sizeof(double));
	
	cudaMemcpy(out,ptr_py_in,sizeof(dbl_complex)*n,cudaMemcpyHostToDevice);
	
	irFFT_CUDA(2*(n-1), out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<double, py::array::c_style> 
	(
		{2*(n-1)},
		{sizeof(double)},
		out,
		free_when_done	
	);
}

template< class DataType >
np_float irfFFT_CUDA_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	flt_complex* ptr_py_in = (flt_complex*) buf_in.ptr;
	float* out;

	cudaMallocManaged((void**)&out, 2*n*sizeof(float));
	
	cudaMemcpy(out,ptr_py_in,sizeof(flt_complex)*n,cudaMemcpyHostToDevice);
	
	irfFFT_CUDA(2*(n-1), out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<float, py::array::c_style> 
	(
		{2*(n-1)},
		{sizeof(float)},
		out,
		free_when_done	
	);
}
