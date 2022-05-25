template< class DataType >
DataType FFT_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	dbl_complex* ptr_py_in = (dbl_complex*) buf_in.ptr;
	dbl_complex* result = (dbl_complex*) fftw_malloc(sizeof(dbl_complex)*n);

	FFT(n, ptr_py_in, result);

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{n},
		{sizeof(dbl_complex)},
		result,
		free_when_done	
	);
}

template< class DataType >
DataType FFT_Parallel_py(DataType py_in, int nthreads)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	dbl_complex* ptr_py_in = (dbl_complex*) buf_in.ptr;
	dbl_complex* result = (dbl_complex*) fftw_malloc(sizeof(dbl_complex)*n);

	FFT_Parallel(n, ptr_py_in, result, nthreads);

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{n},
		{sizeof(dbl_complex)},
		result,
		free_when_done	
	);
}

template< class DataType ,class DataType2>
DataType FFT_py(DataType py_in,int N)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;

	if (N > 4096 )
	{
		throw std::runtime_error("U dumbdumb N too big, can't optimize");
	}

	int howmany = n/N;

	dbl_complex* ptr_py_in = (dbl_complex*) buf_in.ptr;
	dbl_complex* result = (dbl_complex*) fftw_malloc(sizeof(dbl_complex)*N*howmany);

	FFT_Block(n, N, ptr_py_in, result);

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{N*howmany},
		{sizeof(dbl_complex)},
		result,
		free_when_done	
	);
}

template< class DataType ,class DataType2>
DataType FFT_Block_Parallel_py(DataType py_in,int N, int nthreads)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;

	if (N > 4096 )
	{
		throw std::runtime_error("U dumbdumb N too big, can't optimize");
	}

	int howmany = n/N;

	dbl_complex* ptr_py_in = (dbl_complex*) buf_in.ptr;
	dbl_complex* result = (dbl_complex*) fftw_malloc(sizeof(dbl_complex)*N*howmany);

	FFT_Block_Parallel(n, N, ptr_py_in, result, nthreads);

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{N*howmany},
		{sizeof(dbl_complex)},
		result,
		free_when_done	
	);
}

template< class DataType >
DataType iFFT_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;

	dbl_complex* ptr_py_in = (dbl_complex*) buf_in.ptr;
	dbl_complex* result = (dbl_complex*) fftw_malloc(sizeof(dbl_complex)*n);

	iFFT(n, ptr_py_in, result);

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{n},
		{sizeof(dbl_complex)},
		result,
		free_when_done	
	);

}

template< class DataType >
np_complex rFFT_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;

	double* ptr_py_in = (double*) buf_in.ptr;
	dbl_complex* result = (dbl_complex*) fftw_malloc(sizeof(dbl_complex)*(n/2+1));

	rFFT(n, ptr_py_in, result);

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{n/2+1},
		{sizeof(dbl_complex)},
		result,
		free_when_done	
	);

}

template< class DataType ,class DataType2>
np_complex rFFT_py(DataType py_in,int N)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;

	if (N > 4096 )
	{
		throw std::runtime_error("U dumbdumb N too big, can't optimize");
	}
	int howmany = n/N;
	
	double* ptr_py_in = (double*) buf_in.ptr;
	dbl_complex* result = (dbl_complex*) fftw_malloc(sizeof(dbl_complex)*(N/2+1)*howmany);

	rFFT_Block(n, N, ptr_py_in, result);

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{(N/2+1)*howmany},
		{sizeof(dbl_complex)},
		result,
		free_when_done	
	);
}

template< class DataType >
np_double irFFT_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;

	dbl_complex* ptr_py_in = (dbl_complex*) buf_in.ptr;
	double* result = (double*) fftw_malloc(sizeof(double)*2*(n-1));

	irFFT(2*(n-1), ptr_py_in, result);

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<double, py::array::c_style> 
	(
		{2*(n-1)}, //Shape
		{sizeof(double)}, //Stride
		result, //Pointer
		free_when_done //mem clear
	);

}


