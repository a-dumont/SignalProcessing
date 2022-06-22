template< class DataType >
py::array_t<std::complex<DataType>,py::array::c_style> 
FFT_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	std::complex<DataType>* ptr_py_in = (std::complex<DataType>*) buf_in.ptr;
	std::complex<DataType>* result = (std::complex<DataType>*) fftw_malloc(
					sizeof(std::complex<DataType>)*n);

	FFT(n, ptr_py_in, result);

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{n},
		{sizeof(std::complex<DataType>)},
		result,
		free_when_done	
	);
}

template< class DataType >
py::array_t<std::complex<DataType>,py::array::c_style> 
FFT_Parallel_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in, int nthreads)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	std::complex<DataType>* ptr_py_in = (std::complex<DataType>*) buf_in.ptr;
	std::complex<DataType>* result = (std::complex<DataType>*) fftw_malloc(
					sizeof(std::complex<DataType>)*n);

	FFT_Parallel(n, ptr_py_in, result, nthreads);

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{n},
		{sizeof(std::complex<DataType>)},
		result,
		free_when_done	
	);
}

template< class DataType, class DataType2>
py::array_t<std::complex<DataType>,py::array::c_style> 
FFT_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in, DataType2 N)
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

	std::complex<DataType>* ptr_py_in = (std::complex<DataType>*) buf_in.ptr;
	std::complex<DataType>* result = (std::complex<DataType>*) fftw_malloc(
					sizeof(std::complex<DataType>)*N*howmany);

	FFT_Block(n, N, ptr_py_in, result);

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{N*howmany},
		{sizeof(std::complex<DataType>)},
		result,
		free_when_done	
	);
}

template< class DataType ,class DataType2>
py::array_t<std::complex<DataType>,py::array::c_style> 
FFT_Block_Parallel_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in,
				int N, int nthreads)
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

	std::complex<DataType>* ptr_py_in = (std::complex<DataType>*) buf_in.ptr;
	std::complex<DataType>* result = (std::complex<DataType>*) fftw_malloc(
					sizeof(std::complex<DataType>)*N*howmany);

	FFT_Block_Parallel(n, N, ptr_py_in, result, nthreads);

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{N*howmany},
		{sizeof(std::complex<DataType>)},
		result,
		free_when_done	
	);
}

template< class DataType ,class DataType2>
py::array_t<std::complex<DataType>,py::array::c_style> 
FFT_Block_Parallel2_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in,int N)
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

	std::complex<DataType>* ptr_py_in = (std::complex<DataType>*) buf_in.ptr;
	std::complex<DataType>* result = (std::complex<DataType>*) fftw_malloc(
					sizeof(std::complex<DataType>)*N*howmany);

	FFT_Block_Parallel2(n, N, ptr_py_in, result);

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{N*howmany},
		{sizeof(std::complex<DataType>)},
		result,
		free_when_done	
	);
}

template< class DataType >
py::array_t<std::complex<DataType>,py::array::c_style> 
iFFT_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;

	std::complex<DataType>* ptr_py_in = (std::complex<DataType>*) buf_in.ptr;
	std::complex<DataType>* result = (std::complex<DataType>*) fftw_malloc(
					sizeof(std::complex<DataType>)*n);

	iFFT(n, ptr_py_in, result);

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{n},
		{sizeof(std::complex<DataType>)},
		result,
		free_when_done	
	);

}

template< class DataType >
py::array_t<std::complex<DataType>,py::array::c_style> 
rFFT_py(py::array_t<DataType,py::array::c_style> py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;

	DataType* ptr_py_in = (DataType*) buf_in.ptr;
	std::complex<DataType>* result = (std::complex<DataType>*) fftw_malloc(
					sizeof(std::complex<DataType>)*(n/2+1));

	rFFT(n, ptr_py_in, result);

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{n/2+1},
		{sizeof(std::complex<DataType>)},
		result,
		free_when_done	
	);

}

template< class DataType ,class DataType2>
py::array_t<std::complex<DataType>,py::array::c_style> 
rFFT_py(py::array_t<DataType,py::array::c_style> py_in,int N)
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
	
	DataType* ptr_py_in = (DataType*) buf_in.ptr;
	std::complex<DataType>* result = (std::complex<DataType>*) fftw_malloc(
					sizeof(std::complex<DataType>)*(N/2+1)*howmany);

	rFFT_Block(n, N, ptr_py_in, result);

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{(N/2+1)*howmany},
		{sizeof(std::complex<DataType>)},
		result,
		free_when_done	
	);
}

template< class DataType >
py::array_t<DataType,py::array::c_style> 
irFFT_py(py::array_t<std::complex<DataType>,py::array::c_style> py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;

	std::complex<DataType>* ptr_py_in = (std::complex<DataType>*) buf_in.ptr;
	DataType* result = (DataType*) fftw_malloc(sizeof(DataType)*2*(n-1));

	irFFT(2*(n-1), ptr_py_in, result);

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<DataType, py::array::c_style> 
	(
		{2*(n-1)}, //Shape
		{sizeof(DataType)}, //Stride
		result, //Pointer
		free_when_done //mem clear
	);

}


