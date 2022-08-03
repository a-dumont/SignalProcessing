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

	FFT<std::complex<DataType>>(n, ptr_py_in, result);

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

	FFT_Parallel<std::complex<DataType>>(n, ptr_py_in, result, nthreads);

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

	FFT_Block<std::complex<DataType>>(n, N, ptr_py_in, result);

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
				DataType2 N, DataType2 nthreads)
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

	FFT_Block_Parallel<std::complex<DataType>>(n, N, ptr_py_in, result, nthreads);

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

	iFFT<std::complex<DataType>>(n, ptr_py_in, result);

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

	rFFT<DataType>(n, ptr_py_in, result);

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
rFFT_py(py::array_t<DataType,py::array::c_style> py_in, DataType2 N)
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

	rFFT_Block<DataType>(n, N, ptr_py_in, result);

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

	irFFT<DataType>(2*(n-1), ptr_py_in, result);

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<DataType, py::array::c_style> 
	(
		{2*(n-1)}, //Shape
		{sizeof(DataType)}, //Stride
		result, //Pointer
		free_when_done //mem clear
	);

}

template< class DataType >
py::array_t<dbl_complex,py::array::c_style> 
digitizer_FFT_py(py::array_t<DataType,py::array::c_style> py_in, double conv)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	uint64_t n = buf_in.size;
	uint64_t offset = 1<<(sizeof(DataType)*8-1);
	
	DataType* ptr_py_in = (DataType*) buf_in.ptr;
	dbl_complex* result = (dbl_complex*) fftw_malloc(sizeof(dbl_complex)*n);
	
	for(uint64_t i=0;i<=n;i++){result[i]=(ptr_py_in[i]-offset)*conv;}
	

	FFT_Parallel<dbl_complex>(n, result, result, omp_get_max_threads());

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{n},
		{sizeof(dbl_complex)},
		result,
		free_when_done	
	);
}
