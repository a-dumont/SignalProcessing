template< class DataType >
py::array_t<std::complex<DataType>,py::array::c_style> 
autocorrelation_py(py::array_t<DataType,py::array::c_style> py_in)
{
	// Requests the buffer from pyhton
	py::buffer_info buf_in = py_in.request();

	// Checks dimension
	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	// Defines the length
	int n = buf_in.size;
	
	// Get the pointer from python
	DataType* ptr_py_in = (DataType*) buf_in.ptr;
		
	// Creates the output buffer
	std::complex<DataType>* result = (std::complex<DataType>*) fftw_malloc(
					sizeof(std::complex<DataType>)*(n/2+1));

	autocorrelation(n, ptr_py_in, result);

	// Wraps the output to pass it to python
	py::capsule free_when_done( result, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{(n/2+1)}, // Size of array
		{sizeof(std::complex<DataType>)}, // Stride 
		result, // C++ pointer
		free_when_done	// Free function
	);	
}

template< class DataType, class DataType2>
py::array_t<std::complex<DataType>,py::array::c_style> 
autocorrelation_py(py::array_t<DataType,py::array::c_style> py_in, DataType2 N)
{
	// Requests the buffer from pyhton
	py::buffer_info buf_in = py_in.request();

	// Checks dimension
	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	// Defines the length
	int n = buf_in.size;
	
	// Get the pointer from python
	DataType* ptr_py_in = (DataType*) buf_in.ptr;
		
	// Creates the output buffer
	std::complex<DataType>* result = (std::complex<DataType>*) fftw_malloc(
					sizeof(std::complex<DataType>)*(N/2+1));

	autocorrelation_Block(n, N, ptr_py_in, result);

	// Wraps the output to pass it to python
	py::capsule free_when_done( result, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{(N/2+1)}, // Size of array
		{sizeof(std::complex<DataType>)}, // Stride 
		result, // C++ pointer
		free_when_done	// Free function
	);	
}


template< class DataType >
py::array_t<std::complex<DataType>,py::array::c_style> 
xcorrelation_py(py::array_t<DataType,py::array::c_style> py_in1,
				py::array_t<DataType,py::array::c_style> py_in2)
{
	// Requests the buffer from pyhton
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	// Checks dimension
	if ( (buf_in1.ndim != 1) | (buf_in2.ndim != 1) )
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	// Checks dimension
	if (buf_in1.size != buf_in2.size)
	{
		throw std::runtime_error("U dumbdumb size must be same for both.");
	}	

	// Defines the length
	int n = buf_in1.size;
	
	// Get the pointers from python
	DataType* ptr_py_in1 = (DataType*) buf_in1.ptr;
	DataType* ptr_py_in2 = (DataType*) buf_in2.ptr;
		
	// Creates the output buffer
	std::complex<DataType>* result = (std::complex<DataType>*) fftw_malloc(
					sizeof(std::complex<DataType>)*(n/2+1));

	xcorrelation(n, ptr_py_in1, ptr_py_in2,result);

	// Wraps the output to pass it to python
	py::capsule free_when_done( result, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{(n/2+1)}, // Size of array
		{sizeof(std::complex<DataType>)}, // Stride 
		result, // C++ pointer
		free_when_done	// Free function
	);	
}

template< class DataType, class DataType2>
py::array_t<std::complex<DataType>,py::array::c_style> 
xcorrelation_py(py::array_t<DataType,py::array::c_style> py_in1, 
				py::array_t<DataType,py::array::c_style> py_in2, DataType2 N)
{
	// Requests the buffer from pyhton
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	// Checks dimension
	if ( (buf_in1.ndim != 1) | (buf_in2.ndim != 1) )
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	// Checks dimension
	if (buf_in1.size != buf_in2.size)
	{
		throw std::runtime_error("U dumbdumb size must be same for both.");
	}	

	// Defines the length
	int n = buf_in1.size;
	
	// Get the pointers from python
	DataType* ptr_py_in1 = (DataType*) buf_in1.ptr;
	DataType* ptr_py_in2 = (DataType*) buf_in2.ptr;
		
	// Creates the output buffer
	std::complex<DataType>* result = (std::complex<DataType>*) fftw_malloc(
					sizeof(std::complex<DataType>)*(N/2+1));

	xcorrelation_Block(n, N, ptr_py_in1, ptr_py_in2, result);

	// Wraps the output to pass it to python
	py::capsule free_when_done( result, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{(N/2+1)}, // Size of array
		{sizeof(std::complex<DataType>)}, // Stride 
		result, // C++ pointer
		free_when_done	// Free function
	);	
}

template< class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
complete_correlation_py(py::array_t<DataType,py::array::c_style> py_in1, 
				py::array_t<DataType,py::array::c_style> py_in2)
{
	// Requests the buffer from pyhton
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	// Checks dimension
	if ( (buf_in1.ndim != 1) | (buf_in2.ndim != 1) )
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	// Checks dimension
	if (buf_in1.size != buf_in2.size)
	{
		throw std::runtime_error("U dumbdumb size must be same for both.");
	}	

	// Defines the length
	int n = buf_in1.size;
	
	// Get the pointers from python
	DataType* ptr_py_in1 = (DataType*) buf_in1.ptr;
	DataType* ptr_py_in2 = (DataType*) buf_in2.ptr;
		
	// Creates the output buffer
	std::complex<DataType>* result = (std::complex<DataType>*) fftw_malloc(
					3*sizeof(std::complex<DataType>)*(n/2+1));

	complete_correlation(n, ptr_py_in1, ptr_py_in2, result);

	// Wraps the output to pass it to python
	py::capsule free_when_done( result, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{3,(n/2+1)}, // Size of array
		{(n/2+1)*sizeof(std::complex<DataType>),sizeof(std::complex<DataType>)}, // Stride 
		result, // C++ pointer
		free_when_done	// Free function
	);	
}

template< class DataType, class DataType2>
py::array_t<std::complex<DataType>,py::array::c_style> 
complete_correlation_py(py::array_t<DataType,py::array::c_style> py_in1, 
				py::array_t<DataType,py::array::c_style> py_in2, DataType2 N)
{
	// Requests the buffer from pyhton
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	// Checks dimension
	if ( (buf_in1.ndim != 1) | (buf_in2.ndim != 1) )
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	// Checks dimension
	if (buf_in1.size != buf_in2.size)
	{
		throw std::runtime_error("U dumbdumb size must be same for both.");
	}	

	// Defines the length
	int n = buf_in1.size;
	
	// Get the pointers from python
	DataType* ptr_py_in1 = (DataType*) buf_in1.ptr;
	DataType* ptr_py_in2 = (DataType*) buf_in2.ptr;
		
	// Creates the output buffer
	std::complex<DataType>* result = (std::complex<DataType>*) fftw_malloc(
					3*sizeof(std::complex<DataType>)*(N/2+1));

	complete_correlation_Block(n, N, ptr_py_in1, ptr_py_in2, result);

	// Wraps the output to pass it to python
	py::capsule free_when_done( result, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{3,(N/2+1)}, // Size of array
		{(N/2+1)*sizeof(std::complex<DataType>),sizeof(std::complex<DataType>)}, // Stride 
		result, // C++ pointer
		free_when_done	// Free function
	);	
}
