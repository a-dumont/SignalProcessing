template< class Datatype >
np_complex autocorrelation_py(Datatype py_in)
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
	double* ptr_py_in = (double*) buf_in.ptr;
		
	// Creates the output buffer
	dbl_complex* result = (dbl_complex*) fftw_malloc(sizeof(dbl_complex)*(n/2+1));

	import_wisdom(wisdom_path);
	autocorrelation(n, ptr_py_in, result);

	// Wraps the output to pass it to python
	py::capsule free_when_done( result, fftw_free );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{(n/2+1)}, // Size of array
		{sizeof(dbl_complex)}, // Stride 
		result, // C++ pointer
		free_when_done	// Free function
	);	
}

template< class Datatype, class Datatype2>
np_complex autocorrelation_py(Datatype py_in, int N)
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
	double* ptr_py_in = (double*) buf_in.ptr;
		
	// Creates the output buffer
	dbl_complex* result = (dbl_complex*) fftw_malloc(sizeof(dbl_complex)*(N/2+1));

	import_wisdom(wisdom_path);
	autocorrelation_Block(n, N, ptr_py_in, result);
	export_wisdom(wisdom_path);

	// Wraps the output to pass it to python
	py::capsule free_when_done( result, fftw_free );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{(N/2+1)}, // Size of array
		{sizeof(dbl_complex)}, // Stride 
		result, // C++ pointer
		free_when_done	// Free function
	);	
}


template< class Datatype >
np_complex xcorrelation_py(Datatype py_in1, Datatype py_in2)
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
	double* ptr_py_in1 = (double*) buf_in1.ptr;
	double* ptr_py_in2 = (double*) buf_in2.ptr;
		
	// Creates the output buffer
	dbl_complex* result = (dbl_complex*) fftw_malloc(sizeof(dbl_complex)*(n/2+1));

	import_wisdom(wisdom_path);
	xcorrelation(n, ptr_py_in1, ptr_py_in2,result);

	// Wraps the output to pass it to python
	py::capsule free_when_done( result, fftw_free );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{(n/2+1)}, // Size of array
		{sizeof(dbl_complex)}, // Stride 
		result, // C++ pointer
		free_when_done	// Free function
	);	
}

template< class Datatype, class DataTtype2>
np_complex xcorrelation_py(Datatype py_in1, Datatype py_in2, int N)
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
	double* ptr_py_in1 = (double*) buf_in1.ptr;
	double* ptr_py_in2 = (double*) buf_in2.ptr;
		
	// Creates the output buffer
	dbl_complex* result = (dbl_complex*) fftw_malloc(sizeof(dbl_complex)*(N/2+1));

	import_wisdom(wisdom_path);
	xcorrelation_Block(n, N, ptr_py_in1, ptr_py_in2, result);
	export_wisdom(wisdom_path);

	// Wraps the output to pass it to python
	py::capsule free_when_done( result, fftw_free );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{(N/2+1)}, // Size of array
		{sizeof(dbl_complex)}, // Stride 
		result, // C++ pointer
		free_when_done	// Free function
	);	
}

template< class Datatype>
np_complex complete_correlation_py(Datatype py_in1, Datatype py_in2)
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
	double* ptr_py_in1 = (double*) buf_in1.ptr;
	double* ptr_py_in2 = (double*) buf_in2.ptr;
		
	// Creates the output buffer
	dbl_complex* result = (dbl_complex*) fftw_malloc(3*sizeof(dbl_complex)*(n/2+1));

	import_wisdom(wisdom_path);
	complete_correlation(n, ptr_py_in1, ptr_py_in2, result);

	// Wraps the output to pass it to python
	py::capsule free_when_done( result, fftw_free );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{3,(n/2+1)}, // Size of array
		{(n/2+1)*sizeof(dbl_complex),sizeof(dbl_complex)}, // Stride 
		result, // C++ pointer
		free_when_done	// Free function
	);	
}

template< class Datatype, class Datatype2>
np_complex complete_correlation_py(Datatype py_in1, Datatype py_in2, int N)
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
	double* ptr_py_in1 = (double*) buf_in1.ptr;
	double* ptr_py_in2 = (double*) buf_in2.ptr;
		
	// Creates the output buffer
	dbl_complex* result = (dbl_complex*) fftw_malloc(3*sizeof(dbl_complex)*(N/2+1));

	import_wisdom(wisdom_path);
	complete_correlation_Block(n, N, ptr_py_in1, ptr_py_in2, result);
	export_wisdom(wisdom_path);

	// Wraps the output to pass it to python
	py::capsule free_when_done( result, fftw_free );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{3,(N/2+1)}, // Size of array
		{(N/2+1)*sizeof(dbl_complex),sizeof(dbl_complex)}, // Stride 
		result, // C++ pointer
		free_when_done	// Free function
	);	
}
