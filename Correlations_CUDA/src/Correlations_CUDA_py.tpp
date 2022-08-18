template< class DataType >
py::array_t<DataType,py::array::c_style> 
autocorrelation_cuda_py(py::array_t<DataType,py::array::c_style> py_in)
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

	// Bunch of malloc
	std::complex<DataType>* fft_result;
	cudaMallocManaged((void**)&fft_result,(n/2+1)*sizeof(std::complex<DataType>));
	DataType* out;
	out = (DataType*) malloc((n/2+1)*sizeof(DataType));

	// fft
	cudaMemcpy(fft_result,ptr_py_in,n*sizeof(DataType),cudaMemcpyHostToDevice);
	rFFT_CUDA(n,fft_result);

	// Blocks and threads
	int threads = 512;
	int blocks = (n/2+1)/512+1;

	// Autocorrelation
	autocorrelation_cuda((n/2+1),
					(cuDoubleComplex*)fft_result,(double*)fft_result,blocks,threads);

	// output copy
	cudaMemcpy(out,fft_result,(n/2+1)*sizeof(DataType),cudaMemcpyDeviceToHost);
	cudaFree(fft_result);
	
	// Wraps the output to pass it to python
	py::capsule free_when_done( out, free );
	return py::array_t<DataType, py::array::c_style> 
	(
		{(n/2+1)}, // Size of array
		{sizeof(DataType)}, // Stride 
		out, // C++ pointer
		free_when_done	// Free function
	);	
}

template< class DataType >
py::array_t<std::complex<DataType>,py::array::c_style> 
xcorrelation_cuda_py(py::array_t<DataType,py::array::c_style> py_in,
				py::array_t<DataType,py::array::c_style> py_in2)
{
	// Requests the buffer from pyhton
	py::buffer_info buf_in = py_in.request();
	py::buffer_info buf_in2 = py_in2.request();

	// Checks dimension
	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	// Defines the length
	int n = buf_in.size;
	
	// Get the pointer from python
	DataType* ptr_py_in = (DataType*) buf_in.ptr;
	DataType* ptr_py_in2 = (DataType*) buf_in2.ptr;

	// Bunch of malloc
	std::complex<DataType>* fft_result;
	std::complex<DataType>* fft_result2;
	cudaMallocManaged((void**)&fft_result,(n/2+1)*sizeof(std::complex<DataType>));
	cudaMallocManaged((void**)&fft_result2,(n/2+1)*sizeof(std::complex<DataType>));
	std::complex<DataType>* out = (std::complex<DataType>*) malloc(
					(n/2+1)*sizeof(std::complex<DataType>));

	// fft
	cudaMemcpy(fft_result,ptr_py_in,n*sizeof(DataType),cudaMemcpyHostToDevice);
	cudaMemcpy(fft_result2,ptr_py_in2,n*sizeof(DataType),cudaMemcpyHostToDevice);
	rFFT_CUDA(n,fft_result);
	rFFT_CUDA(n,fft_result2);

	// Blocks and threads
	int threads = 512;
	int blocks = (n/2+1)/512+1;

	// Autocorrelation
	xcorrelation_cuda((n/2+1),
					(cuDoubleComplex*)fft_result,(cuDoubleComplex*)fft_result2,blocks,threads);

	// output copy
	cudaMemcpy(out,fft_result,(n/2+1)*sizeof(std::complex<DataType>),cudaMemcpyDeviceToHost);
	cudaFree(fft_result);
	cudaFree(fft_result2);
	
	// Wraps the output to pass it to python
	py::capsule free_when_done( out, free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{(n/2+1)}, // Size of array
		{sizeof(std::complex<DataType>)}, // Stride 
		out, // C++ pointer
		free_when_done	// Free function
	);	
}

template< class DataType >
py::array_t<std::complex<DataType>,py::array::c_style> 
xcorrelation_block_cuda_py(py::array_t<DataType,py::array::c_style> py_in,
				py::array_t<DataType,py::array::c_style> py_in2, int size)
{
	// Requests the buffer from pyhton
	py::buffer_info buf_in = py_in.request();
	py::buffer_info buf_in2 = py_in2.request();

	// Checks dimension
	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	// Defines the length
	int n = buf_in.size;
	int howmany = n/size;
	int N = 1<<(int)std::ceil(std::log2(1.0*howmany));
		
	// Get the pointer from python
	DataType* ptr_py_in = (DataType*) buf_in.ptr;
	DataType* ptr_py_in2 = (DataType*) buf_in2.ptr;

	// Bunch of malloc
	std::complex<DataType>* fft_result;
	std::complex<DataType>* fft_result2;
	cudaMallocManaged((void**)&fft_result,N*(size/2+1)*sizeof(std::complex<DataType>));
	cudaMallocManaged((void**)&fft_result2,N*(size/2+1)*sizeof(std::complex<DataType>));
	std::complex<DataType>* out = (std::complex<DataType>*) malloc(
					(size/2+1)*sizeof(std::complex<DataType>));
	// fft
	cudaMemcpy2D(fft_result,
				(size+2)*sizeof(DataType),
				ptr_py_in,
				size*sizeof(DataType),
				size*sizeof(DataType),
				howmany,
				cudaMemcpyHostToDevice);
	cudaMemcpy2D(fft_result2,
				(size+2)*sizeof(DataType),
				ptr_py_in2,
				size*sizeof(DataType),
				size*sizeof(DataType),
				howmany,
				cudaMemcpyHostToDevice);

	rFFT_Block_CUDA(size*howmany,size,fft_result);
	rFFT_Block_CUDA(size*howmany,size,fft_result2);

	// Blocks and threads
	int threads = 512;
	int blocks = howmany*(size/2+1)/512+1;

	// Autocorrelation
	xcorrelation_cuda(howmany*(size/2+1),(cuDoubleComplex*)fft_result,
					(cuDoubleComplex*)fft_result2,blocks,threads);

	// Reduction
	blocks = N*(size/2+1)/512/2+1;
	reduction_complex_cuda(N,howmany,(cuDoubleComplex*)fft_result,
					(cuDoubleComplex*)fft_result,size,blocks,threads);

	// output copy
	cudaMemcpy(out,fft_result,(size/2+1)*sizeof(std::complex<DataType>),cudaMemcpyDeviceToHost);
	cudaFree(fft_result);
	cudaFree(fft_result2);

	// Wraps the output to pass it to python
	py::capsule free_when_done( out, free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{(size/2+1)}, // Size of array
		{sizeof(std::complex<DataType>)}, // Stride 
		out, // C++ pointer
		free_when_done	// Free function
	);	
}
