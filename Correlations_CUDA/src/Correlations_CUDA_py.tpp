template< class Datatype >
np_double autocorrelation_cuda_py(Datatype py_in)
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

	// Bunch of malloc
	dbl_complex* fft_result;
	cudaMallocManaged((void**)&fft_result,(n/2+1)*sizeof(dbl_complex));
	double* out = (double*) malloc((n/2+1)*sizeof(double));

	// fft
	cudaMemcpy(fft_result,ptr_py_in,n*sizeof(double),cudaMemcpyHostToDevice);
	rFFT_CUDA(n,fft_result,fft_result);

	// Blocks and threads
	int threads = 512;
	int blocks = (n/2+1)/512+1;

	// Autocorrelation
	autocorrelation_cuda((n/2+1),(cuDoubleComplex*)fft_result,(double*)fft_result,blocks,threads);

	// output copy
	cudaMemcpy(out,fft_result,(n/2+1)*sizeof(double),cudaMemcpyDeviceToHost);
	cudaFree(fft_result);
	
	// Wraps the output to pass it to python
	py::capsule free_when_done( out, free );
	return py::array_t<double, py::array::c_style> 
	(
		{(n/2+1)}, // Size of array
		{sizeof(double)}, // Stride 
		out, // C++ pointer
		free_when_done	// Free function
	);	
}

template< class Datatype >
np_complex xcorrelation_cuda_py(Datatype py_in, Datatype py_in2)
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
	double* ptr_py_in = (double*) buf_in.ptr;
	double* ptr_py_in2 = (double*) buf_in2.ptr;

	// Bunch of malloc
	dbl_complex* fft_result;
	dbl_complex* fft_result2;
	cudaMallocManaged((void**)&fft_result,(n/2+1)*sizeof(dbl_complex));
	cudaMallocManaged((void**)&fft_result2,(n/2+1)*sizeof(dbl_complex));
	dbl_complex* out = (dbl_complex*) malloc((n/2+1)*sizeof(dbl_complex));

	// fft
	cudaMemcpy(fft_result,ptr_py_in,n*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(fft_result2,ptr_py_in2,n*sizeof(double),cudaMemcpyHostToDevice);
	rFFT_CUDA(n,fft_result,fft_result);
	rFFT_CUDA(n,fft_result2,fft_result2);

	// Blocks and threads
	int threads = 512;
	int blocks = (n/2+1)/512+1;

	// Autocorrelation
	xcorrelation_cuda((n/2+1),(cuDoubleComplex*)fft_result,(cuDoubleComplex*)fft_result2,blocks,threads);

	// output copy
	cudaMemcpy(out,fft_result,(n/2+1)*sizeof(dbl_complex),cudaMemcpyDeviceToHost);
	cudaFree(fft_result);
	cudaFree(fft_result2);
	
	// Wraps the output to pass it to python
	py::capsule free_when_done( out, free );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{(n/2+1)}, // Size of array
		{sizeof(dbl_complex)}, // Stride 
		out, // C++ pointer
		free_when_done	// Free function
	);	
}

template< class Datatype >
np_complex xcorrelation_block_cuda_py(Datatype py_in, Datatype py_in2, int size)
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
	double* ptr_py_in = (double*) buf_in.ptr;
	double* ptr_py_in2 = (double*) buf_in2.ptr;

	// Bunch of malloc
	dbl_complex* fft_result;
	dbl_complex* fft_result2;
	cudaMallocManaged((void**)&fft_result,N*(size/2+1)*sizeof(dbl_complex));
	cudaMallocManaged((void**)&fft_result2,N*(size/2+1)*sizeof(dbl_complex));
	dbl_complex* out = (dbl_complex*) malloc((size/2+1)*sizeof(dbl_complex));
	// fft
	cudaMemcpy2D(fft_result,
				(size+2)*sizeof(double),
				ptr_py_in,
				size*sizeof(double),
				size*sizeof(double),
				howmany,
				cudaMemcpyHostToDevice);
	cudaMemcpy2D(fft_result2,
				(size+2)*sizeof(double),
				ptr_py_in2,
				size*sizeof(double),
				size*sizeof(double),
				howmany,
				cudaMemcpyHostToDevice);

	rFFT_Block_CUDA(size*howmany,size,fft_result,fft_result);
	rFFT_Block_CUDA(size*howmany,size,fft_result2,fft_result2);

	// Blocks and threads
	int threads = 512;
	int blocks = howmany*(size/2+1)/512+1;

	// Autocorrelation
	xcorrelation_cuda(howmany*(size/2+1),(cuDoubleComplex*)fft_result,(cuDoubleComplex*)fft_result2,blocks,threads);

	// Reduction
	blocks = N*(size/2+1)/512/2+1;
	reduction_complex_cuda(N,howmany,(cuDoubleComplex*)fft_result,(cuDoubleComplex*)fft_result,size,blocks,threads);

	// output copy
	cudaMemcpy(out,fft_result,(size/2+1)*sizeof(dbl_complex),cudaMemcpyDeviceToHost);
	cudaFree(fft_result);
	cudaFree(fft_result2);

	// Wraps the output to pass it to python
	py::capsule free_when_done( out, free );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{(size/2+1)}, // Size of array
		{sizeof(dbl_complex)}, // Stride 
		out, // C++ pointer
		free_when_done	// Free function
	);	
}
