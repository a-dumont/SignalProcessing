template<class DataType>
py::array_t<DataType,py::array::c_style> 
autocorrelation_cuda_py(py::array_t<DataType,py::array::c_style> py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = buf_in.size;
	
	DataType* ptr_py_in = (DataType*) buf_in.ptr;
	
	DataType* out = (DataType*) malloc((n/2+1)*sizeof(DataType));

	std::complex<DataType>* gpu;
	cudaMalloc((void**)&gpu, (n/2+1)*sizeof(std::complex<DataType>));
	
	cudaMemcpy(gpu,ptr_py_in,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	rFFT_CUDA<std::complex<DataType>>(n,gpu);
	autocorrelation_cuda<DataType>(n/2+1,gpu,reinterpret_cast<DataType*>(gpu));
	cudaMemcpy(out,gpu,sizeof(DataType)*(n/2+1),cudaMemcpyDeviceToHost);	

	cudaFree(gpu);

	py::capsule free_when_done( out, free );
	return py::array_t<DataType, py::array::c_style> 
	(
		{n/2+1},
		{sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
cross_correlation_cuda_py(py::array_t<DataType,py::array::c_style> py_in1,
				py::array_t<DataType,py::array::c_style> py_in2)
{
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	if (buf_in1.ndim != 1 && buf_in2.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	long long int n = std::min(buf_in1.size,buf_in2.size);
	
	DataType* ptr_py_in1 = (DataType*) buf_in1.ptr;
	DataType* ptr_py_in2 = (DataType*) buf_in2.ptr;
	
	std::complex<DataType>* out = (std::complex<DataType>*) malloc(2*(n/2+1)*sizeof(DataType));

	std::complex<DataType>* gpu1;
	cudaMalloc((void**)&gpu1, (n/2+1)*sizeof(std::complex<DataType>));
	cudaMemcpy(gpu1,ptr_py_in1,sizeof(DataType)*n,cudaMemcpyHostToDevice);

	std::complex<DataType>* gpu2;
	cudaMalloc((void**)&gpu2, (n/2+1)*sizeof(std::complex<DataType>));
	cudaMemcpy(gpu2,ptr_py_in2,sizeof(DataType)*n,cudaMemcpyHostToDevice);

	rFFT_CUDA<std::complex<DataType>>(n,gpu1);
	rFFT_CUDA<std::complex<DataType>>(n,gpu2);
	
	cross_correlation_cuda<DataType>(n/2+1,gpu1,gpu2,gpu1);
	cudaMemcpy(out,gpu1,sizeof(std::complex<DataType>)*(n/2+1),cudaMemcpyDeviceToHost);	

	cudaFree(gpu1);
	cudaFree(gpu2);

	py::capsule free_when_done( out, free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{n/2+1},
		{sizeof(std::complex<DataType>)},
		out,
		free_when_done	
	);
}

template<class DataType>
std::tuple<py::array_t<DataType,py::array::c_style>, 
				py::array_t<DataType,py::array::c_style>,
				py::array_t<std::complex<DataType>,py::array::c_style>>
complete_correlation_cuda_py(py::array_t<DataType,py::array::c_style> py_in1,
				py::array_t<DataType,py::array::c_style> py_in2)
{
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	if (buf_in1.ndim != 1 && buf_in2.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	long long int n = std::min(buf_in1.size,buf_in2.size);
	
	DataType* ptr_py_in1 = (DataType*) buf_in1.ptr;
	DataType* ptr_py_in2 = (DataType*) buf_in2.ptr;
	
	DataType* out1 = (DataType*) malloc((n/2+1)*sizeof(DataType));
	DataType* out2 = (DataType*) malloc((n/2+1)*sizeof(DataType));
	std::complex<DataType>* out3 = (std::complex<DataType>*) malloc(2*(n/2+1)*sizeof(DataType));

	std::complex<DataType>* gpu1;
	cudaMalloc((void**)&gpu1, (n/2+1)*sizeof(std::complex<DataType>));
	cudaMemcpy(gpu1,ptr_py_in1,sizeof(DataType)*n,cudaMemcpyHostToDevice);

	std::complex<DataType>* gpu2;
	cudaMalloc((void**)&gpu2, (n/2+1)*sizeof(std::complex<DataType>));
	cudaMemcpy(gpu2,ptr_py_in2,sizeof(DataType)*n,cudaMemcpyHostToDevice);

	rFFT_CUDA<std::complex<DataType>>(n,gpu1);
	rFFT_CUDA<std::complex<DataType>>(n,gpu2);

	complete_correlation_cuda<DataType>(n/2+1,gpu1,gpu2,
					reinterpret_cast<DataType*>(gpu2),
					reinterpret_cast<DataType*>(gpu2)+(n/2+1),gpu1);

	cudaMemcpy(out1,gpu2,sizeof(DataType)*(n/2+1),cudaMemcpyDeviceToHost);
	cudaMemcpy(out2,reinterpret_cast<DataType*>(gpu2)+(n/2+1),
					sizeof(DataType)*(n/2+1),cudaMemcpyDeviceToHost);
	cudaMemcpy(out3,gpu1,sizeof(std::complex<DataType>)*(n/2+1),cudaMemcpyDeviceToHost);

	cudaFree(gpu1);
	cudaFree(gpu2);

	py::capsule free_when_done1( out1, free );
	py::capsule free_when_done2( out2, free );
	py::capsule free_when_done3( out3, free );
	
	return std::make_tuple(
		py::array_t<DataType, py::array::c_style> 
		({n/2+1},{sizeof(DataType)},out1,free_when_done1),
		py::array_t<DataType, py::array::c_style> 
		({n/2+1},{sizeof(DataType)},out2,free_when_done2),
		py::array_t<std::complex<DataType>, py::array::c_style> 
		({n/2+1},{sizeof(std::complex<DataType>)},out3,free_when_done3));
}
