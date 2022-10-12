template<class DataType>
py::array_t<DataType,py::array::c_style> 
vector_sum_py(
py::array_t<DataType,py::array::c_style> py_in1,
py::array_t<DataType,py::array::c_style> py_in2)
{
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	if (buf_in1.ndim != 1 || buf_in2.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = std::min(buf_in1.size,buf_in2.size);
	
	DataType* in1 = (DataType*) buf_in1.ptr;
	DataType* in2 = (DataType*) buf_in2.ptr;
	
	DataType* out = (DataType*) malloc(n*sizeof(DataType));

	DataType* gpu;
	cudaMalloc((void**)&gpu, 2*n*sizeof(DataType));
	
	cudaMemcpy(gpu,ptr_py_in1,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	cudaMemcpy(gpu+n,ptr_py_in2,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	
	vector_sum<DataType>(n,gpu,gpu+n);
	
	cudaMemcpy(out,gpu,sizeof(DataType)*n,cudaMemcpyDeviceToHost);	
	cudaFree(gpu);

	py::capsule free_when_done(out, free);
	return py::array_t<DataType, py::array::c_style> 
	(
		{n},
		{sizeof(DataType)},
		out,
		free_when_done	
	);
}