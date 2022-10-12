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

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
vector_sum_complex_py(
py::array_t<std::complex<DataType>,py::array::c_style> py_in1,
py::array_t<std::complex<DataType>,py::array::c_style> py_in2)
{
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	if (buf_in1.ndim != 1 || buf_in2.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = 2*std::min(buf_in1.size,buf_in2.size);
	
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
		{n/2},
		{2*sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<DataType,py::array::c_style> 
vector_product_py(
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
	
	vector_product<DataType>(n,gpu,gpu+n);
	
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

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
vector_product_complex_py(
py::array_t<std::complex<DataType>,py::array::c_style> py_in1,
py::array_t<std::complex<DataType>,py::array::c_style> py_in2)
{
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	if (buf_in1.ndim != 1 || buf_in2.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = 2*std::min(buf_in1.size,buf_in2.size);
	
	DataType* in1 = (DataType*) buf_in1.ptr;
	DataType* in2 = (DataType*) buf_in2.ptr;
	
	DataType* out = (DataType*) malloc(n*sizeof(DataType));

	DataType* gpu;
	cudaMalloc((void**)&gpu, 2*n*sizeof(DataType));
	
	cudaMemcpy(gpu,ptr_py_in1,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	cudaMemcpy(gpu+n,ptr_py_in2,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	
	vector_product<DataType>(n,gpu,gpu+n);
	
	cudaMemcpy(out,gpu,sizeof(DataType)*n,cudaMemcpyDeviceToHost);	
	cudaFree(gpu);

	py::capsule free_when_done(out, free);
	return py::array_t<DataType, py::array::c_style> 
	(
		{n/2},
		{2*sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<DataType,py::array::c_style> 
vector_diff_py(
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
	
	vector_diff<DataType>(n,gpu,gpu+n);
	
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

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
vector_diff_complex_py(
py::array_t<std::complex<DataType>,py::array::c_style> py_in1,
py::array_t<std::complex<DataType>,py::array::c_style> py_in2)
{
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	if (buf_in1.ndim != 1 || buf_in2.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = 2*std::min(buf_in1.size,buf_in2.size);
	
	DataType* in1 = (DataType*) buf_in1.ptr;
	DataType* in2 = (DataType*) buf_in2.ptr;
	
	DataType* out = (DataType*) malloc(n*sizeof(DataType));

	DataType* gpu;
	cudaMalloc((void**)&gpu, 2*n*sizeof(DataType));
	
	cudaMemcpy(gpu,ptr_py_in1,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	cudaMemcpy(gpu+n,ptr_py_in2,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	
	vector_diff<DataType>(n,gpu,gpu+n);
	
	cudaMemcpy(out,gpu,sizeof(DataType)*n,cudaMemcpyDeviceToHost);	
	cudaFree(gpu);

	py::capsule free_when_done(out, free);
	return py::array_t<DataType, py::array::c_style> 
	(
		{n/2},
		{2*sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<DataType,py::array::c_style> 
vector_div_py(
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
	
	vector_div<DataType>(n,gpu,gpu+n);
	
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

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
vector_div_complex_py(
py::array_t<std::complex<DataType>,py::array::c_style> py_in1,
py::array_t<std::complex<DataType>,py::array::c_style> py_in2)
{
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	if (buf_in1.ndim != 1 || buf_in2.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = 2*std::min(buf_in1.size,buf_in2.size);
	
	DataType* in1 = (DataType*) buf_in1.ptr;
	DataType* in2 = (DataType*) buf_in2.ptr;
	
	DataType* out = (DataType*) malloc(n*sizeof(DataType));

	DataType* gpu;
	cudaMalloc((void**)&gpu, 2*n*sizeof(DataType));
	
	cudaMemcpy(gpu,ptr_py_in1,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	cudaMemcpy(gpu+n,ptr_py_in2,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	
	vector_div<DataType>(n,gpu,gpu+n);
	
	cudaMemcpy(out,gpu,sizeof(DataType)*n,cudaMemcpyDeviceToHost);	
	cudaFree(gpu);

	py::capsule free_when_done(out, free);
	return py::array_t<DataType, py::array::c_style> 
	(
		{n/2},
		{2*sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<DataType,py::array::c_style> 
matrix_sum_py(
py::array_t<DataType,py::array::c_style> py_in1,
py::array_t<DataType,py::array::c_style> py_in2)
{
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	if (buf_in1.ndim != 2 || buf_in2.ndim != 2)
	{
		throw std::runtime_error("U dumbdumb dimension must be 2.");
	}	
	if (buf_in1.shape(0) != buf_in2.shape(0) || buf_in1.shape(1) != buf_in2.shape(1))
	{
		throw std::runtime_error("U dumbdumb shapes must be same.");
	}	

	long long int nr = buf_in1.shape(0);
	long long int nc = buf_in1.shape(1);
	long long int n = nr*nc;
	
	DataType* in1 = (DataType*) buf_in1.ptr;
	DataType* in2 = (DataType*) buf_in2.ptr;
	
	DataType* out = (DataType*) malloc(n*sizeof(DataType));

	DataType* gpu;
	cudaMalloc((void**)&gpu, 2*n*sizeof(DataType));
	
	cudaMemcpy(gpu,ptr_py_in1,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	cudaMemcpy(gpu+n,ptr_py_in2,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	
	matrix_sum<DataType>(nr,nc,gpu,gpu+n);
	
	cudaMemcpy(out,gpu,sizeof(DataType)*n,cudaMemcpyDeviceToHost);	
	cudaFree(gpu);

	py::capsule free_when_done(out, free);
	return py::array_t<DataType, py::array::c_style> 
	(
		{nr,nc},
		{nc*sizeof(DataType),sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
matrix_sum_complex_py(
py::array_t<std::complex<DataType>,py::array::c_style> py_in1,
py::array_t<std::complex<DataType>,py::array::c_style> py_in2)
{
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	if (buf_in1.ndim != 2 || buf_in2.ndim != 2)
	{
		throw std::runtime_error("U dumbdumb dimension must be 2.");
	}	
	if (buf_in1.shape(0) != buf_in2.shape(0) || buf_in1.shape(1) != buf_in2.shape(1))
	{
		throw std::runtime_error("U dumbdumb shapes must be same.");
	}	

	long long int nr = 2*buf_in1.shape(0);
	long long int nc = 2*buf_in1.shape(1);
	long long int n = nr*nc;
	
	DataType* in1 = (DataType*) buf_in1.ptr;
	DataType* in2 = (DataType*) buf_in2.ptr;
	
	DataType* out = (DataType*) malloc(n*sizeof(DataType));

	DataType* gpu;
	cudaMalloc((void**)&gpu, 2*n*sizeof(DataType));
	
	cudaMemcpy(gpu,ptr_py_in1,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	cudaMemcpy(gpu+n,ptr_py_in2,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	
	matrix_sum<DataType>(nr,nc,gpu,gpu+n);
	
	cudaMemcpy(out,gpu,sizeof(DataType)*n,cudaMemcpyDeviceToHost);	
	cudaFree(gpu);

	py::capsule free_when_done(out, free);
	return py::array_t<DataType, py::array::c_style> 
	(
		{nr/2,nc/2},
		{nc*sizeof(DataType),2*sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<DataType,py::array::c_style> 
matrix_prod_py(
py::array_t<DataType,py::array::c_style> py_in1,
py::array_t<DataType,py::array::c_style> py_in2)
{
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	if (buf_in1.ndim != 2 || buf_in2.ndim != 2)
	{
		throw std::runtime_error("U dumbdumb dimension must be 2.");
	}	
	if (buf_in1.shape(0) != buf_in2.shape(0) || buf_in1.shape(1) != buf_in2.shape(1))
	{
		throw std::runtime_error("U dumbdumb shapes must be same.");
	}	

	long long int nr = buf_in1.shape(0);
	long long int nc = buf_in1.shape(1);
	long long int n = nr*nc;
	
	DataType* in1 = (DataType*) buf_in1.ptr;
	DataType* in2 = (DataType*) buf_in2.ptr;
	
	DataType* out = (DataType*) malloc(n*sizeof(DataType));

	DataType* gpu;
	cudaMalloc((void**)&gpu, 2*n*sizeof(DataType));
	
	cudaMemcpy(gpu,ptr_py_in1,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	cudaMemcpy(gpu+n,ptr_py_in2,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	
	matrix_prod<DataType>(nr,nc,gpu,gpu+n);
	
	cudaMemcpy(out,gpu,sizeof(DataType)*n,cudaMemcpyDeviceToHost);	
	cudaFree(gpu);

	py::capsule free_when_done(out, free);
	return py::array_t<DataType, py::array::c_style> 
	(
		{nr,nc},
		{nc*sizeof(DataType),sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
matrix_prod_complex_py(
py::array_t<std::complex<DataType>,py::array::c_style> py_in1,
py::array_t<std::complex<DataType>,py::array::c_style> py_in2)
{
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	if (buf_in1.ndim != 2 || buf_in2.ndim != 2)
	{
		throw std::runtime_error("U dumbdumb dimension must be 2.");
	}	
	if (buf_in1.shape(0) != buf_in2.shape(0) || buf_in1.shape(1) != buf_in2.shape(1))
	{
		throw std::runtime_error("U dumbdumb shapes must be same.");
	}	

	long long int nr = 2*buf_in1.shape(0);
	long long int nc = 2*buf_in1.shape(1);
	long long int n = nr*nc;
	
	DataType* in1 = (DataType*) buf_in1.ptr;
	DataType* in2 = (DataType*) buf_in2.ptr;
	
	DataType* out = (DataType*) malloc(n*sizeof(DataType));

	DataType* gpu;
	cudaMalloc((void**)&gpu, 2*n*sizeof(DataType));
	
	cudaMemcpy(gpu,ptr_py_in1,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	cudaMemcpy(gpu+n,ptr_py_in2,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	
	matrix_prod<DataType>(nr,nc,gpu,gpu+n);
	
	cudaMemcpy(out,gpu,sizeof(DataType)*n,cudaMemcpyDeviceToHost);	
	cudaFree(gpu);

	py::capsule free_when_done(out, free);
	return py::array_t<DataType, py::array::c_style> 
	(
		{nr/2,nc/2},
		{nc*sizeof(DataType),2*sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<DataType,py::array::c_style> 
matrix_diff_py(
py::array_t<DataType,py::array::c_style> py_in1,
py::array_t<DataType,py::array::c_style> py_in2)
{
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	if (buf_in1.ndim != 2 || buf_in2.ndim != 2)
	{
		throw std::runtime_error("U dumbdumb dimension must be 2.");
	}	
	if (buf_in1.shape(0) != buf_in2.shape(0) || buf_in1.shape(1) != buf_in2.shape(1))
	{
		throw std::runtime_error("U dumbdumb shapes must be same.");
	}	

	long long int nr = buf_in1.shape(0);
	long long int nc = buf_in1.shape(1);
	long long int n = nr*nc;
	
	DataType* in1 = (DataType*) buf_in1.ptr;
	DataType* in2 = (DataType*) buf_in2.ptr;
	
	DataType* out = (DataType*) malloc(n*sizeof(DataType));

	DataType* gpu;
	cudaMalloc((void**)&gpu, 2*n*sizeof(DataType));
	
	cudaMemcpy(gpu,ptr_py_in1,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	cudaMemcpy(gpu+n,ptr_py_in2,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	
	matrix_diff<DataType>(nr,nc,gpu,gpu+n);
	
	cudaMemcpy(out,gpu,sizeof(DataType)*n,cudaMemcpyDeviceToHost);	
	cudaFree(gpu);

	py::capsule free_when_done(out, free);
	return py::array_t<DataType, py::array::c_style> 
	(
		{nr,nc},
		{nc*sizeof(DataType),sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
matrix_diff_complex_py(
py::array_t<std::complex<DataType>,py::array::c_style> py_in1,
py::array_t<std::complex<DataType>,py::array::c_style> py_in2)
{
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	if (buf_in1.ndim != 2 || buf_in2.ndim != 2)
	{
		throw std::runtime_error("U dumbdumb dimension must be 2.");
	}	
	if (buf_in1.shape(0) != buf_in2.shape(0) || buf_in1.shape(1) != buf_in2.shape(1))
	{
		throw std::runtime_error("U dumbdumb shapes must be same.");
	}	

	long long int nr = 2*buf_in1.shape(0);
	long long int nc = 2*buf_in1.shape(1);
	long long int n = nr*nc;
	
	DataType* in1 = (DataType*) buf_in1.ptr;
	DataType* in2 = (DataType*) buf_in2.ptr;
	
	DataType* out = (DataType*) malloc(n*sizeof(DataType));

	DataType* gpu;
	cudaMalloc((void**)&gpu, 2*n*sizeof(DataType));
	
	cudaMemcpy(gpu,ptr_py_in1,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	cudaMemcpy(gpu+n,ptr_py_in2,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	
	matrix_diff<DataType>(nr,nc,gpu,gpu+n);
	
	cudaMemcpy(out,gpu,sizeof(DataType)*n,cudaMemcpyDeviceToHost);	
	cudaFree(gpu);

	py::capsule free_when_done(out, free);
	return py::array_t<DataType, py::array::c_style> 
	(
		{nr/2,nc/2},
		{nc*sizeof(DataType),2*sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<DataType,py::array::c_style> 
matrix_div_py(
py::array_t<DataType,py::array::c_style> py_in1,
py::array_t<DataType,py::array::c_style> py_in2)
{
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	if (buf_in1.ndim != 2 || buf_in2.ndim != 2)
	{
		throw std::runtime_error("U dumbdumb dimension must be 2.");
	}	
	if (buf_in1.shape(0) != buf_in2.shape(0) || buf_in1.shape(1) != buf_in2.shape(1))
	{
		throw std::runtime_error("U dumbdumb shapes must be same.");
	}	

	long long int nr = buf_in1.shape(0);
	long long int nc = buf_in1.shape(1);
	long long int n = nr*nc;
	
	DataType* in1 = (DataType*) buf_in1.ptr;
	DataType* in2 = (DataType*) buf_in2.ptr;
	
	DataType* out = (DataType*) malloc(n*sizeof(DataType));

	DataType* gpu;
	cudaMalloc((void**)&gpu, 2*n*sizeof(DataType));
	
	cudaMemcpy(gpu,ptr_py_in1,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	cudaMemcpy(gpu+n,ptr_py_in2,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	
	matrix_div<DataType>(nr,nc,gpu,gpu+n);
	
	cudaMemcpy(out,gpu,sizeof(DataType)*n,cudaMemcpyDeviceToHost);	
	cudaFree(gpu);

	py::capsule free_when_done(out, free);
	return py::array_t<DataType, py::array::c_style> 
	(
		{nr,nc},
		{nc*sizeof(DataType),sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
matrix_div_complex_py(
py::array_t<std::complex<DataType>,py::array::c_style> py_in1,
py::array_t<std::complex<DataType>,py::array::c_style> py_in2)
{
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	if (buf_in1.ndim != 2 || buf_in2.ndim != 2)
	{
		throw std::runtime_error("U dumbdumb dimension must be 2.");
	}	
	if (buf_in1.shape(0) != buf_in2.shape(0) || buf_in1.shape(1) != buf_in2.shape(1))
	{
		throw std::runtime_error("U dumbdumb shapes must be same.");
	}	

	long long int nr = 2*buf_in1.shape(0);
	long long int nc = 2*buf_in1.shape(1);
	long long int n = nr*nc;
	
	DataType* in1 = (DataType*) buf_in1.ptr;
	DataType* in2 = (DataType*) buf_in2.ptr;
	
	DataType* out = (DataType*) malloc(n*sizeof(DataType));

	DataType* gpu;
	cudaMalloc((void**)&gpu, 2*n*sizeof(DataType));
	
	cudaMemcpy(gpu,ptr_py_in1,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	cudaMemcpy(gpu+n,ptr_py_in2,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	
	matrix_div<DataType>(nr,nc,gpu,gpu+n);
	
	cudaMemcpy(out,gpu,sizeof(DataType)*n,cudaMemcpyDeviceToHost);	
	cudaFree(gpu);

	py::capsule free_when_done(out, free);
	return py::array_t<DataType, py::array::c_style> 
	(
		{nr/2,nc/2},
		{nc*sizeof(DataType),2*sizeof(DataType)},
		out,
		free_when_done	
	);
}