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

template<class DataType>
py::array_t<DataType,py::array::c_style> 
autocorrelation_block_cuda_py(py::array_t<DataType,py::array::c_style> py_in, long long int size)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = buf_in.size;
	long long int howmany = n/size;
	long long int transfer_size = 1<<24;
	long long int data_size = howmany*size*sizeof(DataType);
	long long int transfers, remaining;

	if(data_size/transfer_size == 0){transfer_size = data_size; transfers=1; remaining=0;}
	else{transfers = data_size/transfer_size; remaining = data_size-transfers*transfer_size;}

	long long int batch = transfer_size/size/sizeof(DataType);

	DataType* ptr_py_in = (DataType*) buf_in.ptr;
	
	DataType* out;
	out = (DataType*) malloc(howmany*(size/2+1)*sizeof(DataType));

	std::complex<DataType>* gpu;
	cudaMalloc((void**)&gpu, howmany*(size/2+1)*sizeof(std::complex<DataType>));

	cufftHandle plan;
	makePlan<DataType, long long int>(&plan,size,batch);

	cudaStream_t streams[2];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);	

	cudaMemcpy2DAsync(gpu,
				(size+2)*sizeof(DataType),
				ptr_py_in,
				size*sizeof(DataType),
				size*sizeof(DataType),
				batch,
				cudaMemcpyHostToDevice,
				streams[0]);
	
	for(long long int i=1;i<transfers;i++)
	{
		cudaMemcpy2DAsync(gpu+i*batch*(size/2+1),
						(size+2)*sizeof(DataType),
						ptr_py_in+i*transfer_size/sizeof(DataType),
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						batch,
						cudaMemcpyHostToDevice,
						streams[1]);
		
		rFFT_Block_Async_CUDA(gpu+(i-1)*batch*(size/2+1),plan,streams[0]);
		autocorrelation_cuda(batch*(size/2+1),
						gpu+(i-1)*batch*(size/2+1),
						reinterpret_cast<DataType*>(gpu)+(i-1)*batch*(size/2+1));
	}

	rFFT_Block_Async_CUDA(gpu+(transfers-1)*batch*(size/2+1),plan,streams[0]);
	autocorrelation_cuda(batch*(size/2+1),
						gpu+(transfers-1)*batch*(size/2+1),
						reinterpret_cast<DataType*>(gpu)+(transfers-1)*batch*(size/2+1));
	cufftDestroy(plan);

	if(remaining != 0)
	{
		cudaMemcpy2DAsync(gpu+transfers*batch*(size/2+1),
						(size+2)*sizeof(DataType),
						ptr_py_in+transfers*transfer_size/sizeof(DataType),
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						remaining/size/sizeof(DataType),
						cudaMemcpyHostToDevice,
						streams[0]);

		makePlan<DataType,long long int>(&plan,size,remaining/size/sizeof(DataType));
		rFFT_Block_Async_CUDA(gpu+transfers*batch*(size/2+1),plan,streams[0]);
		autocorrelation_cuda(remaining/size/sizeof(DataType)*(size/2+1),
							gpu+transfers*batch*(size/2+1),
							reinterpret_cast<DataType*>(gpu)+transfers*batch*(size/2+1));
	
		reduction_general(howmany*(size/2+1),reinterpret_cast<DataType*>(gpu),size/2+1);
		cufftDestroy(plan);
	}
	else{reduction_general(howmany*(size/2+1),reinterpret_cast<DataType*>(gpu),size/2+1);}

	cudaMemcpy(out,gpu,(size/2+1)*sizeof(DataType),cudaMemcpyDeviceToHost);
	for(long long int i=0;i<(size/2+1);i++){out[i] /= howmany;}

	cudaFree(gpu);
	cudaStreamDestroy(streams[0]);
	cudaStreamDestroy(streams[1]);



	py::capsule free_when_done( out, free );
	return py::array_t<DataType, py::array::c_style> 
	(
		{(size/2+1)},
		{sizeof(DataType)},
		out,
		free_when_done	
	);
}
