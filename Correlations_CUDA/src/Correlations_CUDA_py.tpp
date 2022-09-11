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
	out = (DataType*) malloc((size/2+1)*sizeof(DataType));

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
	else{reduction(howmany*(size/2+1),reinterpret_cast<DataType*>(gpu),size/2+1);}

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

template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style> 
cross_correlation_block_cuda_py(
				py::array_t<DataType,py::array::c_style> py_in1,
				py::array_t<DataType,py::array::c_style> py_in2, long long int size)
{
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	if (buf_in1.ndim != 1 || buf_in2.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = std::min(buf_in1.size,buf_in2.size);
	long long int howmany = n/size;
	long long int transfer_size = 1<<24;
	long long int data_size = howmany*size*sizeof(DataType);
	long long int transfers, remaining;

	if(data_size/transfer_size == 0){transfer_size = data_size; transfers=1; remaining=0;}
	else{transfers = data_size/transfer_size; remaining = data_size-transfers*transfer_size;}

	long long int batch = transfer_size/size/sizeof(DataType);

	DataType* ptr_py_in1 = (DataType*) buf_in1.ptr;
	DataType* ptr_py_in2 = (DataType*) buf_in2.ptr;
	
	std::complex<DataType>* out;
	out = (std::complex<DataType>*) malloc((size/2+1)*sizeof(std::complex<DataType>));

	std::complex<DataType>* gpu1;
	cudaMalloc((void**)&gpu1, howmany*(size/2+1)*sizeof(std::complex<DataType>));

	std::complex<DataType>* gpu2;
	cudaMalloc((void**)&gpu2, howmany*(size/2+1)*sizeof(std::complex<DataType>));

	cufftHandle plan;
	makePlan<DataType, long long int>(&plan,size,batch);

	cudaStream_t streams[2];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);	

	cudaMemcpy2DAsync(gpu1,
				(size+2)*sizeof(DataType),
				ptr_py_in1,
				size*sizeof(DataType),
				size*sizeof(DataType),
				batch,
				cudaMemcpyHostToDevice,
				streams[0]);

	cudaMemcpy2DAsync(gpu2,
				(size+2)*sizeof(DataType),
				ptr_py_in2,
				size*sizeof(DataType),
				size*sizeof(DataType),
				batch,
				cudaMemcpyHostToDevice,
				streams[1]);
	
	for(long long int i=1;i<transfers;i++)
	{
		cudaMemcpy2DAsync(gpu1+i*batch*(size/2+1),
						(size+2)*sizeof(DataType),
						ptr_py_in1+i*transfer_size/sizeof(DataType),
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						batch,
						cudaMemcpyHostToDevice,
						streams[1]);

		cudaMemcpy2DAsync(gpu2+i*batch*(size/2+1),
						(size+2)*sizeof(DataType),
						ptr_py_in2+i*transfer_size/sizeof(DataType),
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						batch,
						cudaMemcpyHostToDevice,
						streams[1]);
		
		rFFT_Block_Async_CUDA(gpu1+(i-1)*batch*(size/2+1),plan,streams[0]);
		rFFT_Block_Async_CUDA(gpu2+(i-1)*batch*(size/2+1),plan,streams[0]);

		cross_correlation_cuda(batch*(size/2+1),
						gpu1+(i-1)*batch*(size/2+1),
						gpu2+(i-1)*batch*(size/2+1),
						gpu1+(i-1)*batch*(size/2+1));
	}

	rFFT_Block_Async_CUDA(gpu1+(transfers-1)*batch*(size/2+1),plan,streams[0]);
	rFFT_Block_Async_CUDA(gpu2+(transfers-1)*batch*(size/2+1),plan,streams[0]);
	cross_correlation_cuda(batch*(size/2+1),
					gpu1+(transfers-1)*batch*(size/2+1),
					gpu2+(transfers-1)*batch*(size/2+1),
					gpu1+(transfers-1)*batch*(size/2+1));
	cufftDestroy(plan);

	if(remaining != 0)
	{
		cudaMemcpy2DAsync(gpu1+transfers*batch*(size/2+1),
						(size+2)*sizeof(DataType),
						ptr_py_in1+transfers*transfer_size/sizeof(DataType),
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						remaining/size/sizeof(DataType),
						cudaMemcpyHostToDevice,
						streams[0]);

		cudaMemcpy2DAsync(gpu2+transfers*batch*(size/2+1),
						(size+2)*sizeof(DataType),
						ptr_py_in2+transfers*transfer_size/sizeof(DataType),
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						remaining/size/sizeof(DataType),
						cudaMemcpyHostToDevice,
						streams[0]);

		makePlan<DataType,long long int>(&plan,size,remaining/size/sizeof(DataType));
		rFFT_Block_Async_CUDA(gpu1+transfers*batch*(size/2+1),plan,streams[0]);
		rFFT_Block_Async_CUDA(gpu2+transfers*batch*(size/2+1),plan,streams[0]);
		cross_correlation_cuda(remaining/size/sizeof(DataType)*(size/2+1),
							gpu1+transfers*batch*(size/2+1),
							gpu2+transfers*batch*(size/2+1),
							gpu1+transfers*batch*(size/2+1));
	
		reduction_general(howmany*(size+2),reinterpret_cast<DataType*>(gpu1),size+2);
		cufftDestroy(plan);
	}
	else{reduction(howmany*(size+2),reinterpret_cast<DataType*>(gpu1),size+2);}

	cudaMemcpy(out,gpu1,(size+2)*sizeof(DataType),cudaMemcpyDeviceToHost);
	for(long long int i=0;i<(size/2+1);i++){out[i] /= howmany;}

	cudaFree(gpu1);
	cudaFree(gpu2);
	cudaStreamDestroy(streams[0]);
	cudaStreamDestroy(streams[1]);

	py::capsule free_when_done( out, free );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{(size/2+1)},
		{sizeof(std::complex<DataType>)},
		out,
		free_when_done	
	);
}

template<class DataType>
std::tuple<
py::array_t<DataType,py::array::c_style>, 
py::array_t<DataType,py::array::c_style>,
py::array_t<std::complex<DataType>,py::array::c_style>> 
complete_correlation_block_cuda_py(
				py::array_t<DataType,py::array::c_style> py_in1,
				py::array_t<DataType,py::array::c_style> py_in2, long long int size)
{
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	if (buf_in1.ndim != 1 || buf_in2.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = std::min(buf_in1.size,buf_in2.size);
	long long int howmany = n/size;
	long long int transfer_size = 1<<24;
	long long int data_size = howmany*size*sizeof(DataType);
	long long int transfers, remaining;

	if(data_size/transfer_size == 0){transfer_size = data_size; transfers=1; remaining=0;}
	else{transfers = data_size/transfer_size; remaining = data_size-transfers*transfer_size;}

	long long int batch = transfer_size/size/sizeof(DataType);

	DataType* ptr_py_in1 = (DataType*) buf_in1.ptr;
	DataType* ptr_py_in2 = (DataType*) buf_in2.ptr;

	DataType* out1;
	out1 = (DataType*) malloc((size/2+1)*sizeof(DataType));

	DataType* out2;
	out2 = (DataType*) malloc((size/2+1)*sizeof(DataType));

	std::complex<DataType>* out3;
	out3 = (std::complex<DataType>*) malloc((size/2+1)*sizeof(std::complex<DataType>));

	std::complex<DataType>* gpu1;
	cudaMalloc((void**)&gpu1, howmany*(size/2+1)*sizeof(std::complex<DataType>));

	std::complex<DataType>* gpu2;
	cudaMalloc((void**)&gpu2, howmany*(size/2+1)*sizeof(std::complex<DataType>));

	DataType* gpu3;
	cudaMalloc((void**)&gpu3, howmany*(size/2+1)*sizeof(DataType));

	cufftHandle plan;
	makePlan<DataType, long long int>(&plan,size,batch);

	cudaStream_t streams[2];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);	

	cudaMemcpy2DAsync(gpu1,
				(size+2)*sizeof(DataType),
				ptr_py_in1,
				size*sizeof(DataType),
				size*sizeof(DataType),
				batch,
				cudaMemcpyHostToDevice,
				streams[0]);

	cudaMemcpy2DAsync(gpu2,
				(size+2)*sizeof(DataType),
				ptr_py_in2,
				size*sizeof(DataType),
				size*sizeof(DataType),
				batch,
				cudaMemcpyHostToDevice,
				streams[1]);
	
	for(long long int i=1;i<transfers;i++)
	{
		cudaMemcpy2DAsync(gpu1+i*batch*(size/2+1),
						(size+2)*sizeof(DataType),
						ptr_py_in1+i*transfer_size/sizeof(DataType),
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						batch,
						cudaMemcpyHostToDevice,
						streams[1]);

		cudaMemcpy2DAsync(gpu2+i*batch*(size/2+1),
						(size+2)*sizeof(DataType),
						ptr_py_in2+i*transfer_size/sizeof(DataType),
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						batch,
						cudaMemcpyHostToDevice,
						streams[1]);
		
		rFFT_Block_Async_CUDA(gpu1+(i-1)*batch*(size/2+1),plan,streams[0]);
		rFFT_Block_Async_CUDA(gpu2+(i-1)*batch*(size/2+1),plan,streams[0]);

		complete_correlation_cuda(batch*(size/2+1),
						gpu1+(i-1)*batch*(size/2+1),
						gpu2+(i-1)*batch*(size/2+1),
						reinterpret_cast<DataType*>(gpu2)+(i-1)*batch*(size/2+1),
						gpu3+(i-1)*batch*(size/2+1),
						gpu1+(i-1)*batch*(size/2+1));
	}

	rFFT_Block_Async_CUDA(gpu1+(transfers-1)*batch*(size/2+1),plan,streams[0]);
	rFFT_Block_Async_CUDA(gpu2+(transfers-1)*batch*(size/2+1),plan,streams[0]);
	complete_correlation_cuda(batch*(size/2+1),
					gpu1+(transfers-1)*batch*(size/2+1),
					gpu2+(transfers-1)*batch*(size/2+1),
					reinterpret_cast<DataType*>(gpu2)+(transfers-1)*batch*(size/2+1),
					gpu3+(transfers-1)*batch*(size/2+1),
					gpu1+(transfers-1)*batch*(size/2+1));
	cufftDestroy(plan);

	if(remaining != 0)
	{
		cudaMemcpy2DAsync(gpu1+transfers*batch*(size/2+1),
						(size+2)*sizeof(DataType),
						ptr_py_in1+transfers*transfer_size/sizeof(DataType),
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						remaining/size/sizeof(DataType),
						cudaMemcpyHostToDevice,
						streams[0]);

		cudaMemcpy2DAsync(gpu2+transfers*batch*(size/2+1),
						(size+2)*sizeof(DataType),
						ptr_py_in2+transfers*transfer_size/sizeof(DataType),
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						remaining/size/sizeof(DataType),
						cudaMemcpyHostToDevice,
						streams[0]);

		makePlan<DataType,long long int>(&plan,size,remaining/size/sizeof(DataType));
		rFFT_Block_Async_CUDA(gpu1+transfers*batch*(size/2+1),plan,streams[0]);
		rFFT_Block_Async_CUDA(gpu2+transfers*batch*(size/2+1),plan,streams[0]);
		complete_correlation_cuda(batch*(size/2+1),
						gpu1+transfers*batch*(size/2+1),
						gpu2+transfers*batch*(size/2+1),
						reinterpret_cast<DataType*>(gpu2)+transfers*batch*(size/2+1),
						gpu3+transfers*batch*(size/2+1),
						gpu1+transfers*batch*(size/2+1));
	
		reduction_general(howmany*(size+2),reinterpret_cast<DataType*>(gpu1),size+2);
		reduction_general(howmany*(size/2+1),reinterpret_cast<DataType*>(gpu2),size/2+1);
		reduction_general(howmany*(size/2+1),gpu3,size/2+1);
		cufftDestroy(plan);
	}
	else
	{
		reduction(howmany*(size+2),reinterpret_cast<DataType*>(gpu1),size+2);
		reduction(howmany*(size/2+1),reinterpret_cast<DataType*>(gpu2),size/2+1);
		reduction(howmany*(size/2+1),gpu3,size/2+1);
	}

	cudaMemcpy(out1,gpu2,(size/2+1)*sizeof(DataType),cudaMemcpyDeviceToHost);
	cudaMemcpy(out2,gpu3,(size/2+1)*sizeof(DataType),cudaMemcpyDeviceToHost);
	cudaMemcpy(out3,gpu1,(size+2)*sizeof(DataType),cudaMemcpyDeviceToHost);
	
	for(long long int i=0;i<(size/2+1);i++)
	{
		out1[i] /= howmany;
		out2[i] /= howmany;
		out3[i] /= howmany;
	}

	cudaFree(gpu1);
	cudaFree(gpu2);
	cudaFree(gpu3);
	cudaStreamDestroy(streams[0]);
	cudaStreamDestroy(streams[1]);

	py::capsule free_when_done1( out1, free );
	py::capsule free_when_done2( out2, free );
	py::capsule free_when_done3( out3, free );
	
	return std::make_tuple(
		py::array_t<DataType, py::array::c_style>
		({size/2+1},{sizeof(DataType)},out1,free_when_done1),
		py::array_t<DataType, py::array::c_style>
		({size/2+1},{sizeof(DataType)},out2,free_when_done2),
		py::array_t<std::complex<DataType>, py::array::c_style>
		({(size/2+1)},{sizeof(std::complex<DataType>)},out3,free_when_done3)
		);
}

template<class DataType, class DataType2>
py::array_t<DataType2,py::array::c_style> 
digitizer_autocorrelation_cuda_py(
				py::array_t<DataType,py::array::c_style> py_in, 
				long long int size, 
				DataType2 conv,
				DataType offset)
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
	
	DataType2* out;
	out = (DataType2*) malloc((size/2+1)*sizeof(DataType2));

	std::complex<DataType2>* gpu;
	cudaMalloc((void**)&gpu, howmany*(size/2+1)*sizeof(std::complex<DataType2>));

	DataType* gpu_in;
	cudaMalloc((void**)&gpu_in, howmany*(size+2)*sizeof(DataType));

	cufftHandle plan;
	makePlan<DataType2, long long int>(&plan,size,batch);

	cudaStream_t streams[2];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);	

	cudaMemcpy2DAsync(gpu_in,
				(size+2)*sizeof(DataType),
				ptr_py_in,
				size*sizeof(DataType),
				size*sizeof(DataType),
				batch,
				cudaMemcpyHostToDevice,
				streams[0]);

	for(long long int i=1;i<transfers;i++)
	{
		cudaMemcpy2DAsync(gpu_in+i*batch*(size+2),
						(size+2)*sizeof(DataType),
						ptr_py_in+i*batch*size,
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						batch,
						cudaMemcpyHostToDevice,
						streams[1]);

		convert(batch*(size+2),
						gpu_in+(i-1)*batch*(size+2),
						reinterpret_cast<DataType2*>(gpu)+(i-1)*batch*(size+2),
						conv,
						offset,
						streams[0]);
		
		rFFT_Block_Async_CUDA(gpu+(i-1)*batch*(size/2+1),plan,streams[0]);
		autocorrelation_cuda(batch*(size/2+1),
						gpu+(i-1)*batch*(size/2+1),
						reinterpret_cast<DataType2*>(gpu)+(i-1)*batch*(size/2+1));
	}

	convert(batch*(size+2),
					gpu_in+(transfers-1)*batch*(size+2),
					reinterpret_cast<DataType2*>(gpu)+(transfers-1)*batch*(size+2),
					conv,
					offset,
					streams[0]);

	rFFT_Block_Async_CUDA(gpu+(transfers-1)*batch*(size/2+1),plan,streams[0]);
	autocorrelation_cuda(batch*(size/2+1),
						gpu+(transfers-1)*batch*(size/2+1),
						reinterpret_cast<DataType2*>(gpu)+(transfers-1)*batch*(size/2+1));
	cufftDestroy(plan);

	if(remaining != 0)
	{
		cudaMemcpy2DAsync(gpu_in+transfers*batch*(size+2),
						(size+2)*sizeof(DataType),
						ptr_py_in+transfers*transfer_size/sizeof(DataType),
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						remaining/size/sizeof(DataType),
						cudaMemcpyHostToDevice,
						streams[0]);

		makePlan<DataType2,long long int>(&plan,size,remaining/size/sizeof(DataType));

		convert(remaining/sizeof(DataType),
						gpu_in+transfers*batch*size,
						reinterpret_cast<DataType2*>(gpu)+transfers*batch*(size+2),
						conv,
						offset,
						streams[0]);

		rFFT_Block_Async_CUDA(gpu+transfers*batch*(size/2+1),plan,streams[0]);
		autocorrelation_cuda(remaining/size/sizeof(DataType)*(size/2+1),
							gpu+transfers*batch*(size/2+1),
							reinterpret_cast<DataType2*>(gpu)+transfers*batch*(size/2+1));
	
		reduction_general(howmany*(size/2+1),reinterpret_cast<DataType2*>(gpu),size/2+1);
		cufftDestroy(plan);
	}
	else{reduction(howmany*(size/2+1),reinterpret_cast<DataType2*>(gpu),size/2+1);}

	cudaMemcpy(out,gpu,(size/2+1)*sizeof(DataType2),cudaMemcpyDeviceToHost);
	for(long long int i=0;i<(size/2+1);i++){out[i] /= howmany;}

	cudaFree(gpu);
	cudaFree(gpu_in);
	cudaStreamDestroy(streams[0]);
	cudaStreamDestroy(streams[1]);

	py::capsule free_when_done( out, free );
	return py::array_t<DataType2, py::array::c_style> 
	(
		{(size/2+1)},
		{sizeof(DataType2)},
		out,
		free_when_done	
	);
}

template<class DataType, class DataType2>
py::array_t<std::complex<DataType2>,py::array::c_style> 
digitizer_crosscorrelation_cuda_py(
				py::array_t<DataType,py::array::c_style> py_in1, 
				py::array_t<DataType,py::array::c_style> py_in2, 
				long long int size, 
				DataType2 conv,
				DataType offset)
{
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	if (buf_in1.ndim != 1 || buf_in2.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = std::min(buf_in1.size,buf_in2.size);
	long long int howmany = n/size;
	long long int transfer_size = 1<<24;
	long long int data_size = howmany*size*sizeof(DataType);
	long long int transfers, remaining;

	if(data_size/transfer_size == 0){transfer_size = data_size; transfers=1; remaining=0;}
	else{transfers = data_size/transfer_size; remaining = data_size-transfers*transfer_size;}

	long long int batch = transfer_size/size/sizeof(DataType);

	DataType* ptr_py_in1 = (DataType*) buf_in1.ptr;
	DataType* ptr_py_in2 = (DataType*) buf_in2.ptr;
	
	std::complex<DataType2>* out;
	out = (std::complex<DataType2>*) malloc((size/2+1)*sizeof(std::complex<DataType2>));

	std::complex<DataType2>* gpu1;
	cudaMalloc((void**)&gpu1, 2*howmany*(size/2+1)*sizeof(std::complex<DataType2>));
	std::complex<DataType2>* gpu2 = gpu1+howmany*(size/2+1);

	DataType* gpu_in1;
	cudaMalloc((void**)&gpu_in1, 2*howmany*(size+2)*sizeof(DataType));
	DataType* gpu_in2 = gpu_in1+howmany*(size+2);

	cufftHandle plan;
	makePlan<DataType2, long long int>(&plan,size,batch);

	cudaStream_t streams[2];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);	

	cudaMemcpy2DAsync(gpu_in1,
				(size+2)*sizeof(DataType),
				ptr_py_in1,
				size*sizeof(DataType),
				size*sizeof(DataType),
				batch,
				cudaMemcpyHostToDevice,
				streams[0]);

	cudaMemcpy2DAsync(gpu_in2,
				(size+2)*sizeof(DataType),
				ptr_py_in2,
				size*sizeof(DataType),
				size*sizeof(DataType),
				batch,
				cudaMemcpyHostToDevice,
				streams[0]);

	for(long long int i=1;i<transfers;i++)
	{
		cudaMemcpy2DAsync(gpu_in1+i*batch*(size+2),
						(size+2)*sizeof(DataType),
						ptr_py_in1+i*batch*size,
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						batch,
						cudaMemcpyHostToDevice,
						streams[1]);

		cudaMemcpy2DAsync(gpu_in2+i*batch*(size+2),
						(size+2)*sizeof(DataType),
						ptr_py_in2+i*batch*size,
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						batch,
						cudaMemcpyHostToDevice,
						streams[1]);

		convert(batch*(size+2),
						gpu_in1+(i-1)*batch*(size+2),
						reinterpret_cast<DataType2*>(gpu1)+(i-1)*batch*(size+2),
						conv,
						offset,
						streams[0]);

		convert(batch*(size+2),
						gpu_in2+(i-1)*batch*(size+2),
						reinterpret_cast<DataType2*>(gpu2)+(i-1)*batch*(size+2),
						conv,
						offset,
						streams[0]);

		rFFT_Block_Async_CUDA(gpu1+(i-1)*batch*(size/2+1),plan,streams[0]);
		rFFT_Block_Async_CUDA(gpu2+(i-1)*batch*(size/2+1),plan,streams[0]);
		
		cross_correlation_cuda(batch*(size/2+1),
						gpu1+(i-1)*batch*(size/2+1),
						gpu2+(i-1)*batch*(size/2+1),
						gpu1+(i-1)*batch*(size/2+1));
	}

	convert(batch*(size+2),
					gpu_in1+(transfers-1)*batch*(size+2),
					reinterpret_cast<DataType2*>(gpu1)+(transfers-1)*batch*(size+2),
					conv,
					offset,
					streams[0]);

	convert(batch*(size+2),
					gpu_in2+(transfers-1)*batch*(size+2),
					reinterpret_cast<DataType2*>(gpu2)+(transfers-1)*batch*(size+2),
					conv,
					offset,
					streams[0]);

	rFFT_Block_Async_CUDA(gpu1+(transfers-1)*batch*(size/2+1),plan,streams[0]);
	rFFT_Block_Async_CUDA(gpu2+(transfers-1)*batch*(size/2+1),plan,streams[0]);
	
	cross_correlation_cuda(batch*(size/2+1),
						gpu1+(transfers-1)*batch*(size/2+1),
						gpu2+(transfers-1)*batch*(size/2+1),
						gpu1+(transfers-1)*batch*(size/2+1));

	cufftDestroy(plan);

	if(remaining != 0)
	{
		cudaMemcpy2DAsync(gpu_in1+transfers*batch*(size+2),
						(size+2)*sizeof(DataType),
						ptr_py_in1+transfers*transfer_size/sizeof(DataType),
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						remaining/size/sizeof(DataType),
						cudaMemcpyHostToDevice,
						streams[0]);

		cudaMemcpy2DAsync(gpu_in2+transfers*batch*(size+2),
						(size+2)*sizeof(DataType),
						ptr_py_in2+transfers*transfer_size/sizeof(DataType),
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						remaining/size/sizeof(DataType),
						cudaMemcpyHostToDevice,
						streams[0]);

		makePlan<DataType2,long long int>(&plan,size,remaining/size/sizeof(DataType));

		convert(remaining/sizeof(DataType),
						gpu_in1+transfers*batch*size,
						reinterpret_cast<DataType2*>(gpu1)+transfers*batch*(size+2),
						conv,
						offset,
						streams[0]);

		convert(remaining/sizeof(DataType),
						gpu_in2+transfers*batch*size,
						reinterpret_cast<DataType2*>(gpu2)+transfers*batch*(size+2),
						conv,
						offset,
						streams[0]);

		rFFT_Block_Async_CUDA(gpu1+transfers*batch*(size/2+1),plan,streams[0]);
		rFFT_Block_Async_CUDA(gpu2+transfers*batch*(size/2+1),plan,streams[0]);
		
		cross_correlation_cuda(remaining/size/sizeof(DataType)*(size/2+1),
							gpu1+transfers*batch*(size/2+1),
							gpu2+transfers*batch*(size/2+1),
							gpu1+transfers*batch*(size/2+1));
	
		reduction_general(howmany*(size+2),reinterpret_cast<DataType2*>(gpu1),size+2);
		cufftDestroy(plan);
	}
	else{reduction(howmany*(size+2),reinterpret_cast<DataType2*>(gpu1),size+2);}

	cudaMemcpy(out,gpu1,(size+2)*sizeof(DataType2),cudaMemcpyDeviceToHost);
	for(long long int i=0;i<(size/2+1);i++){out[i] /= howmany;}

	cudaFree(gpu1);
	cudaFree(gpu_in1);
	cudaStreamDestroy(streams[0]);
	cudaStreamDestroy(streams[1]);

	py::capsule free_when_done( out, free );
	return py::array_t<std::complex<DataType2>, py::array::c_style> 
	(
		{(size/2+1)},
		{sizeof(std::complex<DataType2>)},
		out,
		free_when_done	
	);
}

template<class DataType, class DataType2>
std::tuple<
py::array_t<DataType2,py::array::c_style>, 
py::array_t<DataType2,py::array::c_style>,
py::array_t<std::complex<DataType2>,py::array::c_style>> 
digitizer_completecorrelation_cuda_py(
				py::array_t<DataType,py::array::c_style> py_in1, 
				py::array_t<DataType,py::array::c_style> py_in2, 
				long long int size, 
				DataType2 conv,
				DataType offset)
{
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	if (buf_in1.ndim != 1 || buf_in2.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = std::min(buf_in1.size,buf_in2.size);
	long long int howmany = n/size;
	long long int transfer_size = 1<<24;
	long long int data_size = howmany*size*sizeof(DataType);
	long long int transfers, remaining;

	if(data_size/transfer_size == 0){transfer_size = data_size; transfers=1; remaining=0;}
	else{transfers = data_size/transfer_size; remaining = data_size-transfers*transfer_size;}

	long long int batch = transfer_size/size/sizeof(DataType);

	DataType* ptr_py_in1 = (DataType*) buf_in1.ptr;
	DataType* ptr_py_in2 = (DataType*) buf_in2.ptr;
	
	DataType2* out1;
	out1 = (DataType2*) malloc((size/2+1)*sizeof(DataType2));

	DataType2* out2;
	out2 = (DataType2*) malloc((size/2+1)*sizeof(DataType2));
	
	std::complex<DataType2>* out3;
	out3 = (std::complex<DataType2>*) malloc((size/2+1)*sizeof(std::complex<DataType2>));

	std::complex<DataType2>* gpu1;
	cudaMalloc((void**)&gpu1, 5*howmany*(size/2+1)*sizeof(DataType2));
	std::complex<DataType2>* gpu2 = gpu1+howmany*(size/2+1);
	DataType2* gpu3 = (DataType2*) (gpu1+2*howmany*(size/2+1));

	DataType* gpu_in1;
	cudaMalloc((void**)&gpu_in1, 2*howmany*(size+2)*sizeof(DataType));
	DataType* gpu_in2 = gpu_in1+howmany*(size+2);

	cufftHandle plan;
	makePlan<DataType2, long long int>(&plan,size,batch);

	cudaStream_t streams[2];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);	

	cudaMemcpy2DAsync(gpu_in1,
				(size+2)*sizeof(DataType),
				ptr_py_in1,
				size*sizeof(DataType),
				size*sizeof(DataType),
				batch,
				cudaMemcpyHostToDevice,
				streams[0]);

	cudaMemcpy2DAsync(gpu_in2,
				(size+2)*sizeof(DataType),
				ptr_py_in2,
				size*sizeof(DataType),
				size*sizeof(DataType),
				batch,
				cudaMemcpyHostToDevice,
				streams[0]);

	for(long long int i=1;i<transfers;i++)
	{
		cudaMemcpy2DAsync(gpu_in1+i*batch*(size+2),
						(size+2)*sizeof(DataType),
						ptr_py_in1+i*batch*size,
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						batch,
						cudaMemcpyHostToDevice,
						streams[1]);

		cudaMemcpy2DAsync(gpu_in2+i*batch*(size+2),
						(size+2)*sizeof(DataType),
						ptr_py_in2+i*batch*size,
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						batch,
						cudaMemcpyHostToDevice,
						streams[1]);

		convert(batch*(size+2),
						gpu_in1+(i-1)*batch*(size+2),
						reinterpret_cast<DataType2*>(gpu1)+(i-1)*batch*(size+2),
						conv,
						offset,
						streams[0]);

		convert(batch*(size+2),
						gpu_in2+(i-1)*batch*(size+2),
						reinterpret_cast<DataType2*>(gpu2)+(i-1)*batch*(size+2),
						conv,
						offset,
						streams[0]);

		rFFT_Block_Async_CUDA(gpu1+(i-1)*batch*(size/2+1),plan,streams[0]);
		rFFT_Block_Async_CUDA(gpu2+(i-1)*batch*(size/2+1),plan,streams[0]);
		
		complete_correlation_cuda(batch*(size/2+1),
						gpu1+(i-1)*batch*(size/2+1),
						gpu2+(i-1)*batch*(size/2+1),
						reinterpret_cast<DataType2*>(gpu2)+(i-1)*batch*(size/2+1),
						gpu3+(i-1)*batch*(size/2+1),
						gpu1+(i-1)*batch*(size/2+1));
	}

	convert(batch*(size+2),
					gpu_in1+(transfers-1)*batch*(size+2),
					reinterpret_cast<DataType2*>(gpu1)+(transfers-1)*batch*(size+2),
					conv,
					offset,
					streams[0]);

	convert(batch*(size+2),
					gpu_in2+(transfers-1)*batch*(size+2),
					reinterpret_cast<DataType2*>(gpu2)+(transfers-1)*batch*(size+2),
					conv,
					offset,
					streams[0]);

	rFFT_Block_Async_CUDA(gpu1+(transfers-1)*batch*(size/2+1),plan,streams[0]);
	rFFT_Block_Async_CUDA(gpu2+(transfers-1)*batch*(size/2+1),plan,streams[0]);
	
	complete_correlation_cuda(batch*(size/2+1),
						gpu1+(transfers-1)*batch*(size/2+1),
						gpu2+(transfers-1)*batch*(size/2+1),
						reinterpret_cast<DataType2*>(gpu2)+(transfers-1)*batch*(size/2+1),
						gpu3+(transfers-1)*batch*(size/2+1),
						gpu1+(transfers-1)*batch*(size/2+1));

	cufftDestroy(plan);

	if(remaining != 0)
	{
		cudaMemcpy2DAsync(gpu_in1+transfers*batch*(size+2),
						(size+2)*sizeof(DataType),
						ptr_py_in1+transfers*transfer_size/sizeof(DataType),
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						remaining/size/sizeof(DataType),
						cudaMemcpyHostToDevice,
						streams[0]);

		cudaMemcpy2DAsync(gpu_in2+transfers*batch*(size+2),
						(size+2)*sizeof(DataType),
						ptr_py_in2+transfers*transfer_size/sizeof(DataType),
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						remaining/size/sizeof(DataType),
						cudaMemcpyHostToDevice,
						streams[0]);

		makePlan<DataType2,long long int>(&plan,size,remaining/size/sizeof(DataType));

		convert(remaining/sizeof(DataType),
						gpu_in1+transfers*batch*size,
						reinterpret_cast<DataType2*>(gpu1)+transfers*batch*(size+2),
						conv,
						offset,
						streams[0]);

		convert(remaining/sizeof(DataType),
						gpu_in2+transfers*batch*size,
						reinterpret_cast<DataType2*>(gpu2)+transfers*batch*(size+2),
						conv,
						offset,
						streams[0]);

		rFFT_Block_Async_CUDA(gpu1+transfers*batch*(size/2+1),plan,streams[0]);
		rFFT_Block_Async_CUDA(gpu2+transfers*batch*(size/2+1),plan,streams[0]);
		
		complete_correlation_cuda(remaining/size/sizeof(DataType)*(size/2+1),
							gpu1+transfers*batch*(size/2+1),
							gpu2+transfers*batch*(size/2+1),
							reinterpret_cast<DataType2*>(gpu2)+transfers*batch*(size/2+1),
							gpu3+transfers*batch*(size/2+1),
							gpu1+transfers*batch*(size/2+1));
	
		reduction_general(howmany*(size+2),reinterpret_cast<DataType2*>(gpu1),size+2);
		reduction_general(howmany*(size/2+1),reinterpret_cast<DataType2*>(gpu2),size/2+1);
		reduction_general(howmany*(size/2+1),gpu3,size/2+1);
		cufftDestroy(plan);
	}
	else
	{
		reduction(howmany*(size+2),reinterpret_cast<DataType2*>(gpu1),size+2);
		reduction(howmany*(size/2+1),reinterpret_cast<DataType2*>(gpu2),size/2+1);
		reduction(howmany*(size/2+1),gpu3,size/2+1);
	}

	cudaMemcpy(out1,gpu2,(size/2+1)*sizeof(DataType2),cudaMemcpyDeviceToHost);
	cudaMemcpy(out2,gpu3,(size/2+1)*sizeof(DataType2),cudaMemcpyDeviceToHost);
	cudaMemcpy(out3,gpu1,(size+2)*sizeof(DataType2),cudaMemcpyDeviceToHost);
	for(long long int i=0;i<(size/2+1);i++){out1[i]/=howmany;out2[i]/=howmany;out3[i]/=howmany;}

	cudaFree(gpu1);
	cudaFree(gpu_in1);
	cudaStreamDestroy(streams[0]);
	cudaStreamDestroy(streams[1]);

	py::capsule free_when_done1( out1, free );
	py::capsule free_when_done2( out2, free );
	py::capsule free_when_done3( out3, free );

	return std::make_tuple(
	py::array_t<DataType2,py::array::c_style>
	({size/2+1},{sizeof(DataType2)},out1,free_when_done1),	
	py::array_t<DataType2,py::array::c_style>
	({size/2+1},{sizeof(DataType2)},out2,free_when_done2),	
	py::array_t<std::complex<DataType2>, py::array::c_style> 
	({(size/2+1)},{sizeof(std::complex<DataType2>)},out3,free_when_done3));
}

template<class DataType>
class DigitizerAutoCorrelationCuda
{
	private:
		float conv;
		DataType offset;
		long long int N, size, howmany, data_size, transfers, remaining, batch;
		long long int transfer_size = 1<<24;
		long long int count = 0;
		DataType* gpu_raw;
		std::complex<float>* gpu_data;
		double* gpu_accumulate;
		cudaStream_t streams[2];
		cufftHandle plan, plan2;

		void cudaInit()
		{
			cudaStreamCreate(&streams[0]);
			cudaStreamCreate(&streams[1]);
			cudaMalloc((void**)&gpu_raw, howmany*(size+2)*sizeof(DataType));
			cudaMalloc((void**)&gpu_data, howmany*(size/2+1)*sizeof(std::complex<float>));
			cudaMalloc((void**)&gpu_accumulate,(size/2+1)*sizeof(double));
			makePlan<float, long long int>(&plan,size,batch);
			if(remaining != 0)
			{
				makePlan<float, long long int>(&plan2,size,remaining/size/sizeof(DataType));
			}
		}
		void cudaDel()
		{
			cudaStreamDestroy(streams[0]);
			cudaStreamDestroy(streams[1]);	
			cudaFree(gpu_raw);
			cudaFree(gpu_data);
			cudaFree(gpu_accumulate);
			cufftDestroy(plan);
			if(remaining!=0){cufftDestroy(plan2);}
		}
	public:
		DigitizerAutoCorrelationCuda
		(llint_t N_in, llint_t size_in, float conv_in, llint_t offset_in)
		{
			conv = conv_in;
			offset = (DataType) offset_in;
			howmany = N_in/size_in;
			size = size_in;
			N = howmany*size;
			data_size = N*sizeof(DataType);
			if(data_size/transfer_size == 0){transfers=1;transfer_size=data_size;remaining=0;}
			else{transfers=data_size/transfer_size;remaining=data_size-transfers*transfer_size;}
			batch = transfer_size/size/sizeof(DataType);
			cudaInit();
		}
		~DigitizerAutoCorrelationCuda(){cudaDel();}
		
		void accumulate(py::array_t<DataType,py::array::c_style> py_in)
		{
			py::buffer_info buf_in = py_in.request();
			if (buf_in.ndim != 1)
			{
				throw std::runtime_error("U dumbdumb dimension must be 1.");
			}	
			if (buf_in.size < howmany*size)
			{
				throw std::runtime_error("U dumbdumb not enough data.");
			}	

			DataType* cpu_raw = (DataType*) buf_in.ptr;

			cudaMemcpy2DAsync(gpu_raw,
							(size+2)*sizeof(DataType),
							cpu_raw,
							size*sizeof(DataType),
							size*sizeof(DataType),
							batch,
							cudaMemcpyHostToDevice,
							streams[0]);

			for(long long int i=1;i<transfers;i++)
			{
				cudaMemcpy2DAsync(gpu_raw+i*batch*(size+2),
								(size+2)*sizeof(DataType),
								cpu_raw+i*batch*size,
								sizeof(DataType)*size,
								sizeof(DataType)*size,
								batch,
								cudaMemcpyHostToDevice,
								streams[1]);

				convert(batch*(size+2),
								gpu_raw+(i-1)*batch*(size+2),
								reinterpret_cast<float*>(gpu_data)+(i-1)*batch*(size+2),
								conv,
								offset,
								streams[0]);
		
				rFFT_Block_Async_CUDA(gpu_data+(i-1)*batch*(size/2+1),plan,streams[0]);
				autocorrelation_convert(batch*(size/2+1),
								gpu_data+(i-1)*batch*(size/2+1),
								reinterpret_cast<double*>(gpu_data)+(i-1)*batch*(size/2+1));
			}


			convert(batch*(size+2),
							gpu_raw+(transfers-1)*batch*(size+2),
							reinterpret_cast<float*>(gpu_data)+(transfers-1)*batch*(size+2),
							conv,
							offset,
							streams[0]);

			rFFT_Block_Async_CUDA(gpu_data+(transfers-1)*batch*(size/2+1),plan,streams[0]);
			autocorrelation_convert(batch*(size/2+1),
								gpu_data+(transfers-1)*batch*(size/2+1),
								reinterpret_cast<double*>
								(gpu_data)+(transfers-1)*batch*(size/2+1));

			if(remaining != 0)
			{
				cudaMemcpy2DAsync(gpu_raw+transfers*batch*(size+2),
								(size+2)*sizeof(DataType),
								cpu_raw+transfers*transfer_size/sizeof(DataType),
								sizeof(DataType)*size,
								sizeof(DataType)*size,
								remaining/size/sizeof(DataType),
								cudaMemcpyHostToDevice,
								streams[0]);

				convert(remaining/sizeof(DataType),
								gpu_raw+transfers*batch*size,
								reinterpret_cast<float*>(gpu_data)+transfers*batch*(size+2),
								conv,
								offset,
								streams[0]);

				rFFT_Block_Async_CUDA(gpu_data+transfers*batch*(size/2+1),plan,streams[0]);
				autocorrelation_convert(remaining/size/sizeof(DataType)*(size/2+1),
									gpu_data+transfers*batch*(size/2+1),
									reinterpret_cast<double*>
									(gpu_data)+transfers*batch*(size/2+1));
	
				reduction_general(howmany*(size/2+1),
								reinterpret_cast<double*>(gpu_data),size/2+1);
			}
			else{reduction(howmany*(size/2+1),reinterpret_cast<double*>(gpu_data),size/2+1);}
			add_cuda(size/2+1,reinterpret_cast<double*>(gpu_data),gpu_accumulate);
			count += 1;
		}
		py::array_t<double,py::array::c_style> getResult()
		{
			if(count == 0){throw std::runtime_error("U dumbdumb accumulate first");}
			double* out;
			out = (double*) malloc((size/2+1)*sizeof(double));
			cudaMemcpy(out,gpu_accumulate,(size/2+1)*sizeof(double),cudaMemcpyDeviceToHost);
			for(long long int i=0;i<(size/2+1);i++){out[i] *= 1.0/count/howmany;}
			py::capsule free_when_done1(out,free);
			return py::array_t<double,py::array::c_style>
			({size/2+1},{sizeof(double)},out,free_when_done1);
		}
		void clear()
		{
			count = 0;
			cudaMemset(gpu_accumulate,0,(size/2+1)*sizeof(double));
		}
};

template<class DataType>
class DigitizerCrossCorrelationCuda
{
	private:
		float conv;
		DataType offset;
		long long int N, size, howmany, data_size, transfers, remaining, batch;
		long long int transfer_size = 1<<24;
		long long int count = 0;
		DataType* gpu_raw1;
		DataType* gpu_raw2;
		std::complex<float>* gpu_data1;
		std::complex<float>* gpu_data2;
		std::complex<double>* gpu_accumulate;
		cudaStream_t streams[2];
		cufftHandle plan, plan2;

		void cudaInit()
		{
			cudaStreamCreate(&streams[0]);
			cudaStreamCreate(&streams[1]);
			cudaMalloc((void**)&gpu_raw1, howmany*(size+2)*sizeof(DataType));
			cudaMalloc((void**)&gpu_raw2, howmany*(size+2)*sizeof(DataType));
			cudaMalloc((void**)&gpu_data1, howmany*(size/2+1)*sizeof(std::complex<float>));
			cudaMalloc((void**)&gpu_data2, howmany*(size/2+1)*sizeof(std::complex<float>));
			cudaMalloc((void**)&gpu_accumulate,(size/2+1)*sizeof(std::complex<double>));
			makePlan<float, long long int>(&plan,size,batch);
			if(remaining != 0)
			{
				makePlan<float, long long int>(&plan2,size,remaining/size/sizeof(DataType));
			}
		}
		void cudaDel()
		{
			cudaStreamDestroy(streams[0]);
			cudaStreamDestroy(streams[1]);	
			cudaFree(gpu_raw1);
			cudaFree(gpu_raw2);
			cudaFree(gpu_data1);
			cudaFree(gpu_data2);
			cudaFree(gpu_accumulate);
			cufftDestroy(plan);
			if(remaining!=0){cufftDestroy(plan2);}
		}
	public:
		DigitizerCrossCorrelationCuda
		(llint_t N_in, llint_t size_in, float conv_in, llint_t offset_in)
		{
			conv = conv_in;
			offset = (DataType) offset_in;
			howmany = N_in/size_in;
			size = size_in;
			N = howmany*size;
			data_size = N*sizeof(DataType);
			if(data_size/transfer_size == 0){transfers=1;transfer_size=data_size;remaining=0;}
			else{transfers=data_size/transfer_size;remaining=data_size-transfers*transfer_size;}
			batch = transfer_size/size/sizeof(DataType);
			cudaInit();
		}
		~DigitizerCrossCorrelationCuda(){cudaDel();}
		
		void accumulate(
						py::array_t<DataType,py::array::c_style> py_in1,
						py::array_t<DataType,py::array::c_style> py_in2)
		{
			py::buffer_info buf_in1 = py_in1.request();
			py::buffer_info buf_in2 = py_in2.request();
			
			if (buf_in1.ndim != 1 || buf_in2.ndim != 1)
			{
				throw std::runtime_error("U dumbdumb dimension must be 1.");
			}	
			if (buf_in1.size < howmany*size || buf_in2.size < howmany*size)
			{
				throw std::runtime_error("U dumbdumb not enough data.");
			}	

			DataType* cpu_raw1 = (DataType*) buf_in1.ptr;
			DataType* cpu_raw2 = (DataType*) buf_in2.ptr;

			cudaMemcpy2DAsync(gpu_raw1,
							(size+2)*sizeof(DataType),
							cpu_raw1,
							size*sizeof(DataType),
							size*sizeof(DataType),
							batch,
							cudaMemcpyHostToDevice,
							streams[0]);

			cudaMemcpy2DAsync(gpu_raw2,
							(size+2)*sizeof(DataType),
							cpu_raw2,
							size*sizeof(DataType),
							size*sizeof(DataType),
							batch,
							cudaMemcpyHostToDevice,
							streams[0]);

			for(long long int i=1;i<transfers;i++)
			{
				cudaMemcpy2DAsync(gpu_raw1+i*batch*(size+2),
								(size+2)*sizeof(DataType),
								cpu_raw1+i*batch*size,
								sizeof(DataType)*size,
								sizeof(DataType)*size,
								batch,
								cudaMemcpyHostToDevice,
								streams[1]);

				cudaMemcpy2DAsync(gpu_raw2+i*batch*(size+2),
								(size+2)*sizeof(DataType),
								cpu_raw2+i*batch*size,
								sizeof(DataType)*size,
								sizeof(DataType)*size,
								batch,
								cudaMemcpyHostToDevice,
								streams[1]);

				convert(batch*(size+2),
								gpu_raw1+(i-1)*batch*(size+2),
								reinterpret_cast<float*>(gpu_data1)+(i-1)*batch*(size+2),
								conv,
								offset,
								streams[0]);

				convert(batch*(size+2),
								gpu_raw2+(i-1)*batch*(size+2),
								reinterpret_cast<float*>(gpu_data2)+(i-1)*batch*(size+2),
								conv,
								offset,
								streams[0]);

		
				rFFT_Block_Async_CUDA(gpu_data1+(i-1)*batch*(size/2+1),plan,streams[0]);
				rFFT_Block_Async_CUDA(gpu_data2+(i-1)*batch*(size/2+1),plan,streams[0]);
				crosscorrelation_convert(batch*(size/2+1),
								gpu_data1+(i-1)*batch*(size/2+1),
								gpu_data2+(i-1)*batch*(size/2+1),
								reinterpret_cast<double*>(gpu_data1)+(i-1)*batch*(size/2+1),
								reinterpret_cast<double*>(gpu_data2)+(i-1)*batch*(size/2+1));
			}

			convert(batch*(size+2),
							gpu_raw1+(transfers-1)*batch*(size+2),
							reinterpret_cast<float*>(gpu_data1)+(transfers-1)*batch*(size+2),
							conv,
							offset,
							streams[0]);

			convert(batch*(size+2),
							gpu_raw2+(transfers-1)*batch*(size+2),
							reinterpret_cast<float*>(gpu_data2)+(transfers-1)*batch*(size+2),
							conv,
							offset,
							streams[0]);

			rFFT_Block_Async_CUDA(gpu_data1+(transfers-1)*batch*(size/2+1),plan,streams[0]);
			rFFT_Block_Async_CUDA(gpu_data2+(transfers-1)*batch*(size/2+1),plan,streams[0]);
			crosscorrelation_convert(batch*(size/2+1),
								gpu_data1+(transfers-1)*batch*(size/2+1),
								gpu_data2+(transfers-1)*batch*(size/2+1),
								reinterpret_cast<double*>
								(gpu_data1)+(transfers-1)*batch*(size/2+1),
								reinterpret_cast<double*>
								(gpu_data2)+(transfers-1)*batch*(size/2+1));

			if(remaining != 0)
			{
				cudaMemcpy2DAsync(gpu_raw1+transfers*batch*(size+2),
								(size+2)*sizeof(DataType),
								cpu_raw1+transfers*transfer_size/sizeof(DataType),
								sizeof(DataType)*size,
								sizeof(DataType)*size,
								remaining/size/sizeof(DataType),
								cudaMemcpyHostToDevice,
								streams[0]);

				cudaMemcpy2DAsync(gpu_raw2+transfers*batch*(size+2),
								(size+2)*sizeof(DataType),
								cpu_raw2+transfers*transfer_size/sizeof(DataType),
								sizeof(DataType)*size,
								sizeof(DataType)*size,
								remaining/size/sizeof(DataType),
								cudaMemcpyHostToDevice,
								streams[0]);

				convert(remaining/sizeof(DataType),
								gpu_raw1+transfers*batch*size,
								reinterpret_cast<float*>(gpu_data1)+transfers*batch*(size+2),
								conv,
								offset,
								streams[0]);

				convert(remaining/sizeof(DataType),
								gpu_raw2+transfers*batch*size,
								reinterpret_cast<float*>(gpu_data2)+transfers*batch*(size+2),
								conv,
								offset,
								streams[0]);

				rFFT_Block_Async_CUDA(gpu_data1+transfers*batch*(size/2+1),plan,streams[0]);
				rFFT_Block_Async_CUDA(gpu_data2+transfers*batch*(size/2+1),plan,streams[0]);
				crosscorrelation_convert(remaining/size/sizeof(DataType)*(size/2+1),
									gpu_data1+transfers*batch*(size/2+1),
									gpu_data2+transfers*batch*(size/2+1),
									reinterpret_cast<double*>
									(gpu_data1)+transfers*batch*(size/2+1),
									reinterpret_cast<double*>
									(gpu_data2)+transfers*batch*(size/2+1));
	
				reduction_general(howmany*(size/2+1),
								reinterpret_cast<double*>(gpu_data1),size/2+1);
				reduction_general(howmany*(size/2+1),
								reinterpret_cast<double*>(gpu_data2),size/2+1);
			}
			else
			{
				reduction(howmany*(size/2+1),reinterpret_cast<double*>(gpu_data1),size/2+1);
				reduction(howmany*(size/2+1),reinterpret_cast<double*>(gpu_data2),size/2+1);
			}
			add_complex_cuda(size/2+1,
							reinterpret_cast<double*>(gpu_data1),
							reinterpret_cast<double*>(gpu_data2),
							reinterpret_cast<double*>(gpu_accumulate));
			count += 1;
		}
		py::array_t<std::complex<double>,py::array::c_style> getResult()
		{
			if(count == 0){throw std::runtime_error("U dumbdumb accumulate first");}
			std::complex<double>* out;
			out = (std::complex<double>*) malloc((size+2)*sizeof(double));
			cudaMemcpy(out,gpu_accumulate,(size+2)*sizeof(double),cudaMemcpyDeviceToHost);
			for(long long int i=0;i<(size/2+1);i++){out[i] *= 1.0/count/howmany;}
			py::capsule free_when_done1(out,free);
			return py::array_t<std::complex<double>,py::array::c_style>
			({size/2+1},{2*sizeof(double)},out,free_when_done1);
		}
		void clear()
		{
			count = 0;
			cudaMemset(gpu_accumulate,0,(size+2)*sizeof(double));
		}
};

template<class DataType>
class DigitizerCompleteCorrelationCuda
{
	private:
		float conv;
		DataType offset;
		long long int N, size, howmany, data_size, transfers, remaining, batch;
		long long int transfer_size = 1<<24;
		long long int count = 0;
		DataType* gpu_raw1;
		DataType* gpu_raw2;
		std::complex<float>* gpu_data1;
		std::complex<float>* gpu_data2;
		double* gpu_data3;
		double* gpu_data4;
		std::complex<double>* gpu_accumulate1;
		double* gpu_accumulate2;
		double* gpu_accumulate3;
		cudaStream_t streams[2];
		cufftHandle plan, plan2;

		void cudaInit()
		{
			cudaStreamCreate(&streams[0]);
			cudaStreamCreate(&streams[1]);
			cudaMalloc((void**)&gpu_raw1, howmany*(size+2)*sizeof(DataType));
			cudaMalloc((void**)&gpu_raw2, howmany*(size+2)*sizeof(DataType));
			cudaMalloc((void**)&gpu_data1, howmany*(size/2+1)*sizeof(std::complex<float>));
			cudaMalloc((void**)&gpu_data2, howmany*(size/2+1)*sizeof(std::complex<float>));
			cudaMalloc((void**)&gpu_data3, howmany*(size/2+1)*sizeof(std::complex<float>));
			cudaMalloc((void**)&gpu_data4, howmany*(size/2+1)*sizeof(std::complex<float>));
			cudaMalloc((void**)&gpu_accumulate1,(size/2+1)*sizeof(std::complex<double>));
			cudaMalloc((void**)&gpu_accumulate2,(size/2+1)*sizeof(double));
			cudaMalloc((void**)&gpu_accumulate3,(size/2+1)*sizeof(double));
			makePlan<float, long long int>(&plan,size,batch);
			if(remaining != 0)
			{
				makePlan<float, long long int>(&plan2,size,remaining/size/sizeof(DataType));
			}
		}
		void cudaDel()
		{
			cudaStreamDestroy(streams[0]);
			cudaStreamDestroy(streams[1]);	
			cudaFree(gpu_raw1);
			cudaFree(gpu_raw2);
			cudaFree(gpu_data1);
			cudaFree(gpu_data2);
			cudaFree(gpu_data3);
			cudaFree(gpu_data4);
			cudaFree(gpu_accumulate1);
			cudaFree(gpu_accumulate2);
			cudaFree(gpu_accumulate3);
			cufftDestroy(plan);
			if(remaining!=0){cufftDestroy(plan2);}
		}
	public:
		DigitizerCompleteCorrelationCuda
		(llint_t N_in, llint_t size_in, float conv_in, llint_t offset_in)
		{
			conv = conv_in;
			offset = (DataType) offset_in;
			howmany = N_in/size_in;
			size = size_in;
			N = howmany*size;
			data_size = N*sizeof(DataType);
			if(data_size/transfer_size == 0){transfers=1;transfer_size=data_size;remaining=0;}
			else{transfers=data_size/transfer_size;remaining=data_size-transfers*transfer_size;}
			batch = transfer_size/size/sizeof(DataType);
			cudaInit();
		}
		~DigitizerCompleteCorrelationCuda(){cudaDel();}
		
		void accumulate(
						py::array_t<DataType,py::array::c_style> py_in1,
						py::array_t<DataType,py::array::c_style> py_in2)
		{
			py::buffer_info buf_in1 = py_in1.request();
			py::buffer_info buf_in2 = py_in2.request();
			
			if (buf_in1.ndim != 1 || buf_in2.ndim != 1)
			{
				throw std::runtime_error("U dumbdumb dimension must be 1.");
			}	
			if (buf_in1.size < howmany*size || buf_in2.size < howmany*size)
			{
				throw std::runtime_error("U dumbdumb not enough data.");
			}	

			DataType* cpu_raw1 = (DataType*) buf_in1.ptr;
			DataType* cpu_raw2 = (DataType*) buf_in2.ptr;

			cudaMemcpy2DAsync(gpu_raw1,
							(size+2)*sizeof(DataType),
							cpu_raw1,
							size*sizeof(DataType),
							size*sizeof(DataType),
							batch,
							cudaMemcpyHostToDevice,
							streams[0]);

			cudaMemcpy2DAsync(gpu_raw2,
							(size+2)*sizeof(DataType),
							cpu_raw2,
							size*sizeof(DataType),
							size*sizeof(DataType),
							batch,
							cudaMemcpyHostToDevice,
							streams[0]);

			for(long long int i=1;i<transfers;i++)
			{
				cudaMemcpy2DAsync(gpu_raw1+i*batch*(size+2),
								(size+2)*sizeof(DataType),
								cpu_raw1+i*batch*size,
								sizeof(DataType)*size,
								sizeof(DataType)*size,
								batch,
								cudaMemcpyHostToDevice,
								streams[1]);

				cudaMemcpy2DAsync(gpu_raw2+i*batch*(size+2),
								(size+2)*sizeof(DataType),
								cpu_raw2+i*batch*size,
								sizeof(DataType)*size,
								sizeof(DataType)*size,
								batch,
								cudaMemcpyHostToDevice,
								streams[1]);

				convert(batch*(size+2),
								gpu_raw1+(i-1)*batch*(size+2),
								reinterpret_cast<float*>(gpu_data1)+(i-1)*batch*(size+2),
								conv,
								offset,
								streams[0]);

				convert(batch*(size+2),
								gpu_raw2+(i-1)*batch*(size+2),
								reinterpret_cast<float*>(gpu_data2)+(i-1)*batch*(size+2),
								conv,
								offset,
								streams[0]);

		
				rFFT_Block_Async_CUDA(gpu_data1+(i-1)*batch*(size/2+1),plan,streams[0]);
				rFFT_Block_Async_CUDA(gpu_data2+(i-1)*batch*(size/2+1),plan,streams[0]);
				completecorrelation_convert(batch*(size/2+1),
								gpu_data1+(i-1)*batch*(size/2+1),
								gpu_data2+(i-1)*batch*(size/2+1),
								reinterpret_cast<double*>(gpu_data1)+(i-1)*batch*(size/2+1),
								reinterpret_cast<double*>(gpu_data2)+(i-1)*batch*(size/2+1),
								gpu_data3+(i-1)*batch*(size/2+1),
								gpu_data4+(i-1)*batch*(size/2+1));
			}

			convert(batch*(size+2),
							gpu_raw1+(transfers-1)*batch*(size+2),
							reinterpret_cast<float*>(gpu_data1)+(transfers-1)*batch*(size+2),
							conv,
							offset,
							streams[0]);

			convert(batch*(size+2),
							gpu_raw2+(transfers-1)*batch*(size+2),
							reinterpret_cast<float*>(gpu_data2)+(transfers-1)*batch*(size+2),
							conv,
							offset,
							streams[0]);

			rFFT_Block_Async_CUDA(gpu_data1+(transfers-1)*batch*(size/2+1),plan,streams[0]);
			rFFT_Block_Async_CUDA(gpu_data2+(transfers-1)*batch*(size/2+1),plan,streams[0]);
			completecorrelation_convert(batch*(size/2+1),
								gpu_data1+(transfers-1)*batch*(size/2+1),
								gpu_data2+(transfers-1)*batch*(size/2+1),
								reinterpret_cast<double*>
								(gpu_data1)+(transfers-1)*batch*(size/2+1),
								reinterpret_cast<double*>
								(gpu_data2)+(transfers-1)*batch*(size/2+1),
								gpu_data3+(transfers-1)*batch*(size/2+1),
								gpu_data4+(transfers-1)*batch*(size/2+1));

			if(remaining != 0)
			{
				cudaMemcpy2DAsync(gpu_raw1+transfers*batch*(size+2),
								(size+2)*sizeof(DataType),
								cpu_raw1+transfers*transfer_size/sizeof(DataType),
								sizeof(DataType)*size,
								sizeof(DataType)*size,
								remaining/size/sizeof(DataType),
								cudaMemcpyHostToDevice,
								streams[0]);

				cudaMemcpy2DAsync(gpu_raw2+transfers*batch*(size+2),
								(size+2)*sizeof(DataType),
								cpu_raw2+transfers*transfer_size/sizeof(DataType),
								sizeof(DataType)*size,
								sizeof(DataType)*size,
								remaining/size/sizeof(DataType),
								cudaMemcpyHostToDevice,
								streams[0]);

				convert(remaining/sizeof(DataType),
								gpu_raw1+transfers*batch*size,
								reinterpret_cast<float*>(gpu_data1)+transfers*batch*(size+2),
								conv,
								offset,
								streams[0]);

				convert(remaining/sizeof(DataType),
								gpu_raw2+transfers*batch*size,
								reinterpret_cast<float*>(gpu_data2)+transfers*batch*(size+2),
								conv,
								offset,
								streams[0]);

				rFFT_Block_Async_CUDA(gpu_data1+transfers*batch*(size/2+1),plan,streams[0]);
				rFFT_Block_Async_CUDA(gpu_data2+transfers*batch*(size/2+1),plan,streams[0]);
				completecorrelation_convert(remaining/size/sizeof(DataType)*(size/2+1),
									gpu_data1+transfers*batch*(size/2+1),
									gpu_data2+transfers*batch*(size/2+1),
									reinterpret_cast<double*>
									(gpu_data1)+transfers*batch*(size/2+1),
									reinterpret_cast<double*>
									(gpu_data2)+transfers*batch*(size/2+1),
									gpu_data3+transfers*batch*(size/2+1),
									gpu_data4+transfers*batch*(size/2+1));
	
				reduction_general(howmany*(size/2+1),
								reinterpret_cast<double*>(gpu_data1),size/2+1);
				reduction_general(howmany*(size/2+1),
								reinterpret_cast<double*>(gpu_data2),size/2+1);
				reduction_general(howmany*(size/2+1),gpu_data3,size/2+1);
				reduction_general(howmany*(size/2+1),gpu_data4,size/2+1);
			}
			else
			{
				reduction(howmany*(size/2+1),reinterpret_cast<double*>(gpu_data1),size/2+1);
				reduction(howmany*(size/2+1),reinterpret_cast<double*>(gpu_data2),size/2+1);
				reduction(howmany*(size/2+1),gpu_data3,size/2+1);
				reduction(howmany*(size/2+1),gpu_data4,size/2+1);
			}
			add_complex_cuda(size/2+1,
							reinterpret_cast<double*>(gpu_data1),
							reinterpret_cast<double*>(gpu_data2),
							reinterpret_cast<double*>(gpu_accumulate1));
			add_cuda(size/2+1,gpu_data3,gpu_accumulate2);
			add_cuda(size/2+1,gpu_data4,gpu_accumulate3);
			count += 1;
		}
		std::tuple<
		py::array_t<double,py::array::c_style>,
		py::array_t<double,py::array::c_style>,
		py::array_t<std::complex<double>,py::array::c_style>> getResult()
		{
			if(count == 0){throw std::runtime_error("U dumbdumb accumulate first");}
			double* out1;
			double* out2;
			std::complex<double>* out3;
			out1 = (double*) malloc((size/2+1)*sizeof(double));
			out2 = (double*) malloc((size/2+1)*sizeof(double));
			out3 = (std::complex<double>*) malloc((size+2)*sizeof(double));
			cudaMemcpy(out1,gpu_accumulate2,(size/2+1)*sizeof(double),cudaMemcpyDeviceToHost);
			cudaMemcpy(out2,gpu_accumulate3,(size/2+1)*sizeof(double),cudaMemcpyDeviceToHost);
			cudaMemcpy(out3,gpu_accumulate1,(size+2)*sizeof(double),cudaMemcpyDeviceToHost);
			double norm = 1.0/count/howmany;
			for(long long int i=0;i<(size/2+1);i++){out1[i]*=norm;out2[i]*=norm;out3[i]*=norm;}
			py::capsule free_when_done1(out1,free);
			py::capsule free_when_done2(out2,free);
			py::capsule free_when_done3(out3,free);
			return std::make_tuple(
			py::array_t<double,py::array::c_style>
			({size/2+1},{sizeof(double)},out1,free_when_done1),
			py::array_t<double,py::array::c_style>
			({size/2+1},{sizeof(double)},out2,free_when_done2),
			py::array_t<std::complex<double>,py::array::c_style>
			({size/2+1},{2*sizeof(double)},out3,free_when_done3));
		}
		void clear()
		{
			count = 0;
			cudaMemset(gpu_accumulate1,0,(size+2)*sizeof(double));
			cudaMemset(gpu_accumulate2,0,(size/2+1)*sizeof(double));
			cudaMemset(gpu_accumulate3,0,(size/2+1)*sizeof(double));
		}
};
