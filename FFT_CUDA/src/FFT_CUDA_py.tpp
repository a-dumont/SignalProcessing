template< class DataType >
py::array_t<DataType, py::array::c_style> FFT_CUDA_py(
				py::array_t<DataType, py::array::c_style> py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	DataType* ptr_py_in = (DataType*) buf_in.ptr;
	DataType* out;

	cudaMallocManaged((void**)&out, n*sizeof(DataType));
	
	cudaMemcpy(out,ptr_py_in,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	
	FFT_CUDA<DataType>(n, out);
	cudaDeviceSynchronize();

	py::capsule free_when_done( out, cuFree );
	return py::array_t<DataType, py::array::c_style> 
	(
		{n},
		{sizeof(DataType)},
		out,
		free_when_done	
	);
}

template< class DataType ,class DataType2>
py::array_t<DataType, py::array::c_style> FFT_Block_CUDA_py(
				py::array_t<DataType, py::array::c_style>  py_in, DataType2 N)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	DataType2 n = (DataType2) buf_in.size;
	DataType2 howmany = (DataType2) n/N;

	DataType* ptr_py_in = (DataType*) buf_in.ptr;
	DataType* out;

	cudaMallocManaged((void**)&out, sizeof(DataType)*N*howmany);

	cudaMemcpy(out,ptr_py_in,sizeof(DataType)*N*howmany,cudaMemcpyHostToDevice);

	FFT_Block_CUDA<DataType,DataType2>(n, N, out);
	cudaDeviceSynchronize();

	py::capsule free_when_done( out, cuFree );
	return py::array_t<DataType, py::array::c_style> 
	(
		{N*howmany},
		{sizeof(DataType)},
		out,
		free_when_done	
	);
}

template< class DataType, class DataType2>
py::array_t<DataType, py::array::c_style> FFT_Block_Async_CUDA_py(
				py::array_t<DataType, py::array::c_style> py_in, DataType2 size)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	DataType2 n = buf_in.size;
	DataType2 howmany = n/size;
	DataType2 transfer_size = 1<<28;
	DataType2 data_size = howmany*size*sizeof(DataType);
	DataType2 transfers;
	DataType2 remaining;

	if(data_size/transfer_size == 0)
	{
		return FFT_Block_CUDA_py<DataType,DataType2>(py_in,size);
	}
	else
	{
		transfers = data_size/transfer_size;
		remaining = data_size-transfers*transfer_size;
	}

	DataType* ptr_py_in = (DataType*) buf_in.ptr;
	DataType* out;

	cudaMallocManaged((void**)&out, data_size);

	cufftHandle plan;
	makePlan<DataType, DataType2>(&plan,size,transfer_size/sizeof(DataType)/size);
	
	cudaStream_t streams[2];
	for(int i=0; i<2; i++)
	{
    	cudaStreamCreate(&streams[i]);
	}

	cudaMemcpy(out,ptr_py_in,transfer_size,cudaMemcpyHostToDevice);
	for(long long int i=1;i<transfers;i++)
	{
		cudaMemcpyAsync(out+i*transfer_size/sizeof(DataType),
						ptr_py_in+i*transfer_size/sizeof(DataType),
						transfer_size,
						cudaMemcpyHostToDevice,
						streams[0]);
		FFT_Block_Async_CUDA<DataType>(out+(i-1)*transfer_size/sizeof(DataType),plan,streams[1]);
	}

	if(remaining != 0)
	{
		cudaMemcpyAsync(out+transfers*transfer_size/sizeof(DataType),
					ptr_py_in+transfers*transfer_size/sizeof(DataType),
					remaining,
					cudaMemcpyHostToDevice,streams[0]);
		FFT_Block_Async_CUDA<DataType>(out+(transfers-1)*transfer_size/sizeof(DataType),
						plan,streams[1]);
		cufftDestroy(plan);
		
		cufftHandle plan2;
		makePlan<DataType, DataType2>(&plan2,size,remaining/sizeof(DataType)/size);
		
		FFT_Block_Async_CUDA<DataType>(out+transfers*transfer_size/sizeof(DataType),
						plan2,streams[1]);
		cufftDestroy(plan2);
	}
	else
	{
		FFT_Block_Async_CUDA<DataType>(out+(transfers-1)*transfer_size/sizeof(DataType),
						plan,streams[1]);
		cufftDestroy(plan);
	}

	cudaDeviceSynchronize();
	py::capsule free_when_done( out, cuFree );
	return py::array_t<DataType, py::array::c_style> 
	(
		{size*howmany},
		{sizeof(DataType)},
		out,
		free_when_done	
	);
}

template< class DataType >
py::array_t<DataType, py::array::c_style> iFFT_CUDA_py(
				py::array_t<DataType, py::array::c_style> py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	DataType* ptr_py_in = (DataType*) buf_in.ptr;
	DataType* out;

	cudaMallocManaged((void**)&out, n*sizeof(DataType));
	
	cudaMemcpy(out,ptr_py_in,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	
	iFFT_CUDA<DataType>(n, out);
	cudaDeviceSynchronize();

	py::capsule free_when_done( out, cuFree );
	return py::array_t<DataType, py::array::c_style> 
	(
		{n},
		{sizeof(DataType)},
		out,
		free_when_done	
	);
}

template< class DataType >
py::array_t<std::complex<DataType>, py::array::c_style> rFFT_CUDA_py(
				py::array_t<DataType, py::array::c_style> py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	DataType* ptr_py_in = (DataType*) buf_in.ptr;
	std::complex<DataType>* out;

	cudaMallocManaged((void**)&out, (n/2+1)*sizeof(std::complex<DataType>));
	
	cudaMemcpy(out,ptr_py_in,sizeof(DataType)*n,cudaMemcpyHostToDevice);
	
	rFFT_CUDA<std::complex<DataType>>(n,out);
	cudaDeviceSynchronize();

	py::capsule free_when_done( out, cuFree );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{n/2+1},
		{sizeof(std::complex<DataType>)},
		out,
		free_when_done	
	);
}

template< class DataType ,class DataType2>
py::array_t<std::complex<DataType>, py::array::c_style> rFFT_Block_CUDA_py(
				py::array_t<DataType, py::array::c_style> py_in, DataType2 N)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	DataType2 n = buf_in.size;
	DataType2 howmany = n/N;

	DataType* ptr_py_in = (DataType*) buf_in.ptr;
	std::complex<DataType>* out;

	cudaMallocManaged((void**)&out, sizeof(std::complex<DataType>)*(N/2+1)*howmany);
	cudaMemcpy2D(out,(N+2)*sizeof(DataType),ptr_py_in,N*sizeof(DataType),
					sizeof(DataType)*N,howmany,cudaMemcpyHostToDevice);

	rFFT_Block_CUDA<std::complex<DataType>>(n, N, out);
	cudaDeviceSynchronize();

	py::capsule free_when_done( out, cuFree );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{(N/2+1)*howmany},
		{sizeof(std::complex<DataType>)},
		out,
		free_when_done	
	);
}

template< class DataType ,class DataType2>
py::array_t<std::complex<DataType>, py::array::c_style> rFFT_Block_CUDA2_py(
				py::array_t<DataType, py::array::c_style> py_in, DataType2 size)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	DataType2 n = buf_in.size;
	DataType2 howmany = n/size;
	DataType2 transfer_size = 1<<27;
	DataType2 data_size = howmany*size*sizeof(DataType);
	DataType2 transfers;
	DataType2 remaining;

	if(data_size/transfer_size == 0)
	{
		return rFFT_Block_CUDA_py<DataType,DataType2>(py_in,size);
	}
	else
	{
		transfers = data_size/transfer_size;
		remaining = data_size-transfers*transfer_size;
	}
	DataType2 batch = transfer_size/size/sizeof(DataType);

	DataType* ptr_py_in = (DataType*) buf_in.ptr;
	std::complex<DataType>* out;

	cudaMallocManaged((void**)&out, howmany*(size+2)*sizeof(DataType));

	cufftHandle plan;
	makePlan<DataType,DataType2>(&plan, size, batch);
	
	cudaStream_t streams[2];
	for(int i=0; i<2; i++)
	{
    	cudaStreamCreate(&streams[i]);
	}

	cudaMemcpy2D(out,
				(size+2)*sizeof(DataType),
				ptr_py_in,
				size*sizeof(DataType),
				size*sizeof(DataType),
				batch,
				cudaMemcpyHostToDevice);

	for(int i=1;i<transfers;i++)
	{
		cudaMemcpy2DAsync(out+i*batch*(size/2+1),
						(size+2)*sizeof(DataType),
						ptr_py_in+i*transfer_size/sizeof(DataType),
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						batch,
						cudaMemcpyHostToDevice,
						streams[0]);
		
		rFFT_Block_Async_CUDA(out+(i-1)*batch*(size/2+1),plan,streams[1]);
	}
	if(remaining != 0)
	{
		cudaMemcpy2DAsync(out+transfers*batch*(size/2+1),
						(size+2)*sizeof(DataType),
						ptr_py_in+transfers*transfer_size/sizeof(DataType),
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						remaining/sizeof(DataType)/size,
						cudaMemcpyHostToDevice,
						streams[0]);
		rFFT_Block_Async_CUDA(out+(transfers-1)*batch*(size/2+1),plan,streams[1]);

		cufftDestroy(plan);
		cufftHandle plan2;
		makePlan<DataType, DataType2>(&plan2,size,remaining/sizeof(DataType)/size);

		rFFT_Block_Async_CUDA(out+transfers*batch*(size/2+1),
						plan,
						streams[1]);
		cufftDestroy(plan2);
	}
	else
	{
		cudaMemPrefetchAsync(out, transfer_size, cudaCpuDeviceId, streams[0]);
		rFFT_Block_Async_CUDA(out+(transfers-1)*batch*(size/2+1),
						plan,
						streams[1]);
		cufftDestroy(plan);
	}

	cudaDeviceSynchronize();
	py::capsule free_when_done( out, cuFree );
	return py::array_t<std::complex<DataType>, py::array::c_style> 
	(
		{(size/2+1)*howmany},
		{sizeof(std::complex<DataType>)},
		out,
		free_when_done	
	);
}

template< class DataType >
py::array_t<DataType, py::array::c_style> irFFT_CUDA_py(
			   py::array_t<std::complex<DataType>, py::array::c_style> py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	std::complex<DataType>* ptr_py_in = (std::complex<DataType>*) buf_in.ptr;
	DataType* out;

	cudaMallocManaged((void**)&out, 2*n*sizeof(DataType));
	
	cudaMemcpy(out,ptr_py_in,sizeof(std::complex<DataType>)*n,cudaMemcpyHostToDevice);
	
	irFFT_CUDA<DataType>(2*(n-1), out);
	cudaDeviceSynchronize();

	py::capsule free_when_done( out, cuFree );
	return py::array_t<DataType, py::array::c_style> 
	(
		{2*(n-1)},
		{sizeof(DataType)},
		out,
		free_when_done	
	);
}

template< class DataType ,class DataType2>
py::array_t<std::complex<DataType2>, py::array::c_style> digitizer_FFT_Block_Async_CUDA_py(
				py::array_t<DataType, py::array::c_style> py_in, 
				long long int size,
				DataType2 conv,
				long long int offset)
{
	py::buffer_info buf_in = py_in.request();
	DataType* ptr_py_in = (DataType*) buf_in.ptr;

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = buf_in.size;
	long long int howmany = n/size;
	long long int transfer_size = 1<<19;
	long long int data_size = howmany*size*sizeof(DataType);
	long long int transfers;
	long long int remaining;

	if(data_size/transfer_size == 0)
	{
		transfers = 1;
		transfer_size = data_size;
		remaining = 0;
	}
	else
	{
		transfers = data_size/transfer_size;
		remaining = data_size-transfers*transfer_size;
	}
	long long int batch = transfer_size/size/sizeof(DataType);
	long long int N = batch*size;

	std::complex<DataType2>* out;
	cudaMallocManaged((void**)&out, size*howmany*sizeof(std::complex<DataType2>));

	DataType* gpu;
	cudaMallocManaged((void**)&gpu, size*howmany*sizeof(DataType));
	
	cudaStream_t streams[2];
	for(int i=0; i<2; i++)
	{
    	cudaStreamCreate(&streams[i]);
	}

	cudaMemcpyAsync(gpu,ptr_py_in,transfer_size,cudaMemcpyHostToDevice,streams[0]);

	cufftHandle plan;
	makePlan<std::complex<DataType2>, long long int>(&plan,size,batch);

	for(long long int i=1;i<transfers;i++)
	{
		cudaMemcpyAsync(gpu+i*N,ptr_py_in+i*N,transfer_size,cudaMemcpyHostToDevice,streams[0]);
		convertComplex<DataType,DataType2>(N,gpu+(i-1)*N,
						out+(i-1)*N,conv,offset,streams[1]);
		FFT_Block_Async_CUDA<std::complex<DataType2>>(out+(i-1)*N,plan,streams[1]);
	}

	if(remaining != 0)
	{
		cudaMemcpyAsync(gpu+transfers*N,ptr_py_in+transfers*N,
						remaining,cudaMemcpyHostToDevice,streams[0]);
		convertComplex<DataType,DataType2>(N,gpu+(transfers-1)*N,
						out+(transfers-1)*N,conv,offset,streams[1]);
		FFT_Block_Async_CUDA<std::complex<DataType2>>(out+(transfers-1)*N,plan,streams[1]);
		cufftDestroy(plan);
		
		cufftHandle plan2;
		makePlan<DataType2, long long int>(&plan2,size,remaining/sizeof(DataType)/size);
		
		convertComplex<DataType,DataType2>(remaining/sizeof(DataType),gpu+transfers*N,
						out+transfers*N,conv,offset,streams[1]);
		FFT_Block_Async_CUDA<std::complex<DataType2>>(out+transfers*N,plan2,streams[1]);
		cufftDestroy(plan2);
	}
	else
	{
		convertComplex<DataType,DataType2>(N,gpu+(transfers-1)*N,
						out+(transfers-1)*N,conv,offset,streams[1]);
		FFT_Block_Async_CUDA<std::complex<DataType2>>(out+(transfers-1)*N,plan,streams[1]);
		cufftDestroy(plan);
	}
	cudaFree(gpu);

	cudaDeviceSynchronize();
	py::capsule free_when_done( out, cuFree );
	return py::array_t<std::complex<DataType2>, py::array::c_style> 
	(
		{howmany*size},
		{sizeof(std::complex<DataType2>)},
		out,
		free_when_done	
	);
}

template< class DataType ,class DataType2>
py::array_t<std::complex<DataType2>, py::array::c_style> digitizer_rFFT_Block_Async_CUDA_py(
				py::array_t<DataType, py::array::c_style> py_in, 
				long long int size,
				DataType2 conv,
				long long int offset)
{
	py::buffer_info buf_in = py_in.request();
	DataType* ptr_py_in = (DataType*) buf_in.ptr;

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = buf_in.size;
	long long int howmany = n/size;
	long long int transfer_size = 1<<19;
	long long int data_size = howmany*size*sizeof(DataType);
	long long int transfers;
	long long int remaining;

	if(data_size/transfer_size == 0)
	{
		transfers = 1;
		transfer_size = data_size;
		remaining = 0;
	}
	else
	{
		transfers = data_size/transfer_size;
		remaining = data_size-transfers*transfer_size;
	}
	long long int batch = transfer_size/size/sizeof(DataType);
	long long int N = batch*size;

	std::complex<DataType2>* out;
	cudaMallocManaged((void**)&out, (size+2)*howmany*sizeof(DataType2));

	DataType* gpu;
	cudaMallocManaged((void**)&gpu, (size+2)*howmany*sizeof(DataType));

	cudaStream_t streams[2];
	for(int i=0; i<2; i++)
	{
    	cudaStreamCreate(&streams[i]);
	}

	cudaMemcpy2D(gpu,(size+2)*sizeof(DataType),ptr_py_in,size*sizeof(DataType),
				size*sizeof(DataType),batch,cudaMemcpyHostToDevice);

	cufftHandle plan;
	makePlan<DataType2, long long int>(&plan,size,batch);

	for(long long int i=1;i<transfers;i++)
	{
		cudaMemcpy2DAsync(gpu+i*batch*(size+2),
						(size+2)*sizeof(DataType),
						ptr_py_in+i*N,
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						batch,
						cudaMemcpyHostToDevice,
						streams[0]);

		convert<DataType,DataType2>(batch*(size+2),
						gpu+(i-1)*batch*(size+2),
						reinterpret_cast<DataType2*>(out+(i-1)*batch*(size/2+1)),
						conv,offset,streams[1]);
		rFFT_Block_Async_CUDA(out+(i-1)*batch*(size/2+1),plan,streams[1]);
	}

	if(remaining != 0)
	{
		cudaMemcpy2DAsync(gpu+transfers*batch*(size+2),
						(size+2)*sizeof(DataType),
						ptr_py_in+transfers*N,
						sizeof(DataType)*size,
						sizeof(DataType)*size,
						remaining/sizeof(DataType)/size,
						cudaMemcpyHostToDevice,
						streams[0]);
		convert<DataType,DataType2>(batch*(size+2),
						gpu+(transfers-1)*batch*(size+2),
						reinterpret_cast<DataType2*>(out+(transfers-1)*batch*(size/2+1)),
						conv,offset,streams[1]);
		rFFT_Block_Async_CUDA(out+(transfers-1)*batch*(size/2+1),plan,streams[1]);
		cufftDestroy(plan);
		
		cufftHandle plan2;
		makePlan<DataType2, long long int>(&plan2,size,remaining/sizeof(DataType)/size);
		
		convert<DataType,DataType2>(remaining/sizeof(DataType),
						gpu+transfers*N,
						reinterpret_cast<DataType2*>(out+transfers*batch*(size/2+1)),
						conv,offset,streams[1]);
		rFFT_Block_Async_CUDA(out+transfers*batch*(size/2+1),plan,streams[1]);
		cufftDestroy(plan2);
	}
	else
	{
		convert<DataType,DataType2>(batch*(size+2),
						gpu+(transfers-1)*batch*(size+2),
						reinterpret_cast<DataType2*>(out+(transfers-1)*batch*(size/2+1)),
						conv,offset,streams[1]);
		rFFT_Block_Async_CUDA(out+(transfers-1)*batch*(size/2+1),plan,streams[1]);
		cufftDestroy(plan);
	}
	cudaFree(gpu);

	cudaDeviceSynchronize();
	py::capsule free_when_done( out, cuFree );
	return py::array_t<std::complex<DataType2>, py::array::c_style> 
	(
		{howmany*(size/2+1)},
		{sizeof(std::complex<DataType2>)},
		out,
		free_when_done	
	);
}
