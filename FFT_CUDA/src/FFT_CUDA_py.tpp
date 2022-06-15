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
		// double cast is not elegant but migt work?
		rFFT_Block_Async_CUDA(out+(i-1)*batch*(size/2+1),
						plan,
						streams[1]);
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
		rFFT_Block_Async_CUDA(out+(transfers-1)*batch*(size/2+1),
						plan,
						streams[1]);
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

	py::capsule free_when_done( out, cuFree );
	return py::array_t<DataType, py::array::c_style> 
	(
		{2*(n-1)},
		{sizeof(DataType)},
		out,
		free_when_done	
	);
}
