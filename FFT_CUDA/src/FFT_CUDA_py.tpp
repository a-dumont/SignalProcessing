template< class DataType >
DataType FFT_CUDA_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	dbl_complex* ptr_py_in = (dbl_complex*) buf_in.ptr;
	dbl_complex* out;

	cudaMallocManaged((void**)&out, n*sizeof(dbl_complex));
	
	cudaMemcpy(out,ptr_py_in,sizeof(dbl_complex)*n,cudaMemcpyHostToDevice);
	
	FFT_CUDA(n,out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{n},
		{sizeof(dbl_complex)},
		out,
		free_when_done	
	);
}

template< class DataType >
DataType fFFT_CUDA_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	flt_complex* ptr_py_in = (flt_complex*) buf_in.ptr;
	flt_complex* out;

	cudaMallocManaged((void**)&out, n*sizeof(flt_complex));
	
	cudaMemcpy(out,ptr_py_in,sizeof(flt_complex)*n,cudaMemcpyHostToDevice);
	
	fFFT_CUDA(n,out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<flt_complex, py::array::c_style> 
	(
		{n},
		{sizeof(flt_complex)},
		out,
		free_when_done	
	);
}

template< class DataType ,class DataType2>
DataType FFT_Block_CUDA_py(DataType py_in,long long int N)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = (long long int) buf_in.size;
	long long int howmany = (long long int) n/N;

	dbl_complex* ptr_py_in = (dbl_complex*) buf_in.ptr;
	dbl_complex* out;

	cudaMallocManaged((void**)&out, sizeof(dbl_complex)*N*howmany);

	cudaMemcpy(out,ptr_py_in,sizeof(dbl_complex)*N*howmany,cudaMemcpyHostToDevice);

	FFT_Block_CUDA(n, N, out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{N*howmany},
		{sizeof(dbl_complex)},
		out,
		free_when_done	
	);
}

template< class DataType ,class DataType2>
DataType FFT_Block_CUDA2_py(DataType py_in, int size)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	int howmany = n/size;
	long long int transfer_size = 1<<28;
	long long int data_size = howmany*size*sizeof(dbl_complex);
	int transfers;
	int remaining;

	if(data_size/transfer_size == 0)
	{
		return FFT_Block_CUDA_py<np_complex,long long int>(py_in,size);
	}
	else
	{
		transfers = data_size/transfer_size;
		remaining = data_size-transfers*transfer_size;
	}

	dbl_complex* ptr_py_in = (dbl_complex*) buf_in.ptr;
	dbl_complex* out;

	cudaMallocManaged((void**)&out, data_size);

	cufftHandle plan;
	if (cufftPlan1d(&plan,size,CUFFT_Z2Z, transfer_size/sizeof(dbl_complex)/size) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	
	cudaStream_t streams[2];
	for(int i=0; i<2; i++)
	{
    	cudaStreamCreate(&streams[i]);
	}

	cudaMemcpy(out,ptr_py_in,transfer_size,cudaMemcpyHostToDevice);
	for(int i=1;i<transfers;i++)
	{
		cudaMemcpyAsync(out+i*transfer_size/sizeof(dbl_complex),
						ptr_py_in+i*transfer_size/sizeof(dbl_complex),
						transfer_size,
						cudaMemcpyHostToDevice,
						streams[0]);
		FFT_Block2_CUDA(out+(i-1)*transfer_size/sizeof(dbl_complex),plan,streams[1]);
	}

	if(remaining != 0)
	{
		cudaMemcpyAsync(out+transfers*transfer_size/sizeof(dbl_complex),
					ptr_py_in+transfers*transfer_size/sizeof(dbl_complex),
					remaining,
					cudaMemcpyHostToDevice,streams[0]);
		FFT_Block2_CUDA(out+(transfers-1)*transfer_size/sizeof(dbl_complex),plan,streams[1]);
		cufftDestroy(plan);
		cufftHandle plan2;
		if(cufftPlan1d(&plan2,size,CUFFT_Z2Z,remaining/sizeof(dbl_complex)/size)!=CUFFT_SUCCESS)
		{
			throw std::runtime_error("CUFFT error: Plan creation failed");
		}
		FFT_Block2_CUDA(out+transfers*transfer_size/sizeof(dbl_complex),plan,streams[1]);
		cufftDestroy(plan2);
	}
	else
	{
		FFT_Block2_CUDA(out+(transfers-1)*transfer_size/sizeof(dbl_complex)-1,plan,streams[1]);
		cufftDestroy(plan);
	}

	cudaDeviceSynchronize();
	py::capsule free_when_done( out, cuFree );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{size*howmany},
		{sizeof(dbl_complex)},
		out,
		free_when_done	
	);
}

template< class DataType ,class DataType2>
DataType fFFT_Block_CUDA_py(DataType py_in,long long int N)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = (long long int) buf_in.size;
	long long int howmany = (long long int) n/N;

	flt_complex* ptr_py_in = (flt_complex*) buf_in.ptr;
	flt_complex* out;

	cudaMallocManaged((void**)&out, sizeof(flt_complex)*N*howmany);

	cudaMemcpy(out,ptr_py_in,sizeof(flt_complex)*N*howmany,cudaMemcpyHostToDevice);

	fFFT_Block_CUDA(n, N, out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<flt_complex, py::array::c_style> 
	(
		{N*howmany},
		{sizeof(flt_complex)},
		out,
		free_when_done	
	);
}

template< class DataType >
DataType iFFT_CUDA_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	dbl_complex* ptr_py_in = (dbl_complex*) buf_in.ptr;
	dbl_complex* out;

	cudaMallocManaged((void**)&out, n*sizeof(dbl_complex));
	
	cudaMemcpy(out,ptr_py_in,sizeof(dbl_complex)*n,cudaMemcpyHostToDevice);
	
	iFFT_CUDA(n,out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{n},
		{sizeof(dbl_complex)},
		out,
		free_when_done	
	);
}

template< class DataType >
DataType ifFFT_CUDA_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	flt_complex* ptr_py_in = (flt_complex*) buf_in.ptr;
	flt_complex* out;

	cudaMallocManaged((void**)&out, n*sizeof(dbl_complex));
	
	cudaMemcpy(out,ptr_py_in,sizeof(flt_complex)*n,cudaMemcpyHostToDevice);
	
	ifFFT_CUDA(n,out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<flt_complex, py::array::c_style> 
	(
		{n},
		{sizeof(flt_complex)},
		out,
		free_when_done	
	);
}


template< class DataType >
np_complex rFFT_CUDA_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	double* ptr_py_in = (double*) buf_in.ptr;
	dbl_complex* out;

	cudaMallocManaged((void**)&out, (n/2+1)*sizeof(dbl_complex));
	
	cudaMemcpy(out,ptr_py_in,sizeof(double)*n,cudaMemcpyHostToDevice);
	
	rFFT_CUDA(n,out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{n/2+1},
		{sizeof(dbl_complex)},
		out,
		free_when_done	
	);
}

template< class DataType >
np_fcomplex rfFFT_CUDA_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	float* ptr_py_in = (float*) buf_in.ptr;
	flt_complex* out;

	cudaMallocManaged((void**)&out, (n/2+1)*sizeof(flt_complex));
	
	cudaMemcpy(out,ptr_py_in,sizeof(float)*n,cudaMemcpyHostToDevice);
	
	rfFFT_CUDA(n,out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<flt_complex, py::array::c_style> 
	(
		{n/2+1},
		{sizeof(flt_complex)},
		out,
		free_when_done	
	);
}


template< class DataType ,class DataType2>
np_complex rFFT_Block_CUDA_py(DataType py_in,int N)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = (long long int) buf_in.size;
	long long int howmany = (long long int) n/N;

	dbl_complex* ptr_py_in = (dbl_complex*) buf_in.ptr;
	dbl_complex* out;

	cudaMallocManaged((void**)&out, sizeof(dbl_complex)*(N/2+1)*howmany);
	cudaMemcpy2D(out,(N+2)*sizeof(double),ptr_py_in,N*sizeof(double),
					sizeof(double)*N,howmany,cudaMemcpyHostToDevice);

	rFFT_Block_CUDA(n, N, out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{(N/2+1)*howmany},
		{sizeof(dbl_complex)},
		out,
		free_when_done	
	);
}

template< class DataType ,class DataType2>
np_complex rFFT_Block_CUDA2_py(DataType py_in, int size)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	int howmany = n/size;
	long long int transfer_size = 1<<28;
	long long int data_size = howmany*size*sizeof(double);
	int transfers;
	int remaining;

	if(data_size/transfer_size == 0)
	{
		return rFFT_Block_CUDA_py<np_double,long long int>(py_in,size);
	}
	else
	{
		transfers = data_size/transfer_size;
		remaining = data_size-transfers*transfer_size;
	}
	int batch = transfer_size/size/sizeof(double);

	double* ptr_py_in = (double*) buf_in.ptr;
	dbl_complex* out;

	cudaMallocManaged((void**)&out, howmany*(size+2)*sizeof(double));

	cufftHandle plan;
	if (cufftPlan1d(&plan,size,CUFFT_D2Z, batch) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	
	cudaStream_t streams[2];
	for(int i=0; i<2; i++)
	{
    	cudaStreamCreate(&streams[i]);
	}

	cudaMemcpy2D(out,
				(size+2)*sizeof(double),
				ptr_py_in,
				size*sizeof(double),
				size*sizeof(double),
				batch,
				cudaMemcpyHostToDevice);

	for(int i=1;i<transfers;i++)
	{
		cudaMemcpy2DAsync(out+i*batch*(size/2+1),
						(size+2)*sizeof(double),
						ptr_py_in+i*transfer_size/sizeof(double),
						sizeof(double)*size,
						sizeof(double)*size,
						batch,
						cudaMemcpyHostToDevice,
						streams[0]);
		// double cast is not elegant but migt work?
		rFFT_Block2_CUDA((double*)(out+(i-1)*batch*(size/2+1)),
						out+(i-1)*batch*(size/2+1),
						plan,
						streams[1]);
	}
	if(remaining != 0)
	{
		cudaMemcpy2DAsync(out+transfers*batch*(size/2+1),
						(size+2)*sizeof(double),
						ptr_py_in+transfers*transfer_size/sizeof(double),
						sizeof(double)*size,
						sizeof(double)*size,
						remaining/sizeof(double)/size,
						cudaMemcpyHostToDevice,
						streams[0]);
		rFFT_Block2_CUDA((double*)(out+(transfers-1)*batch*(size/2+1)),
						out+(transfers-1)*batch*(size/2+1),
						plan,
						streams[1]);
		cufftDestroy(plan);
		cufftHandle plan2;
		if(cufftPlan1d(&plan2,size,CUFFT_D2Z,remaining/sizeof(double)/size)!=CUFFT_SUCCESS)
		{
			throw std::runtime_error("CUFFT error: Plan creation failed");
		}
		rFFT_Block2_CUDA((double*)(out+transfers*batch*(size/2+1)),
						(dbl_complex*)(out+transfers*batch*(size/2+1)),
						plan,
						streams[1]);
		cufftDestroy(plan2);
	}
	else
	{
		cudaMemPrefetchAsync(out, transfer_size, cudaCpuDeviceId, streams[0]);
		rFFT_Block2_CUDA((double*)(out+(transfers-1)*batch*(size/2+1)),
						out+(transfers-1)*batch*(size/2+1),
						plan,
						streams[1]);
		cufftDestroy(plan);
	}

	cudaDeviceSynchronize();
	py::capsule free_when_done( out, cuFree );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{(size/2+1)*howmany},
		{sizeof(dbl_complex)},
		out,
		free_when_done	
	);
}


template< class DataType ,class DataType2>
np_fcomplex rfFFT_Block_CUDA_py(DataType py_in,int N)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = (long long int) buf_in.size;
	long long int howmany = (long long int) n/N;

	flt_complex* ptr_py_in = (flt_complex*) buf_in.ptr;
	flt_complex* out;

	cudaMallocManaged((void**)&out, sizeof(flt_complex)*(N/2+1)*howmany);
	cudaMemcpy2D(out,(N+2)*sizeof(float),ptr_py_in,N*sizeof(float),
					sizeof(float)*N,howmany,cudaMemcpyHostToDevice);

	rfFFT_Block_CUDA(n, N, out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<flt_complex, py::array::c_style> 
	(
		{(N/2+1)*howmany},
		{sizeof(flt_complex)},
		out,
		free_when_done	
	);
}


template< class DataType >
np_double irFFT_CUDA_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	dbl_complex* ptr_py_in = (dbl_complex*) buf_in.ptr;
	double* out;

	cudaMallocManaged((void**)&out, 2*n*sizeof(double));
	
	cudaMemcpy(out,ptr_py_in,sizeof(dbl_complex)*n,cudaMemcpyHostToDevice);
	
	irFFT_CUDA(2*(n-1), out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<double, py::array::c_style> 
	(
		{2*(n-1)},
		{sizeof(double)},
		out,
		free_when_done	
	);
}

template< class DataType >
np_float irfFFT_CUDA_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	
	flt_complex* ptr_py_in = (flt_complex*) buf_in.ptr;
	float* out;

	cudaMallocManaged((void**)&out, 2*n*sizeof(float));
	
	cudaMemcpy(out,ptr_py_in,sizeof(flt_complex)*n,cudaMemcpyHostToDevice);
	
	irfFFT_CUDA(2*(n-1), out, out);

	py::capsule free_when_done( out, cuFree );
	return py::array_t<float, py::array::c_style> 
	(
		{2*(n-1)},
		{sizeof(float)},
		out,
		free_when_done	
	);
}
