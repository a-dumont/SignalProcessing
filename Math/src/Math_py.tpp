template<class DataType>
DataType gradient_py(DataType py_in1, DataType py_in2)
{
	py::buffer_info buf_x = py_in1.request();
	py::buffer_info buf_t = py_in2.request();

	if ((buf_x.ndim != 1) | (buf_t.ndim != 1))
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	if (buf_x.size != buf_t.size)
	{
		throw std::runtime_error("U dumbdumb size must be same.");
	}	

	int n = buf_x.size;

	double* x = (double*) buf_x.ptr;
	double* t = (double*) buf_t.ptr;
	double* out = (double*) malloc(sizeof(double)*n);

	gradient<double>(n, x, t, out);

	py::capsule free_when_done( out, free );
	return py::array_t<double, py::array::c_style> 
	(
		{n},
		{sizeof(double)},
		out,
		free_when_done	
	);
}

template<class DataType, class DataType2>
DataType gradient_py(DataType py_in1, DataType2 dt)
{
	py::buffer_info buf_x = py_in1.request();

	if (buf_x.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_x.size;

	double* x = (double*) buf_x.ptr;
	double* out = (double*) malloc(sizeof(double)*n);

	gradient<double,double>(n, x, dt, out);

	py::capsule free_when_done( out, free );
	return py::array_t<double, py::array::c_style> 
	(
		{n},
		{sizeof(double)},
		out,
		free_when_done	
	);
}

template<class DataType>
DataType rolling_average_py(DataType py_in, int size)
{
	py::buffer_info buf_x = py_in.request();

	if (buf_x.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_x.size;

	double* in = (double*) buf_x.ptr;
	double* out = (double*) malloc(sizeof(double)*(n-size+1));
	rolling_average(n, in, out, size);

	py::capsule free_when_done( out, free );
	return py::array_t<double, py::array::c_style> 
	(
		{n-size+1},
		{sizeof(double)},
		out,
		free_when_done	
	);
}

template<class DataType>
DataType finite_difference_coefficients_py(int M, int N)
{
	double* coeff = finite_difference_coefficients(M,N);
	N = 2*N;
	int n = N+1;
	py::capsule free_when_done( coeff, free );
	return py::array_t<double, py::array::c_style> 
	(
		{(N+1)},
		{sizeof(double)},
		coeff+(M*n*n+N*n),
		free_when_done	
	);
}

template<class DataType, class DataType2>
DataType nth_order_gradient_py(DataType py_in, DataType2 dt,int M, int N)
{
	py::buffer_info buf_x = py_in.request();

	if (buf_x.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_x.size;

	double* x = (double*) buf_x.ptr;
	double* out = (double*) malloc(sizeof(double)*(n-2*N));
	std::memset(out,0,sizeof(double)*(n-2*N));

	nth_order_gradient(n, x, dt, out, M, N);

	py::capsule free_when_done( out, free );
	return py::array_t<double, py::array::c_style> 
	(
		{n-N-N},
		{sizeof(double)},
		out,
		free_when_done	
	);
}

template<class DataType>
np_int continuous_max_py(py::array_t<DataType,py::array::c_style> py_in)
{
	if (py_in.request().ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	
	long int* out = (long int*) malloc(py_in.request().size*sizeof(long int));
	continuous_max(out,(DataType*) py_in.request().ptr,py_in.request().size);
	py::capsule free_when_done( out, free );
	return py::array_t<long int, py::array::c_style> 
	(
		{py_in.request().size},
		{sizeof(long int)},
		out,
		free_when_done	
		);
}

template<class DataType>
np_int continuous_min_py(py::array_t<DataType,py::array::c_style> py_in)
{
	if (py_in.request().ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	
	long int* out = (long int*) malloc(py_in.request().size*sizeof(long int));
	continuous_min(out,(DataType*) py_in.request().ptr,py_in.request().size);
	py::capsule free_when_done( out, free );
	return py::array_t<long int, py::array::c_style> 
	(
		{py_in.request().size},
		{sizeof(long int)},
		out,
		free_when_done	
		);
	}
	
template<class DataType>
DataType sum_py(py::array_t<DataType,py::array::c_style> py_in1)
{
	py::buffer_info buf1 = py_in.request();
	return sum<DataType>(buf1.ptr,buf1.size);
}

template<class DataType>
DataType mean_py(py::array_t<DataType,py::array::c_style> py_in1)
{
	py::buffer_info buf1 = py_in.request();
	return mean(buf1.ptr,buf1.size);
}

template<class DataType>
DataType variance_py(py::array_t<DataType,py::array::c_style> py_in1)
{
	py::buffer_info buf1 = py_in.request();
	return variance(buf1.ptr,buf1.size);
}

template<class DataType>
DataType poisson_py(py::array_t<DataType,py::array::c_style> py_in1)
{
	py::buffer_info buf1 = py_in.request();
	return poisson(buf1.ptr,buf1.size);
}

template<class DataType, class DataType2>
DataType product_py(py::array_t<DataType,py::array::c_style> py_in1,py::array_t<DataType,py::array::c_style> py_in2)
{
	py::buffer_info buf1 = py_in.request();
	py::buffer_info buf2 = py_in2.request();
	if ((buf1.ndim != 1) | (buf2.ndim != 1))
	{
		throw std::runtime_error("U dumbdumb dimension must be same.");
	}	
	if (buf1.size != buf2.size)
	{
		throw std::runtime_error("U dumbdumb size must be same.");
	}
	DataType* out = (DataType*) malloc(sizeof(DataType)*buf1.size);
	product(buf1.ptr,buf2.ptr,out,buf1.size);
	ndim = py_in1.ndim();
	std::vector<int> shape;
	std::vector<int> strides;
	for(int i=0;i<ndim;i++)
	{
		shape.push_back(py_in1.shape(i));
		shape.push_back((int) py_in1.strides(i)/sizeof(DataType));
	}
	py::capsule free_when_done( out, free );
	return py::array_t<long int, py::array::c_style> 
	(
		shape,
		strides,
		out,
		free_when_done	
		);
}

template<class DataType, class DataType2>
DataType sum_py(py::array_t<DataType,py::array::c_style> py_in1,py::array_t<DataType,py::array::c_style> py_in2)
{
	py::buffer_info buf1 = py_in.request();
	py::buffer_info buf2 = py_in2.request();
	if ((buf1.ndim != 1) | (buf2.ndim != 1))
	{
		throw std::runtime_error("U dumbdumb dimension must be same.");
	}	
	if (buf1.size != buf2.size)
	{
		throw std::runtime_error("U dumbdumb size must be same.");
	}
	DataType* out = (DataType*) malloc(sizeof(DataType)*buf1.size);
	sum(buf1.ptr,buf2.ptr,out,buf1.size);
	ndim = py_in1.ndim();
	std::vector<int> shape;
	std::vector<int> strides;
	for(int i=0;i<ndim;i++)
	{
		shape.push_back(py_in1.shape(i));
		shape.push_back((int) py_in1.strides(i)/sizeof(DataType));
	}
	py::capsule free_when_done( out, free );
	return py::array_t<long int, py::array::c_style> 
	(
		shape,
		strides,
		out,
		free_when_done	
		);
}

template<class DataType, class DataType2>
DataType difference_py(py::array_t<DataType,py::array::c_style> py_in1,py::array_t<DataType,py::array::c_style> py_in2)
{
	py::buffer_info buf1 = py_in.request();
	py::buffer_info buf2 = py_in2.request();
	if ((buf1.ndim != 1) | (buf2.ndim != 1))
	{
		throw std::runtime_error("U dumbdumb dimension must be same.");
	}	
	if (buf1.size != buf2.size)
	{
		throw std::runtime_error("U dumbdumb size must be same.");
	}
	DataType* out = (DataType*) malloc(sizeof(DataType)*buf1.size);
	difference(buf1.ptr,buf2.ptr,out,buf1.size);
	ndim = py_in1.ndim();
	std::vector<int> shape;
	std::vector<int> strides;
	for(int i=0;i<ndim;i++)
	{
		shape.push_back(py_in1.shape(i));
		shape.push_back((int) py_in1.strides(i)/sizeof(DataType));
	}
	py::capsule free_when_done( out, free );
	return py::array_t<long int, py::array::c_style> 
	(
		shape,
		strides,
		out,
		free_when_done	
		);
}

template<class DataType, class DataType2>
DataType division_py(py::array_t<DataType,py::array::c_style> py_in1,py::array_t<DataType,py::array::c_style> py_in2)
{
	py::buffer_info buf1 = py_in.request();
	py::buffer_info buf2 = py_in2.request();
	if ((buf1.ndim != 1) | (buf2.ndim != 1))
	{
		throw std::runtime_error("U dumbdumb dimension must be same.");
	}	
	if (buf1.size != buf2.size)
	{
		throw std::runtime_error("U dumbdumb size must be same.");
	}
	DataType* out = (DataType*) malloc(sizeof(DataType)*buf1.size);
	division(buf1.ptr,buf2.ptr,out,buf1.size);
	ndim = py_in1.ndim();
	std::vector<int> shape;
	std::vector<int> strides;
	for(int i=0;i<ndim;i++)
	{
		shape.push_back(py_in1.shape(i));
		shape.push_back((int) py_in1.strides(i)/sizeof(DataType));
	}
	py::capsule free_when_done( out, free );
	return py::array_t<long int, py::array::c_style> 
	(
		shape,
		strides,
		out,
		free_when_done	
		);
}