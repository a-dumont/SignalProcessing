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
DataType histogram_vectorial_average_py(DataType py_in, int row, int col)
{
	py::buffer_info buf_hist = py_in.request();
	if(buf_hist.ndim != 2)
	{
		throw std::runtime_error("U dumbdumb histogram must be 2D.");
	}
	if(py_in.shape(0) != py_in.shape(1))
	{
		throw std::runtime_error("U dumbdumb histogram must be square.");
	}
	int nbins = py_in.shape(0);
	double* out = (double*) malloc(2*sizeof(double));
	out[0] = 0;
	out[1] = 0;
	double* hist = (double*) buf_hist.ptr;
	histogram_vectorial_average(nbins,hist,out,row,col);

	py::capsule free_when_done( out, free );
	return py::array_t<double, py::array::c_style> 
	(
		{2},
		{sizeof(double)},
		out,
		free_when_done	
	);
}

template<class DataType>
DataType inverse_probability2D_py(DataType py_in1, DataType py_in2)
{
	py::buffer_info buf_gamma = py_in1.request();
	py::buffer_info buf_density = py_in2.request();
	if(buf_density.ndim != 2)
	{
		throw std::runtime_error("U dumbdumb histogram must be 2D.");
	}
	if(buf_gamma.size != buf_density.size*buf_density.size)
	{
		throw std::runtime_error("U dumbdumb dimensions mismatch");
	}
	if(py_in2.shape(0) != py_in2.shape(1))
	{
		throw std::runtime_error("U dumbdumb histogram must be square.");
	}
	int nbins = py_in2.shape(0);
	double* out = (double*) malloc(nbins*nbins*nbins*nbins*sizeof(double));
	double* ptr_gamma = (double*) buf_gamma.ptr;
	double* ptr_density = (double*) buf_density.ptr;
	inverse_probability2D(nbins,ptr_gamma,out,ptr_density);
	py::capsule free_when_done( out, free );
	return py::array_t<double, py::array::c_style> 
	(
		{(nbins*nbins),nbins,nbins},
		{(nbins*nbins)*sizeof(double),nbins*sizeof(double),sizeof(double)},
		out,
		free_when_done	
	);
}
