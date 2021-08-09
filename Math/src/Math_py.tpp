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

