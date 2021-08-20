template <class Datatype, class Datatype2>
std::tuple<np_int,np_double> Histogram_py(Datatype py_in, int nbins)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;

	double* data = (double*) buf_in.ptr;
	long* hist = (long*) malloc(sizeof(long)*nbins);
	double* edges = GetEdges(data, n, nbins);

	Histogram(hist, edges, data, n, nbins);

	py::capsule free_when_done1( hist, free );
	py::capsule free_when_done2( edges, free );

	np_int hist_py = np_int(
					{nbins},
					{sizeof(long)},
					hist,
					free_when_done1);

	Datatype edges_py = np_double(
					{nbins+1},
					{sizeof(long)},
					edges,
					free_when_done2);	

	std::tuple<np_int,np_double> result = std::make_tuple(hist_py, edges_py);

	return result; 
}

template <class Datatype, class Datatype2>
std::tuple<np_double,np_double> Histogram_Density_py(Datatype py_in, int nbins, bool density)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	if ( density == false )
	{
			std::tuple<np_int,np_double> out = Histogram_py<np_double,int>(py_in,nbins);
			py::buffer_info buf_in = std::get<0>(out).request();
			long* hist_long = (long*) buf_in.ptr;
			double* hist = (double*) malloc(sizeof(double)*buf_in.size);
			for(int i=0; i<buf_in.size;i++)
			{
				hist[i] = (double)hist_long[i];
			}
			
			py::capsule free_when_done1( hist, free );
			np_double hist_py = np_double(
						{nbins},
						{sizeof(double)},
						hist,
						free_when_done1);

			std::tuple<np_double,np_double> result = std::make_tuple(hist_py, std::get<1>(out));

			return result;
	}
	else if ( density == true )
	{
		int n = buf_in.size;

		double* data = (double*) buf_in.ptr;
		double* hist = (double*) malloc(sizeof(double)*nbins);
		double* edges = GetEdges(data, n, nbins);

		Histogram_Density(hist, edges, data, n, nbins);

		py::capsule free_when_done1( hist, free );
		py::capsule free_when_done2( edges, free );

		np_double hist_py = np_double(
						{nbins},
						{sizeof(double)},
						hist,
						free_when_done1);

		Datatype edges_py = np_double(
						{nbins+1},
						{sizeof(double)},
						edges,
						free_when_done2);	

		std::tuple<np_double,np_double> result = std::make_tuple(hist_py, edges_py);

		return result; 
	}
	else
	{
		throw std::runtime_error("U dumbdumb density must be bool.");
	}
}

template <class Datatype>
np_int Histogram_py(Datatype py_in, Datatype edges_py)
{
	py::buffer_info buf_in = py_in.request();
	py::buffer_info buf_edges = edges_py.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;
	int nbins = buf_edges.size-1;

	double* data = (double*) buf_in.ptr;
	long* hist = (long*) malloc(sizeof(long)*nbins);
	double* edges = (double*) buf_edges.ptr;

	Histogram(hist, edges, data, n, nbins);

	py::capsule free_when_done1( hist, free );

	np_int hist_py = np_int(
					{nbins},
					{sizeof(long)},
					hist,
					free_when_done1);

	return hist_py; 
}

template <class Datatype>
np_double Histogram_Density_py(Datatype py_in, Datatype edges_py, bool density)
{
	py::buffer_info buf_in = py_in.request();
	py::buffer_info buf_edges = edges_py.request();
	int nbins = buf_edges.size-1;

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	if ( density == false )
	{
			np_int in = Histogram_py<np_double>(py_in,edges_py);
			py::buffer_info buf_in = in.request();
			long* hist_long = (long*) buf_in.ptr;
			double* hist = (double*) malloc(sizeof(double)*buf_in.size);
			for(int i=0; i<buf_in.size;i++)
			{
				hist[i] = (double)hist_long[i];
			}
			
			py::capsule free_when_done1( hist, free );
			np_double hist_py = np_double(
						{buf_in.size},
						{sizeof(double)},
						hist,
						free_when_done1);


			return hist_py;
	}
	else if ( density == true )
	{
		int n = buf_in.size;

		double* data = (double*) buf_in.ptr;
		double* hist = (double*) malloc(sizeof(double)*nbins);
		double* edges = (double*) buf_edges.ptr;


		Histogram_Density(hist, edges, data, n, nbins);

		py::capsule free_when_done1( hist, free );

		np_double hist_py = np_double(
						{nbins},
						{sizeof(double)},
						hist,
						free_when_done1);

		return hist_py; 
	}
	else
	{
		throw std::runtime_error("U dumbdumb density must be bool.");
	}
}

