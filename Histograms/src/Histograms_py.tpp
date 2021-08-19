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
	int* hist = (int*) malloc(sizeof(int)*nbins);
	double* edges = GetEdges(data, n, nbins);

	Histogram(hist, edges, data, n, nbins);

	py::capsule free_when_done1( hist, free );
	py::capsule free_when_done2( edges, free );

	np_int hist_py = np_int(
					{nbins},
					{sizeof(int)},
					hist,
					free_when_done1);

	Datatype edges_py = np_double(
					{nbins+1},
					{sizeof(double)},
					edges,
					free_when_done2);	

	std::tuple<np_int,np_double> result = std::make_tuple(hist_py, edges_py);

	return result; 
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
	int* hist = (int*) malloc(sizeof(int)*nbins);
	double* edges = (double*) buf_edges.ptr;

	Histogram(hist, edges, data, n, nbins);

	py::capsule free_when_done1( hist, free );

	np_int hist_py = np_int(
					{nbins},
					{sizeof(int)},
					hist,
					free_when_done1);

	return hist_py; 
}
