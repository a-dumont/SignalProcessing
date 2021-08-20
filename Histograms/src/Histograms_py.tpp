#include <functional>
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
					{sizeof(double)},
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

template <class Datatype>
std::tuple<np_int,np_double,np_double> Histogram_2D_py(Datatype py_x, Datatype py_y, int nbins)
{
	py::buffer_info buf_x = py_x.request();
	py::buffer_info buf_y = py_y.request();

	if (buf_x.ndim != 1 || buf_y.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb inputs dimension must be 1.");
	}	

	else if (buf_x.size != buf_y.size)
	{
		throw std::runtime_error("U dumbdumb inputs must have same length.");
	}	

	int n = buf_x.size;
	int N = nbins*nbins;

	double* xdata = (double*) buf_x.ptr;
	double* ydata = (double*) buf_y.ptr;
	double* xedges = GetEdges(xdata, n, nbins);
	double* yedges = GetEdges(ydata, n, nbins);

	long* hist = (long*) malloc(sizeof(long)*N);

	Histogram_2D(hist, xedges, yedges, xdata, ydata, n, nbins);

	py::capsule free_when_done1( hist, free );
	py::capsule free_when_done2( xedges, free );
	py::capsule free_when_done3( yedges, free );

	np_int hist_py = np_int(
					{nbins,nbins},
					{nbins*sizeof(long),sizeof(long)},
					hist,
					free_when_done1);

	Datatype xedges_py = np_double(
					{nbins+1},
					{sizeof(double)},
					xedges,
					free_when_done2);	

	Datatype yedges_py = np_double(
					{nbins+1},
					{sizeof(double)},
					yedges,
					free_when_done3);	

	std::tuple<np_int,np_double,np_double> result = std::make_tuple(hist_py, xedges_py, yedges_py);

	return result; 
}

template <class Datatype>
std::tuple<np_double,np_double,np_double> Histogram_2D_Density_py(Datatype py_x, Datatype py_y, int nbins, bool density)
{	
	if (density == false)
	{
		std::tuple<np_int,np_double,np_double> out = Histogram_2D_py(py_x,py_y,nbins);
		long* hist_old = (long*) std::get<0>(out).request().ptr;
		double* hist = (double*) malloc(sizeof(double)*nbins*nbins);
		for(int i=0;i<nbins*nbins;i++)
		{
			hist[i] = (double)hist_old[i];
		}
		py::capsule free_when_done1( hist, free );
		np_double hist_py = np_double(
						{nbins,nbins},
						{nbins*sizeof(double),sizeof(double)},
						hist,
						free_when_done1);
		return std::tuple<np_double,np_double,np_double>(hist_py,std::get<1>(out),std::get<2>(out));
	}
	else if(density == true)
	{
		py::buffer_info buf_x = py_x.request();
		py::buffer_info buf_y = py_y.request();

		if (buf_x.ndim != 1 || buf_y.ndim != 1)
		{
			throw std::runtime_error("U dumbdumb inputs dimension must be 1.");
		}	

		else if (buf_x.size != buf_y.size)
		{
			throw std::runtime_error("U dumbdumb inputs must have same length.");
		}	

		int n = buf_x.size;
		int N = nbins*nbins;

		double* xdata = (double*) buf_x.ptr;
		double* ydata = (double*) buf_y.ptr;
		double* xedges = GetEdges(xdata, n, nbins);
		double* yedges = GetEdges(ydata, n, nbins);

		double* hist = (double*) malloc(sizeof(double)*N);

		Histogram_2D_Density(hist, xedges, yedges, xdata, ydata, n, nbins);

		py::capsule free_when_done1( hist, free );
		py::capsule free_when_done2( xedges, free );
		py::capsule free_when_done3( yedges, free );

		np_double hist_py = np_double(
						{nbins,nbins},
						{nbins*sizeof(double),sizeof(double)},
						hist,
						free_when_done1);

		Datatype xedges_py = np_double(
						{nbins+1},
						{sizeof(double)},
						xedges,
						free_when_done2);	

		Datatype yedges_py = np_double(
						{nbins+1},
						{sizeof(double)},
						yedges,
						free_when_done3);	

		std::tuple<np_double,np_double,np_double> result = std::make_tuple(hist_py, xedges_py, yedges_py);

		return result; 
	}
	else
	{
		throw std::runtime_error("U dumbdumb density must be bool.");
	}
}

template <class Datatype, class Datatype2>
np_int Histogram_2D_py(Datatype py_x, Datatype py_y, std::tuple<np_double,np_double> bins)
{	
	np_double py_xedges = std::get<0>(bins);
	np_double py_yedges = std::get<1>(bins);
	py::buffer_info buf_x = py_x.request();
	py::buffer_info buf_y = py_y.request();
	py::buffer_info buf_xedges = py_xedges.request();
	py::buffer_info buf_yedges = py_yedges.request();

	if (buf_x.ndim != 1 || buf_y.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb inputs dimension must be 1.");
	}	

	else if (buf_x.size != buf_y.size)
	{
		throw std::runtime_error("U dumbdumb inputs must have same length.");
	}	

	int n = buf_x.size;

	double* xedges = (double*) buf_xedges.ptr;
	double* yedges = (double*) buf_yedges.ptr;
	int nbins = buf_yedges.size-1;
	int N = nbins*nbins;

	double* xdata = (double*) buf_x.ptr;
	double* ydata = (double*) buf_y.ptr;

	long* hist = (long*) malloc(sizeof(long)*N);

	Histogram_2D(hist, xedges, yedges, xdata, ydata, n, nbins);

	py::capsule free_when_done1( hist, free );

	np_int hist_py = np_int(
					{nbins,nbins},
					{nbins*sizeof(long),sizeof(long)},
					hist,
					free_when_done1);

	return hist_py; 
}


