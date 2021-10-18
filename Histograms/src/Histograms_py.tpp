#include <cstdio>
#include <tuple>
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
	for(int i=0;i<nbins;i++)
	{
		hist[i] = 0;
	}

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
		for(int i=0;i<nbins;i++)
		{
			hist[i] = 0;
		}

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
	for(int i=0;i<nbins;i++)
	{
		hist[i] = 0;
	}

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
		for(int i=0;i<nbins;i++)
		{
			hist[i] = 0;
		}

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
	for(int i=0;i<N;i++)
	{
		hist[i] = 0;
	}

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
		for(int i=0;i<N;i++)
		{
			hist[i] = 0;
		}

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
	for(int i=0;i<N;i++)
	{
		hist[i] = 0;
	}

	Histogram_2D(hist, xedges, yedges, xdata, ydata, n, nbins);

	py::capsule free_when_done1( hist, free );

	np_int hist_py = np_int(
					{nbins,nbins},
					{nbins*sizeof(long),sizeof(long)},
					hist,
					free_when_done1);

	return hist_py; 
}

template <class Datatype, class Datatype2>
np_double Histogram_2D_Density_py(Datatype py_x, Datatype py_y, std::tuple<np_double,np_double>  bins, bool density)
{	
	if (density == false)
	{
		np_int out = Histogram_2D_py<Datatype,Datatype2>(py_x,py_y,bins);
		long* hist_old = (long*) out.request().ptr;
		int nbins = std::get<0>(bins).request().size-1;
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
		return hist_py;
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

		int nbins = std::get<0>(bins).request().size-1;
		int n = buf_x.size;
		int N = nbins*nbins;

		double* xdata = (double*) buf_x.ptr;
		double* ydata = (double*) buf_y.ptr;
		double* xedges = (double*) std::get<0>(bins).request().ptr;
		double* yedges = (double*) std::get<1>(bins).request().ptr;

		double* hist = (double*) malloc(sizeof(double)*N);
		for(int i=0;i<N;i++)
		{
			hist[i] = 0;
		}

		Histogram_2D_Density(hist, xedges, yedges, xdata, ydata, n, nbins);

		py::capsule free_when_done1( hist, free );

		np_double hist_py = np_double(
						{nbins,nbins},
						{nbins*sizeof(double),sizeof(double)},
						hist,
						free_when_done1);

		return hist_py; 
	}
	else
	{
		throw std::runtime_error("U dumbdumb density must be bool.");
	}
}

template <class Datatype, class Datatype2>
int Find_First_In_Bin_py(Datatype py_data, Datatype2 py_edges)
{
	double* data = (double*) py_data.request().ptr;
	double* edges = (double*) py_edges.request().ptr;

	return Find_First_In_Bin(data,edges,py_data.request().size);
}

template <class Datatype, class Datatype2>
int Find_First_In_Bin_2D_py(Datatype py_xdata, Datatype py_ydata, Datatype2 py_xedges, Datatype2 py_yedges)
{
	double* xdata = (double*) py_xdata.request().ptr;
	double* xedges = (double*) py_xedges.request().ptr;
	double* ydata = (double*) py_ydata.request().ptr;
	double* yedges = (double*) py_yedges.request().ptr;

	int nx = py_xdata.request().size;
	int ny = py_ydata.request().size;
	int n = 0;
	if(nx <= ny)
	{
		n += nx;
	}
	else
	{
		n += ny;
	}

	return Find_First_In_Bin_2D(xdata,ydata,xedges,yedges,n);
}

template <class Datatype>
std::tuple<np_uint64,np_double,np_double> Histogram_And_Displacement_2D_py(Datatype py_x, Datatype py_y, int nbins)
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
	int N = (nbins*nbins)*(nbins*nbins+1);

	double* xdata = (double*) buf_x.ptr;
	double* ydata = (double*) buf_y.ptr;
	double* xedges = GetEdges(xdata, n, nbins);
	double* yedges = GetEdges(ydata, n, nbins);

	uint64_t* hist = (uint64_t*) malloc(sizeof(uint64_t)*N);
	for(int i=0;i<N;i++)
	{
		hist[i] = 0;
	}

	Histogram_And_Displacement_2D(hist, xedges, yedges, xdata, ydata, n, nbins);

	py::capsule free_when_done1( hist, free );
	py::capsule free_when_done2( xedges, free );
	py::capsule free_when_done3( yedges, free );

	np_uint64 hist_py = np_uint64(
					{(nbins*nbins+1),nbins,nbins},
					{(nbins*nbins)*sizeof(uint64_t),nbins*sizeof(uint64_t),sizeof(uint64_t)},
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

	std::tuple<np_uint64,np_double,np_double> result = std::make_tuple(hist_py, xedges_py, yedges_py);

	return result; 
}

template <class Datatype, class Datatype2>
np_uint64 Histogram_And_Displacement_2D_py(Datatype py_x, Datatype py_y, std::tuple<np_double,np_double> bins)
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
	int N = (nbins*nbins)*(nbins*nbins+1);

	double* xdata = (double*) buf_x.ptr;
	double* ydata = (double*) buf_y.ptr;

	uint64_t* hist = (uint64_t*) malloc(sizeof(uint64_t)*N);
	for(int i=0;i<N;i++)
		{
			hist[i] = 0;
		}

	Histogram_And_Displacement_2D(hist, xedges, yedges, xdata, ydata, n, nbins);

	py::capsule free_when_done1( hist, free );

	np_uint64 hist_py = np_uint64(
					{(nbins*nbins+1),nbins,nbins},
					{(nbins*nbins)*sizeof(uint64_t),nbins*sizeof(uint64_t),sizeof(uint64_t)},
					hist,
					free_when_done1);

	return hist_py; 
}

template<class Datatype>
class cHistogram2D_py: public cHistogram2D<double>
{
	private:
		static int check(Datatype xdata_py, Datatype ydata_py)
		{
			py::buffer_info buf_x = xdata_py.request();
			py::buffer_info buf_y = ydata_py.request();	
			if (buf_x.ndim != 1 || buf_y.ndim != 1)
			{
				throw std::runtime_error("U dumbdumb inputs dimension must be 1.");
			}	
			else if (buf_x.size != buf_y.size)
			{
				throw std::runtime_error("U dumbdumb inputs must have same length.");
			}
			return buf_x.size;
		}	
	public:
		cHistogram2D_py(Datatype xdata_py, Datatype ydata_py, int Nbins):
				cHistogram2D(
								(double*)xdata_py.request().ptr,
								(double*)ydata_py.request().ptr,
								Nbins,check(xdata_py,ydata_py))
		{
		}
		void accumulate(Datatype xdata_py, Datatype ydata_py)
		{
			n = check(xdata_py,ydata_py);
			double* xdata = (double*)xdata_py.request().ptr;
			double* ydata = (double*)ydata_py.request().ptr;
			Histogram_2D(hist,xedges,yedges,xdata,ydata,n,nbins);
			count += 1;
		}
		np_int getHistogram()
		{
			long* hist2 = (long*)malloc(sizeof(long)*nbins*nbins);
			for(int i=0;i<(nbins*nbins);i++)
			{
				hist2[i] = hist[i];
			}
			py::capsule free_when_done1( hist2, free );

			np_int hist_py = np_int(
							{nbins,nbins},
							{nbins*sizeof(long),sizeof(long)},
							hist2,
							free_when_done1);
			return hist_py;
		}
		std::tuple<Datatype,Datatype> getEdges()
		{
			double* xedges2 = (double*)malloc(sizeof(double)*(nbins+1));
			double* yedges2 = (double*)malloc(sizeof(double)*(nbins+1));
			py::capsule free_when_done2( xedges2, free );
			py::capsule free_when_done3( yedges2, free );
			for(int i=0;i<(nbins+1);i++)
			{
				xedges2[i] = xedges[i];
				yedges2[i] = yedges[i];
			}


			Datatype xedges_py = np_double(
							{nbins+1},
							{sizeof(double)},
							xedges2,
							free_when_done2);	

			Datatype yedges_py = np_double(
							{nbins+1},
							{sizeof(double)},
							yedges2,
							free_when_done3);
			return std::make_tuple(xedges_py,yedges_py);
		}

};

template<class Datatype>
class cHistogram_2D_Density_py: public cHistogram_2D_Density<double>
{
	private:
		static int check(Datatype xdata_py, Datatype ydata_py)
		{
			py::buffer_info buf_x = xdata_py.request();
			py::buffer_info buf_y = ydata_py.request();	
			if (buf_x.ndim != 1 || buf_y.ndim != 1)
			{
				throw std::runtime_error("U dumbdumb inputs dimension must be 1.");
			}	
			else if (buf_x.size != buf_y.size)
			{
				throw std::runtime_error("U dumbdumb inputs must have same length.");
			}
			return buf_x.size;
		}	
	public:
		cHistogram_2D_Density_py(Datatype xdata_py, Datatype ydata_py, int Nbins):
				cHistogram_2D_Density(
								(double*)xdata_py.request().ptr,
								(double*)ydata_py.request().ptr,
								Nbins,check(xdata_py,ydata_py))
		{
		}
		void accumulate(Datatype xdata_py, Datatype ydata_py)
		{
			n = check(xdata_py,ydata_py);
			double* xdata = (double*)xdata_py.request().ptr;
			double* ydata = (double*)ydata_py.request().ptr;
			Histogram_2D_Density(hist,xedges,yedges,xdata,ydata,n,nbins);
			count += 1;
		}
		Datatype getHistogram()
		{
			double* hist2 = (double*)malloc(sizeof(double)*nbins*nbins);
			for(int i=0;i<(nbins*nbins);i++)
			{
				hist2[i] = hist[i];
			}

			py::capsule free_when_done1( hist2, free );

			Datatype hist_py = np_double(
							{nbins,nbins},
							{nbins*sizeof(double),sizeof(double)},
							hist2,
							free_when_done1);
			return hist_py;
		}
		std::tuple<Datatype,Datatype> getEdges()
		{
			double* xedges2 = (double*)malloc(sizeof(double)*(nbins+1));
			double* yedges2 = (double*)malloc(sizeof(double)*(nbins+1));
			py::capsule free_when_done2( xedges2, free );
			py::capsule free_when_done3( yedges2, free );
			for(int i=0;i<(nbins+1);i++)
			{
				xedges2[i] = xedges[i];
				yedges2[i] = yedges[i];
			}


			Datatype xedges_py = np_double(
							{nbins+1},
							{sizeof(double)},
							xedges2,
							free_when_done2);	

			Datatype yedges_py = np_double(
							{nbins+1},
							{sizeof(double)},
							yedges2,
							free_when_done3);
			return std::make_tuple(xedges_py,yedges_py);
		}

};

template<class Datatype>
class cHistogram_And_Displacement_2D_py: public cHistogram_And_Displacement_2D<double>
{
	private:
		static int check(Datatype xdata_py, Datatype ydata_py)
		{
			py::buffer_info buf_x = xdata_py.request();
			py::buffer_info buf_y = ydata_py.request();	
			if (buf_x.ndim != 1 || buf_y.ndim != 1)
			{
				throw std::runtime_error("U dumbdumb inputs dimension must be 1.");
			}	
			else if (buf_x.size != buf_y.size)
			{
				throw std::runtime_error("U dumbdumb inputs must have same length.");
			}
			return buf_x.size;
		}	
	public:
		cHistogram_And_Displacement_2D_py(Datatype xdata_py, Datatype ydata_py, int Nbins):
				cHistogram_And_Displacement_2D(
								(double*)xdata_py.request().ptr,
								(double*)ydata_py.request().ptr,
								Nbins,check(xdata_py,ydata_py))
		{
		}
		void accumulate(Datatype xdata_py, Datatype ydata_py)
		{
			n = check(xdata_py,ydata_py);
			double* xdata = (double*)xdata_py.request().ptr;
			double* ydata = (double*)ydata_py.request().ptr;
			Histogram_And_Displacement_2D(hist,xedges,yedges,xdata,ydata,n,nbins);
			count += 1;
		}
		np_uint64 getHistogram()
		{
			uint64_t* hist2 = (uint64_t*)malloc(sizeof(uint64_t)*nbins*nbins*(nbins*nbins+1));
			for(int i=0;i<(nbins*nbins*(nbins*nbins+1));i++)
			{
				hist2[i] = hist[i];
			}
			py::capsule free_when_done1( hist2, free );

			np_uint64 hist_py = np_uint64(
							{(nbins*nbins+1),nbins,nbins},
							{(nbins*nbins)*sizeof(uint64_t),nbins*sizeof(uint64_t),sizeof(uint64_t)},
							hist2,
							free_when_done1);
			return hist_py;
		}
		std::tuple<Datatype,Datatype> getEdges()
		{
			double* xedges2 = (double*)malloc(sizeof(double)*(nbins+1));
			double* yedges2 = (double*)malloc(sizeof(double)*(nbins+1));
			py::capsule free_when_done2( xedges2, free );
			py::capsule free_when_done3( yedges2, free );
			for(int i=0;i<(nbins+1);i++)
			{
				xedges2[i] = xedges[i];
				yedges2[i] = yedges[i];
			}


			Datatype xedges_py = np_double(
							{nbins+1},
							{sizeof(double)},
							xedges2,
							free_when_done2);	

			Datatype yedges_py = np_double(
							{nbins+1},
							{sizeof(double)},
							yedges2,
							free_when_done3);
			return std::make_tuple(xedges_py,yedges_py);
		}

};
