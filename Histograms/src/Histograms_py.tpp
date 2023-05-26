#include <stdexcept>
#include <vector>
template <class DataType>
std::tuple<np_int,py::array_t<DataType, py::array::c_style>>Histogram_py(
				py::array_t<DataType, py::array::c_style> py_in, long long int nbins)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = buf_in.size;

	DataType* data = (DataType*) buf_in.ptr;
	long long int* hist = (long long int*) malloc(sizeof(long long int)*nbins);	
	std::memset(hist,0,sizeof(long long int)*nbins);

	DataType* edges = (DataType*) malloc(sizeof(DataType)*(nbins+1));
	GetEdges<DataType>(data, n, nbins, edges);

	Histogram<DataType>(hist, edges, data, n, nbins);

	py::capsule free_when_done1( hist, free );
	py::capsule free_when_done2( edges, free );

	np_int hist_py = np_int(
					{nbins},
					{sizeof(long long int)},
					hist,
					free_when_done1);

	py::array_t<DataType,py::array::c_style> edges_py = py::array_t<DataType,py::array::c_style>(
					{nbins+1},
					{sizeof(DataType)},
					edges,
					free_when_done2);	

	std::tuple<np_int,py::array_t<DataType,py::array::c_style>> result; 
	result = std::make_tuple(hist_py, edges_py);

	return result; 
}

template <class DataType>
std::tuple<py::array_t<DataType, py::array::c_style>, py::array_t<DataType, py::array::c_style>> 
Histogram_Density_py(py::array_t<DataType, py::array::c_style> py_in,
				long long int nbins, bool density)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	if ( density == false )
	{
			std::tuple<np_int,py::array_t<DataType, py::array::c_style>> out;
			out = Histogram_py<DataType>(py_in,nbins);
			py::buffer_info buf_in = std::get<0>(out).request();

			long long int* hist_long = (long long int*) buf_in.ptr;
			DataType* hist = (DataType*) malloc(sizeof(DataType)*nbins);	

			for(long long int i=0; i<buf_in.size;i++)
			{
				hist[i] = (DataType) hist_long[i];
			}
			
			py::capsule free_when_done1( hist, free );
			py::array_t<DataType,py::array::c_style> hist_py = py::array_t<DataType,py::array::c_style>(
						{nbins},
						{sizeof(DataType)},
						hist,
						free_when_done1);

			std::tuple<py::array_t<DataType, py::array::c_style>, 
					py::array_t<DataType, py::array::c_style>> result; 
			result = std::make_tuple(hist_py, std::get<1>(out));

			return result;
	}
	else if ( density == true )
	{
		long long int n = buf_in.size;

		DataType* data = (DataType*) buf_in.ptr;
		DataType* hist = (DataType*) malloc(sizeof(DataType)*nbins);
		std::memset(hist,0,sizeof(DataType)*nbins);

		DataType* edges = (DataType*) malloc(sizeof(DataType)*(nbins+1));
		GetEdges<DataType>(data, n, nbins, edges);

		Histogram_Density<DataType>(hist, edges, data, n, nbins);

		py::capsule free_when_done1( hist, free );
		py::capsule free_when_done2( edges, free );

		py::array_t<DataType,py::array::c_style> hist_py=py::array_t<DataType,py::array::c_style>(
						{nbins},
						{sizeof(DataType)},
						hist,
						free_when_done1);

		py::array_t<DataType,py::array::c_style> edges_py=py::array_t<DataType,py::array::c_style>(
						{nbins+1},
						{sizeof(DataType)},
						edges,
						free_when_done2);	

		std::tuple<py::array_t<DataType, py::array::c_style>, 
					py::array_t<DataType, py::array::c_style>> result;
		result = std::make_tuple(hist_py, edges_py);

		return result; 
	}
	else
	{
		throw std::runtime_error("U dumbdumb density must be bool.");
	}
}

template <class DataType, class DataType2>
np_int Histogram_py(py::array_t<DataType, py::array::c_style> py_in, 
				py::array_t<DataType2, py::array::c_style> edges_py)
{
	py::buffer_info buf_in = py_in.request();
	py::buffer_info buf_edges = edges_py.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	long long int n = buf_in.size;
	long long int nbins = buf_edges.size-1;

	DataType* data = (DataType*) buf_in.ptr;
	long long int * hist = (long long int *) malloc(sizeof(long long int)*nbins);
	std::memset(hist,0,sizeof(long long int)*nbins);
	DataType2* edges = (DataType2*) buf_edges.ptr;

	Histogram<DataType>(hist, edges, data, n, nbins);

	py::capsule free_when_done1( hist, free );

	np_int hist_py = np_int(
					{nbins},
					{sizeof(long long int)},
					hist,
					free_when_done1);

	return hist_py; 
}

template <class DataType, class DataType2>
py::array_t<DataType, py::array::c_style> Histogram_Density_py(
				py::array_t<DataType,py::array::c_style> py_in,
				py::array_t<DataType2,py::array::c_style> edges_py, bool density)
{
	py::buffer_info buf_in = py_in.request();
	py::buffer_info buf_edges = edges_py.request();
	long long int nbins = buf_edges.size-1;

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	if ( density == false )
	{
			np_int in = Histogram_py<DataType,DataType2>(py_in,edges_py);
			py::buffer_info buf_in = in.request();
			long long int* hist_long = (long long int*) buf_in.ptr;
			DataType* hist = (DataType*) malloc(sizeof(DataType)*buf_in.size);
			
			for(long long int i=0; i<buf_in.size;i++)
			{
				hist[i] = (DataType)hist_long[i];
			}
			
			py::capsule free_when_done1( hist, free );
			py::array_t<DataType,py::array::c_style> hist_py = py::array_t<DataType,py::array::c_style>(
						{buf_in.size},
						{sizeof(DataType)},
						hist,
						free_when_done1);

			return hist_py;
	}
	else if ( density == true )
	{
		long long int n = buf_in.size;

		DataType* data = (DataType*) buf_in.ptr;
		DataType* hist = (DataType*) malloc(sizeof(DataType)*nbins);
		std::memset(hist,0,sizeof(DataType)*nbins);

		DataType2* edges = (DataType2*) buf_edges.ptr;

		Histogram_Density<DataType>(hist, edges, data, n, nbins);

		py::capsule free_when_done1( hist, free );

		py::array_t<DataType,py::array::c_style> hist_py=py::array_t<DataType,py::array::c_style>(
						{nbins},
						{sizeof(DataType)},
						hist,
						free_when_done1);

		return hist_py; 
	}
	else
	{
		throw std::runtime_error("U dumbdumb density must be bool.");
	}
}

template <class DataType>
std::tuple<np_int,py::array_t<DataType,py::array::c_style>,py::array_t<DataType,py::array::c_style>> Histogram_2D_py(py::array_t<DataType,py::array::c_style> py_x,
				py::array_t<DataType,py::array::c_style> py_y, long long int nbins)
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

	long long int n = buf_x.size;
	long long int N = nbins*nbins;

	DataType* xdata = (DataType*) buf_x.ptr;
	DataType* ydata = (DataType*) buf_y.ptr;

	DataType* xedges = (DataType*) malloc(sizeof(DataType)*(nbins+1)); 
	DataType* yedges = (DataType*) malloc(sizeof(DataType)*(nbins+1));
	GetEdges(xdata, n, nbins, xedges);
	GetEdges(ydata, n, nbins, yedges);

	long long int* hist = (long long int*) malloc(sizeof(long long int)*N);
	std::memset(hist,0,sizeof(long long int)*N);

	Histogram_2D<DataType>(hist, xedges, yedges, xdata, ydata, n, nbins);

	py::capsule free_when_done1( hist, free );
	py::capsule free_when_done2( xedges, free );
	py::capsule free_when_done3( yedges, free );

	np_int hist_py = np_int(
					{nbins,nbins},
					{(long int) nbins*sizeof(long long int),sizeof(long long int)},
					hist,
					free_when_done1);

	py::array_t<DataType,py::array::c_style> xedges_py = py::array_t<DataType,py::array::c_style>(
					{nbins+1},
					{sizeof(DataType)},
					xedges,
					free_when_done2);	

	py::array_t<DataType,py::array::c_style> yedges_py = py::array_t<DataType,py::array::c_style>(
					{nbins+1},
					{sizeof(DataType)},
					yedges,
					free_when_done3);	

	std::tuple<np_int,py::array_t<DataType,py::array::c_style>,py::array_t<DataType,py::array::c_style>> result;
	result = std::make_tuple(hist_py, xedges_py, yedges_py);

	return result; 
}

template <class DataType>
std::tuple<py::array_t<DataType,py::array::c_style>,
		py::array_t<DataType,py::array::c_style>,
		py::array_t<DataType,py::array::c_style>> Histogram_2D_Density_py(
						py::array_t<DataType,py::array::c_style> py_x, 
						py::array_t<DataType,py::array::c_style> py_y, 
						long long int nbins, bool density)
{	
	if (density == false)
	{
		std::tuple<np_int,py::array_t<DataType,py::array::c_style>,
		py::array_t<DataType,py::array::c_style>> out;
		out = Histogram_2D_py<DataType>(py_x,py_y,nbins);
		long long int* hist_old = (long long int*) std::get<0>(out).request().ptr;
		DataType* hist = (DataType*) malloc(sizeof(DataType)*nbins*nbins);
		for(long long int i=0;i<nbins*nbins;i++)
		{
			hist[i] = (DataType) hist_old[i];
		}
		py::capsule free_when_done1( hist, free );
		py::array_t<DataType,py::array::c_style> hist_py;
		hist_py = py::array_t<DataType,py::array::c_style>(
						{nbins,nbins},
						{(long int) nbins*sizeof(DataType),sizeof(DataType)},
						hist,
						free_when_done1);
		return std::tuple<py::array_t<DataType,py::array::c_style>,
		py::array_t<DataType,py::array::c_style>,
		py::array_t<DataType,py::array::c_style>>(hist_py,std::get<1>(out),std::get<2>(out));
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

		long long int n = buf_x.size;
		long long int N = nbins*nbins;

		DataType* xdata = (DataType*) buf_x.ptr;
		DataType* ydata = (DataType*) buf_y.ptr;
		
		DataType* xedges = (DataType*) malloc(sizeof(DataType)*(nbins+1));
		DataType* yedges = (DataType*) malloc(sizeof(DataType)*(nbins+1));
		GetEdges(xdata, n, nbins, xedges);
		GetEdges(ydata, n, nbins, yedges);

		DataType* hist = (DataType*) malloc(sizeof(DataType)*N);
		std::memset(hist,0,sizeof(DataType)*N);

		Histogram_2D_Density<DataType>(hist, xedges, yedges, xdata, ydata, n, nbins);

		py::capsule free_when_done1( hist, free );
		py::capsule free_when_done2( xedges, free );
		py::capsule free_when_done3( yedges, free );

		py::array_t<DataType,py::array::c_style> hist_py;
	    hist_py = py::array_t<DataType,py::array::c_style>(
						{nbins,nbins},
						{(long int) nbins*sizeof(DataType),sizeof(DataType)},
						hist,
						free_when_done1);

		py::array_t<DataType,py::array::c_style> xedges_py;
	   	xedges_py = py::array_t<DataType,py::array::c_style>(
						{nbins+1},
						{sizeof(DataType)},
						xedges,
						free_when_done2);	

		py::array_t<DataType,py::array::c_style> yedges_py;
	   	yedges_py = py::array_t<DataType,py::array::c_style>(
						{nbins+1},
						{sizeof(DataType)},
						yedges,
						free_when_done3);	

		std::tuple<py::array_t<DataType,py::array::c_style>,
		py::array_t<DataType,py::array::c_style>,
		py::array_t<DataType,py::array::c_style>> result;
	   	result = std::make_tuple(hist_py, xedges_py, yedges_py);
		return result; 
	}
	else
	{
		throw std::runtime_error("U dumbdumb density must be bool.");
	}
}

template <class DataType, class DataType2>
np_int Histogram_2D_py(py::array_t<DataType,py::array::c_style> py_x, 
				py::array_t<DataType,py::array::c_style> py_y, 
				std::tuple<py::array_t<DataType2,py::array::c_style>,
				py::array_t<DataType2,py::array::c_style>> bins)
{	
	py::buffer_info buf_x = py_x.request();
	py::buffer_info buf_y = py_y.request();
	py::buffer_info buf_xedges = std::get<0>(bins).request();
	py::buffer_info buf_yedges = std::get<1>(bins).request();

	if (buf_x.ndim != 1 || buf_y.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb inputs dimension must be 1.");
	}	

	else if (buf_x.size != buf_y.size)
	{
		throw std::runtime_error("U dumbdumb inputs must have same length.");
	}	

	long long int n = buf_x.size;

	DataType* xedges = (DataType2*) buf_xedges.ptr;
	DataType* yedges = (DataType2*) buf_yedges.ptr;
	long long int nbins = buf_yedges.size-1;
	long long int N = nbins*nbins;

	DataType* xdata = (DataType*) buf_x.ptr;
	DataType* ydata = (DataType*) buf_y.ptr;

	long long int* hist = (long long int*) malloc(sizeof(long long int)*N);
	std::memset(hist,0,sizeof(long long int)*N);

	Histogram_2D<DataType>(hist, xedges, yedges, xdata, ydata, n, nbins);

	py::capsule free_when_done1( hist, free );

	np_int hist_py = np_int(
					{nbins,nbins},
					{(long int) nbins*sizeof(long long int),sizeof(long long int)},
					hist,
					free_when_done1);

	return hist_py; 
}

template <class DataType, class DataType2>
py::array_t<DataType,py::array::c_style> Histogram_2D_Density_py(
				py::array_t<DataType,py::array::c_style> py_x, 
				py::array_t<DataType,py::array::c_style> py_y, 
				std::tuple<py::array_t<DataType2,py::array::c_style>,
				py::array_t<DataType2,py::array::c_style>> bins, bool density)
{	
	if (density == false)
	{
		np_int out = Histogram_2D_py<DataType,DataType2>(py_x,py_y,bins);
		long long int* hist_old = (long long int*) out.request().ptr;
		long long int nbins = std::get<0>(bins).request().size-1;
		DataType* hist = (DataType*) malloc(sizeof(DataType)*nbins*nbins);
		for(long long int i=0;i<nbins*nbins;i++)
		{
			hist[i] = (DataType) hist_old[i];
		}
		py::capsule free_when_done1( hist, free );
		py::array_t<DataType,py::array::c_style> hist_py;
		hist_py	= py::array_t<DataType,py::array::c_style>(
						{nbins,nbins},
						{(long int) nbins*sizeof(DataType),sizeof(DataType)},
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

		long long int nbins = std::get<0>(bins).request().size-1;
		long long int n = buf_x.size;
		long long int N = nbins*nbins;

		DataType* xdata = (DataType*) buf_x.ptr;
		DataType* ydata = (DataType*) buf_y.ptr;
		DataType2* xedges = (DataType2*) std::get<0>(bins).request().ptr;
		DataType2* yedges = (DataType2*) std::get<1>(bins).request().ptr;

		DataType* hist = (DataType*) malloc(sizeof(DataType)*N);
		std::memset(hist,0,sizeof(DataType)*N);

		Histogram_2D_Density<DataType>(hist, xedges, yedges, xdata, ydata, n, nbins);

		py::capsule free_when_done1( hist, free );

		py::array_t<DataType,py::array::c_style> hist_py; 
		hist_py = py::array_t<DataType,py::array::c_style>(
						{nbins,nbins},
						{(long int) nbins*sizeof(double),sizeof(double)},
						hist,
						free_when_done1);

		return hist_py; 
	}
	else
	{
		throw std::runtime_error("U dumbdumb density must be bool.");
	}
}

template <class DataType>
long long int Find_First_In_Bin_py(py::array_t<DataType,py::array::c_style> py_data,
				py::array_t<DataType,py::array::c_style> py_edges)
{
	DataType* data = (DataType*) py_data.request().ptr;
	DataType* edges = (DataType*) py_edges.request().ptr;

	return Find_First_In_Bin<DataType>(data,edges,(long long int) py_data.request().size);
}

template <class DataType>
long long int Find_First_In_Bin_2D_py(py::array_t<DataType,py::array::c_style> py_xdata,
				py::array_t<DataType,py::array::c_style> py_ydata, 
				py::array_t<DataType,py::array::c_style> py_xedges, 
				py::array_t<DataType,py::array::c_style> py_yedges)
{
	DataType* xdata = (DataType*) py_xdata.request().ptr;
	DataType* xedges = (DataType*) py_xedges.request().ptr;
	DataType* ydata = (DataType*) py_ydata.request().ptr;
	DataType* yedges = (DataType*) py_yedges.request().ptr;

	long long int nx = py_xdata.request().size;
	long long int ny = py_ydata.request().size;
	long long int n = std::min(nx,ny);

	return Find_First_In_Bin_2D<DataType>(xdata,ydata,xedges,yedges,n);
}

template <class DataType>
std::tuple<np_uint64,
		py::array_t<DataType,py::array::c_style>,
		py::array_t<DataType,py::array::c_style>> Histogram_And_Displacement_2D_py(
						py::array_t<DataType,py::array::c_style> py_x,
						py::array_t<DataType,py::array::c_style> py_y, long long int nbins)
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

	long long int n = buf_x.size;
	long long int N = (nbins*nbins)*(nbins*nbins+1);

	DataType* xdata = (DataType*) buf_x.ptr;
	DataType* ydata = (DataType*) buf_y.ptr;

	DataType* xedges = (DataType*) malloc(sizeof(DataType)*(nbins+1));
	DataType* yedges = (DataType*) malloc(sizeof(DataType)*(nbins+1));
	GetEdges(xdata, n, nbins, xedges);
	GetEdges(ydata, n, nbins, yedges);

	uint64_t* hist = (uint64_t*) malloc(sizeof(uint64_t)*N);
	std::memset(hist,0,sizeof(uint64_t)*N);
	
	Histogram_And_Displacement_2D<DataType>(hist, xedges, yedges, xdata, ydata, n, nbins);

	py::capsule free_when_done1( hist, free );
	py::capsule free_when_done2( xedges, free );
	py::capsule free_when_done3( yedges, free );

	np_uint64 hist_py = np_uint64(
					{(nbins*nbins+1),nbins,nbins},
					{(long int) (nbins*nbins)*sizeof(uint64_t),
					(long int) nbins*sizeof(uint64_t),sizeof(uint64_t)},
					hist,
					free_when_done1);

	py::array_t<DataType,py::array::c_style> xedges_py;
   	xedges_py = py::array_t<DataType,py::array::c_style>(
					{nbins+1},
					{sizeof(DataType)},
					xedges,
					free_when_done2);	

	py::array_t<DataType,py::array::c_style> yedges_py; 
	yedges_py = py::array_t<DataType,py::array::c_style>(
					{nbins+1},
					{sizeof(DataType)},
					yedges,
					free_when_done3);	

	std::tuple<np_uint64,py::array_t<DataType,py::array::c_style>,
			py::array_t<DataType,py::array::c_style>> result; 
	result = std::make_tuple(hist_py, xedges_py, yedges_py);

	return result; 
}

template <class DataType, class DataType2>
np_uint64 Histogram_And_Displacement_2D_py(
				py::array_t<DataType,py::array::c_style> py_x,
				py::array_t<DataType,py::array::c_style> py_y,
				std::tuple<py::array_t<DataType2,py::array::c_style>,
				py::array_t<DataType2,py::array::c_style>> bins)
{	
	py::buffer_info buf_x = py_x.request();
	py::buffer_info buf_y = py_y.request();
	py::buffer_info buf_xedges = std::get<0>(bins).request();
	py::buffer_info buf_yedges = std::get<1>(bins).request();

	if (buf_x.ndim != 1 || buf_y.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb inputs dimension must be 1.");
	}	

	else if (buf_x.size != buf_y.size)
	{
		throw std::runtime_error("U dumbdumb inputs must have same length.");
	}	

	long long int n = buf_x.size;

	DataType2* xedges = (DataType2*) buf_xedges.ptr;
	DataType2* yedges = (DataType2*) buf_yedges.ptr;
	long long int nbins = buf_yedges.size-1;
	long long int N = (nbins*nbins)*(nbins*nbins+1);

	DataType* xdata = (DataType*) buf_x.ptr;
	DataType* ydata = (DataType*) buf_y.ptr;

	uint64_t* hist = (uint64_t*) malloc(sizeof(uint64_t)*N);
	std::memset(hist,0,sizeof(uint64_t)*N);

	Histogram_And_Displacement_2D<DataType>(hist, xedges, yedges, xdata, ydata, n, nbins);

	py::capsule free_when_done1( hist, free );

	np_uint64 hist_py = np_uint64(
					{(nbins*nbins+1),nbins,nbins},
					{(long int)(nbins*nbins)*sizeof(uint64_t),
					(long int) nbins*sizeof(uint64_t),sizeof(uint64_t)},
					hist,
					free_when_done1);

	return hist_py; 
}

template<class DataType>
class cHistogram2D_py: public cHistogram2D<double>
{
	private:
		static long long int check(py::array_t<DataType,py::array::c_style> xdata_py,
						py::array_t<DataType,py::array::c_style> ydata_py)
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
		cHistogram2D_py(py::array_t<DataType,py::array::c_style> xdata_py,
						py::array_t<DataType,py::array::c_style> ydata_py,
						long long int Nbins):
				cHistogram2D((DataType*) xdata_py.request().ptr,
								(DataType*) ydata_py.request().ptr,
								Nbins,check(xdata_py,ydata_py))
		{
		}
		void accumulate(py::array_t<DataType,py::array::c_style> xdata_py,
						py::array_t<DataType,py::array::c_style> ydata_py)
		{
			n = check(xdata_py,ydata_py);
			DataType* xdata = (DataType*) xdata_py.request().ptr;
			DataType* ydata = (DataType*) ydata_py.request().ptr;
			Histogram_2D<DataType>(hist,xedges,yedges,xdata,ydata,n,nbins);
			count += 1;
		}
		np_int getHistogram()
		{
			long long int* hist2 = (long long int*)malloc(sizeof(long long int)*nbins*nbins);
			std::memcpy(hist2,hist,sizeof(long long int)*nbins*nbins);
			py::capsule free_when_done1( hist2, free );

			np_int hist_py = np_int(
							{nbins,nbins},
							{(long int) nbins*sizeof(long long int),sizeof(long long int)},
							hist2,
							free_when_done1);
			return hist_py;
		}
		std::tuple<py::array_t<DataType,py::array::c_style>,
				py::array_t<DataType,py::array::c_style>> getEdges()
		{
			DataType* xedges2 = (DataType*)malloc(sizeof(DataType)*(nbins+1));
			DataType* yedges2 = (DataType*)malloc(sizeof(DataType)*(nbins+1));
			xedges2 = (DataType*) std::memcpy(xedges2,xedges,sizeof(DataType)*(nbins+1));
			yedges2 = (DataType*) std::memcpy(yedges2,yedges,sizeof(DataType)*(nbins+1));
			py::capsule free_when_done2( xedges2, free );
			py::capsule free_when_done3( yedges2, free );

			py::array_t<DataType,py::array::c_style> xedges_py;
			xedges_py = py::array_t<DataType,py::array::c_style>(
							{nbins+1},
							{sizeof(DataType)},
							xedges2,
							free_when_done2);	

			py::array_t<DataType,py::array::c_style> yedges_py;
			yedges_py = py::array_t<DataType,py::array::c_style>(
							{nbins+1},
							{sizeof(DataType)},
							yedges2,
							free_when_done3);
			return std::make_tuple(xedges_py,yedges_py);
		}
		void resetHistogram()
		{
			std::memset(hist,0,sizeof(long long int)*nbins*nbins);
			count = 0;
		}
		void setEdges(py::array_t<DataType,py::array::c_style> xe,
						py::array_t<DataType,py::array::c_style> ye)
		{
			xedges = (DataType*)xe.request().ptr;
			yedges = (DataType*)ye.request().ptr;
		}
};

template<class DataType>
class cHistogram_2D_Density_py: public cHistogram_2D_Density<double>
{
	private:
		static long long int check(py::array_t<DataType,py::array::c_style> xdata_py,
						py::array_t<DataType,py::array::c_style> ydata_py)
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
		cHistogram_2D_Density_py(
						py::array_t<DataType,py::array::c_style> xdata_py,
						py::array_t<DataType,py::array::c_style> ydata_py,
						long long int Nbins):
				cHistogram_2D_Density(
								(DataType*)xdata_py.request().ptr,
								(DataType*)ydata_py.request().ptr,
								Nbins,check(xdata_py,ydata_py))
		{
		}
		void accumulate(
					   py::array_t<DataType,py::array::c_style>	xdata_py,
					   py::array_t<DataType,py::array::c_style> ydata_py)
		{
			n = check(xdata_py,ydata_py);
			DataType* xdata = (DataType*)xdata_py.request().ptr;
			DataType* ydata = (DataType*)ydata_py.request().ptr;
			Histogram_2D_Density(hist,xedges,yedges,xdata,ydata,n,nbins);
			count += 1;
		}
		py::array_t<DataType,py::array::c_style> getHistogram()
		{
			DataType* hist2 = (DataType*)malloc(sizeof(DataType)*nbins*nbins);
			std::memcpy(hist2,hist,sizeof(DataType)*nbins*nbins);

			py::capsule free_when_done1( hist2, free );

			py::array_t<DataType,py::array::c_style> hist_py;
			hist_py = py::array_t<DataType,py::array::c_style>(
							{nbins,nbins},
							{(long int) nbins*sizeof(DataType),sizeof(DataType)},
							hist2,
							free_when_done1);
			return hist_py;
		}
		std::tuple<py::array_t<DataType,py::array::c_style>,
				py::array_t<DataType,py::array::c_style>> getEdges()
		{
			DataType* xedges2 = (DataType*)malloc(sizeof(DataType)*(nbins+1));
			DataType* yedges2 = (DataType*)malloc(sizeof(DataType)*(nbins+1));
			std::memcpy(xedges2,xedges,sizeof(DataType)*(nbins+1));
			std::memcpy(yedges2,yedges,sizeof(DataType)*(nbins+1));
			py::capsule free_when_done2( xedges2, free );
			py::capsule free_when_done3( yedges2, free );

			py::array_t<DataType,py::array::c_style> xedges_py;
			xedges_py = py::array_t<DataType,py::array::c_style>(
							{nbins+1},
							{sizeof(double)},
							xedges2,
							free_when_done2);	

			py::array_t<DataType,py::array::c_style> yedges_py;
			yedges_py = py::array_t<DataType,py::array::c_style>(
							{nbins+1},
							{sizeof(double)},
							yedges2,
							free_when_done3);
			return std::make_tuple(xedges_py,yedges_py);
		}
		void resetHistogram()
		{
			std::memset(hist,0,sizeof(DataType)*nbins*nbins);
			count = 0;
		}
		void setEdges(py::array_t<DataType,py::array::c_style> xe,
						py::array_t<DataType,py::array::c_style> ye)
		{
			xedges = (DataType*)xe.request().ptr;
			yedges = (DataType*)ye.request().ptr;
		}
};

template<class DataType>
class cHistogram_And_Displacement_2D_py: public cHistogram_And_Displacement_2D<double>
{
	private:
		static long long int check(
						py::array_t<DataType,py::array::c_style> xdata_py,
						py::array_t<DataType,py::array::c_style> ydata_py)
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
		cHistogram_And_Displacement_2D_py(
						py::array_t<DataType,py::array::c_style> xdata_py,
						py::array_t<DataType,py::array::c_style> ydata_py,
						long long int Nbins):
				cHistogram_And_Displacement_2D(
								(DataType*)xdata_py.request().ptr,
								(DataType*)ydata_py.request().ptr,
								Nbins,check(xdata_py,ydata_py))
		{
		}
		void accumulate(py::array_t<DataType,py::array::c_style> xdata_py,
						py::array_t<DataType,py::array::c_style> ydata_py)
		{
			n = check(xdata_py,ydata_py);
			DataType* xdata = (DataType*)xdata_py.request().ptr;
			DataType* ydata = (DataType*)ydata_py.request().ptr;
			Histogram_And_Displacement_2D(hist,xedges,yedges,xdata,ydata,n,nbins);
			count += 1;
		}
		np_uint64 getHistogram()
		{
			uint64_t* hist2 = (uint64_t*)malloc(sizeof(uint64_t)*nbins*nbins*(nbins*nbins+1));
			std::memcpy(hist2,hist,sizeof(uint64_t)*nbins*nbins*(nbins*nbins+1));
			
			py::capsule free_when_done1( hist2, free );

			np_uint64 hist_py = np_uint64(
							{(nbins*nbins+1),nbins,nbins},
							{(long int) (nbins*nbins)*sizeof(uint64_t),
							(long int) nbins*sizeof(uint64_t),sizeof(uint64_t)},
							hist2,
							free_when_done1);
			return hist_py;
		}
		std::tuple<py::array_t<DataType,py::array::c_style>,
				py::array_t<DataType,py::array::c_style>> getEdges()
		{
			DataType* xedges2 = (DataType*)malloc(sizeof(DataType)*(nbins+1));
			DataType* yedges2 = (DataType*)malloc(sizeof(DataType)*(nbins+1));
			std::memcpy(xedges2,xedges,sizeof(DataType)*(nbins+1));
			std::memcpy(yedges2,yedges,sizeof(DataType)*(nbins+1));
			py::capsule free_when_done2( xedges2, free );
			py::capsule free_when_done3( yedges2, free );

			py::array_t<DataType,py::array::c_style> xedges_py;
			xedges_py = py::array_t<DataType,py::array::c_style>(
							{nbins+1},
							{sizeof(DataType)},
							xedges2,
							free_when_done2);	

			py::array_t<DataType,py::array::c_style> yedges_py;
			yedges_py = py::array_t<DataType,py::array::c_style>(
							{nbins+1},
							{sizeof(DataType)},
							yedges2,
							free_when_done3);
			return std::make_tuple(xedges_py,yedges_py);
		}
		void resetHistogram()
		{
			std::memset(hist,0,sizeof(uint64_t)*nbins*nbins*(nbins*nbins+1));
			count = 0;
		}
		void setEdges(
						py::array_t<DataType,py::array::c_style> xe,
						py::array_t<DataType,py::array::c_style> ye)
		{
			xedges = (DataType*) xe.request().ptr;
			yedges = (DataType*) ye.request().ptr;
		}
};

template<class DataType>
class cHistogram_And_Displacement_2D_steps_py: public cHistogram_And_Displacement_2D_steps<double>
{
	private:
		static long long int check(
						py::array_t<DataType,py::array::c_style> xdata_py,
						py::array_t<DataType,py::array::c_style> ydata_py)
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
		cHistogram_And_Displacement_2D_steps_py(
						py::array_t<DataType,py::array::c_style> xdata_py,
						py::array_t<DataType,py::array::c_style> ydata_py,
						long long int Nbins, long long int steps):
				cHistogram_And_Displacement_2D_steps(
								(DataType*)xdata_py.request().ptr,
								(DataType*)ydata_py.request().ptr,
								Nbins,check(xdata_py,ydata_py),steps)
		{
		}
		void accumulate(py::array_t<DataType,py::array::c_style> xdata_py,
						py::array_t<DataType,py::array::c_style> ydata_py)
		{
			n = check(xdata_py,ydata_py);
			DataType* xdata = (DataType*)xdata_py.request().ptr;
			DataType* ydata = (DataType*)ydata_py.request().ptr;
			Histogram_And_Displacement_2D_steps(hist,hist_before,xedges,yedges,xdata,ydata,n,nbins,steps);
			count += 1;
		}
		np_uint64 getHistogram()
		{
			uint64_t* hist2 = (uint64_t*)malloc(sizeof(uint64_t)*nbins*nbins*(2*steps*nbins*nbins+1));
			std::memcpy(hist2,hist,sizeof(uint64_t)*nbins*nbins*(2*steps*nbins*nbins+1));
			py::capsule free_when_done1( hist2, free );

			np_uint64 hist_py = np_uint64(
							{(2*steps*nbins*nbins+1),nbins,nbins},
							{(long int) (nbins*nbins*sizeof(uint64_t)),
							(long int) (nbins*sizeof(uint64_t)),(long int) sizeof(uint64_t)},
							hist2,
							free_when_done1);
			return hist_py;
		}
		std::tuple<py::array_t<DataType,py::array::c_style>,
				py::array_t<DataType,py::array::c_style>> getEdges()
		{
			DataType* xedges2 = (DataType*)malloc(sizeof(DataType)*(nbins+1));
			DataType* yedges2 = (DataType*)malloc(sizeof(DataType)*(nbins+1));
			std::memcpy(xedges2,xedges,sizeof(DataType)*(nbins+1));
			std::memcpy(yedges2,yedges,sizeof(DataType)*(nbins+1));
			py::capsule free_when_done2( xedges2, free );
			py::capsule free_when_done3( yedges2, free );

			py::array_t<DataType,py::array::c_style> xedges_py;
			xedges_py = py::array_t<DataType,py::array::c_style>(
							{nbins+1},
							{sizeof(DataType)},
							xedges2,
							free_when_done2);	

			py::array_t<DataType,py::array::c_style> yedges_py;
			yedges_py = py::array_t<DataType,py::array::c_style>(
							{nbins+1},
							{sizeof(DataType)},
							yedges2,
							free_when_done3);
			return std::make_tuple(xedges_py,yedges_py);
		}
		void resetHistogram()
		{
			std::memset(hist,0,sizeof(uint64_t)*nbins*nbins*(2*steps*nbins*nbins+1));
			count = 0;
		}
		void setEdges(py::array_t<DataType,py::array::c_style> xe,
						py::array_t<DataType,py::array::c_style> ye)
		{
			xedges = (DataType*) xe.request().ptr;
			yedges = (DataType*) ye.request().ptr;
		}
};

template<class DataType>
py::array_t<DataType,py::array::c_style> histogram_vectorial_average_py(
				py::array_t<DataType,py::array::c_style> py_in,
				long long int row, long long int col)
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
	long long int nbins = py_in.shape(0);
	DataType* out = (DataType*) malloc(2*sizeof(DataType));
	out[0] = 0;
	out[1] = 0;
	DataType* hist = (DataType*) buf_hist.ptr;
	histogram_vectorial_average<DataType>(nbins,hist,out,row,col);

	py::capsule free_when_done( out, free );
	return py::array_t<DataType, py::array::c_style> 
	(
		{2},
		{sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<DataType,py::array::c_style> histogram_nth_order_derivative_py(
				py::array_t<DataType,py::array::c_style> data_after_py,
				py::array_t<DataType,py::array::c_style> data_before_py, 
				DataType dt, long long int m, long long int n)
{
	if(data_after_py.ndim() != 5 || data_before_py.ndim() != 5)
	{
		throw std::runtime_error("U dumbdumb ndim must be 5.");
	}
	if(data_after_py.shape(0) < m || data_before_py.shape(0) < m)
	{
		throw std::runtime_error("U dumbdumb must have atleast m nbins**4 arrays to compute.");
	}
	if(data_after_py.shape(1)*data_after_py.shape(2)*data_after_py.shape(3)*data_after_py.shape(4) != data_after_py.shape(4)*data_after_py.shape(4)*data_after_py.shape(4)*data_after_py.shape(4))
	{
		throw std::runtime_error("U dumbdumb dimensions 1-4 must have same shape.");
	}
	if(data_before_py.shape(1)*data_before_py.shape(2)*data_before_py.shape(3)*data_before_py.shape(4) != data_before_py.shape(4)*data_before_py.shape(4)*data_before_py.shape(4)*data_before_py.shape(4))
	{
		throw std::runtime_error("U dumbdumb dimensions 1-4 must have same shape.");
	}
	long long int nbins = data_after_py.shape(4);
	int nbins2 = data_after_py.shape(4);
	long long int size = nbins*nbins*nbins*nbins;
	DataType* out = (DataType*) malloc(sizeof(DataType)*size);
	DataType* data_after = (DataType*) data_after_py.request().ptr;
	DataType* data_before = (DataType*) data_before_py.request().ptr;

	DataType* coeff = (DataType*) malloc(sizeof(DataType)*(2*n+1)*(2*n+1)*(m+1));
	std::memset(coeff,0,sizeof(DataType)*(2*n+1)*(2*n+1)*(m+1));
	finite_difference_coefficients(m,n,coeff);
	histogram_nth_order_derivative(nbins,data_after,data_before,dt,m,n,out,coeff);
	free(coeff);
	py::capsule free_when_done( out, free );
	return py::array_t<DataType, py::array::c_style> 
	(
		{nbins2,nbins2,nbins2,nbins2},
		{nbins2*nbins2*nbins2*sizeof(DataType),
		nbins2*nbins2*sizeof(DataType),
		nbins2*sizeof(DataType),sizeof(DataType)},
		out,
		free_when_done	
	);
}

template<class DataType>
py::array_t<DataType,py::array::c_style> detailed_balance_py(
				py::array_t<DataType,py::array::c_style> p_density_py,
				py::array_t<DataType,py::array::c_style> gamma_py,
				long long int time_index)
{
	py::buffer_info buf_p = p_density_py.request();
	py::buffer_info buf_g = gamma_py.request();
	if(buf_p.ndim != 2)
	{throw std::runtime_error("U dumbdumb p_density must be a matrix.");}
	if(p_density_py.shape(0) != p_density_py.shape(1))
	{throw std::runtime_error("U dumbdumb p_density must be square.");}
	if(buf_g.ndim != 5)
	{throw std::runtime_error("U dumbdumb gamma must have shape (N,bins,bins,bins,bins).");}
	if(p_density_py.shape(0) != gamma_py.shape(1))
	{throw std::runtime_error("U dumbdumb p_density and gamma shapes must match.");}
	if(time_index >= gamma_py.shape(0))
	{throw std::runtime_error("U dumbdumb time_index too large.");}
	long long int bins = gamma_py.shape(1);
	long long int size = bins*bins*bins*bins;
	long long int offset = time_index*size;
	uint64_t stride3 = bins*bins*bins*sizeof(DataType);
	uint64_t stride2 = bins*bins*sizeof(DataType);
	DataType* out;
	out = (DataType*) malloc(sizeof(DataType)*size);

	DataType* p_density = (DataType*) buf_p.ptr;
	DataType* gamma = (DataType*) buf_g.ptr;

	detailed_balance(bins,p_density,gamma+offset,out);
	
	py::capsule free_when_done(out,free);
	return py::array_t<DataType,py::array::c_style>
	(
		{bins,bins,bins,bins},
		std::vector<uint64_t>{stride3,stride2,bins*sizeof(DataType),sizeof(DataType)},
		out,
		free_when_done
	);
}

template<class DataType>
np_uint32 digitizer_histogram_py(py::array_t<DataType,py::array::c_style> data_in)
{
	uint64_t size = (1<<(8*sizeof(DataType)));
	uint32_t* hist = (uint32_t*) malloc(size*sizeof(uint32_t));
	std::memset(hist,0,size*sizeof(uint32_t));

	DataType* data = (DataType*) data_in.request().ptr;
	uint64_t N = (uint64_t) data_in.size();
	
	digitizer_histogram(hist,data,N);
	py::capsule free_when_done(hist,free);
	return np_uint32
	(
		{size},
		{sizeof(uint32_t)},
		hist,
		free_when_done
	);
}

template<class DataType>
np_uint32 digitizer_histogram_subbyte_py(
				py::array_t<DataType,py::array::c_style> data_in, int nbits)
{
	uint64_t size = (1<<nbits);
	uint32_t* hist = (uint32_t*) malloc(size*sizeof(uint32_t));
	std::memset(hist,0,size*sizeof(uint32_t));

	DataType* data = (DataType*) data_in.request().ptr;
	uint64_t N = (uint64_t) data_in.size();
	
	digitizer_histogram_subbyte<DataType>(hist,data,N,nbits);
	py::capsule free_when_done(hist,free);
	return np_uint32
	(
		{size},
		{sizeof(uint32_t)},
		hist,
		free_when_done
	);
}

template<class DataType>
np_uint32 digitizer_histogram2D_py(
				py::array_t<DataType,py::array::c_style> data_x_in,
				py::array_t<DataType,py::array::c_style> data_y_in)
{
	uint64_t size = 1<<(sizeof(DataType)*8);
	uint64_t size2 = size<<(sizeof(DataType)*8);
	uint32_t* hist = (uint32_t*) malloc(size2*sizeof(uint32_t));
	std::memset(hist,0,size2*sizeof(uint32_t));

	DataType* data_x = (DataType*) data_x_in.request().ptr;
	DataType* data_y = (DataType*) data_y_in.request().ptr;
	uint64_t N = (uint64_t) std::min(data_x_in.size(),data_y_in.size());
	
	digitizer_histogram2D<DataType>(hist,data_x,data_y,N);
	
	py::capsule free_when_done(hist,free);
	return np_uint32
	(
		{size,size},
		{size*sizeof(uint32_t),sizeof(uint32_t)},
		hist,
		free_when_done
	);
}

template<class DataType>
np_uint32 digitizer_histogram2D_10bits_py(
				py::array_t<DataType,py::array::c_style> data_x_in,
				py::array_t<DataType,py::array::c_style> data_y_in)
{
	uint32_t* hist = (uint32_t*) malloc(sizeof(uint32_t)*(1<<20));
	std::memset(hist,0,sizeof(uint32_t)*(1<<20));

	DataType* data_x = (DataType*) data_x_in.request().ptr;
	DataType* data_y = (DataType*) data_y_in.request().ptr;
	uint64_t N = (uint64_t) std::min(data_x_in.size(),data_y_in.size());
	
	digitizer_histogram2D_10bits<DataType>(hist,data_x,data_y,N);
	py::capsule free_when_done(hist,free);
	return np_uint32
	(
		{1<<10,1<<10},
		{sizeof(uint32_t)*(1<<10),sizeof(uint32_t)},
		hist,
		free_when_done
	);
}

template<class DataType>
np_uint32 digitizer_histogram2D_subbyte_py(
				py::array_t<DataType,py::array::c_style> data_x_in,
				py::array_t<DataType,py::array::c_style> data_y_in, uint64_t nbits)
{
	if(nbits > (uint64_t) 10){throw std::runtime_error("U dumbdumb hist too large.");}
	uint64_t size = 1<<nbits;
	uint64_t size2 = size<<nbits;
	uint32_t* hist = (uint32_t*) malloc(size2*sizeof(uint32_t));
	std::memset(hist,0,size2*sizeof(uint32_t));

	DataType* data_x = (DataType*) data_x_in.request().ptr;
	DataType* data_y = (DataType*) data_y_in.request().ptr;
	uint64_t N = (uint64_t) std::min(data_x_in.size(),data_y_in.size());
	
	digitizer_histogram2D_subbyte<DataType>(hist,data_x,data_y,N,nbits);
	py::capsule free_when_done(hist,free);
	return np_uint32
	(
		{size,size},
		{size*sizeof(uint32_t),sizeof(uint32_t)},
		hist,
		free_when_done
	);
}

template<class DataType>
np_uint32 digitizer_histogram2D_step_py(
				py::array_t<DataType,py::array::c_style> data_x_in, 
				py::array_t<DataType,py::array::c_style> data_y_in, 
				uint8_t nbits)
{
	uint64_t size = 1<<nbits;
	uint64_t size2 = size<<nbits;
	uint64_t size4 = size2<<nbits<<nbits;
	
	DataType* data_x = (DataType*) data_x_in.request().ptr;
	DataType* data_y = (DataType*) data_y_in.request().ptr;
	
	uint64_t N = (uint64_t) std::min(data_x_in.size(),data_y_in.size());

	uint32_t* hist = (uint32_t*) malloc((size4+size2)*sizeof(uint32_t));
	std::memset(hist,0,(size4+size2)*sizeof(uint32_t));

	digitizer_histogram2D_step<DataType>(hist,data_x,data_y,N,nbits);

	std::vector<uint64_t> out_size = {size2+1,size,size};
	py::capsule free_when_done(hist,free);
	return np_uint32
	(
		out_size,
		{size2*sizeof(uint32_t),size*sizeof(uint32_t),sizeof(uint32_t)},
		hist,
		free_when_done
	);
}

template<class DataType>
np_uint32 digitizer_histogram2D_steps_py(
				py::array_t<DataType,py::array::c_style> data_x_in, 
				py::array_t<DataType,py::array::c_style> data_y_in, 
				uint8_t nbits, uint8_t steps)
{
	uint64_t size = 1<<nbits;
	uint64_t size2 = size<<nbits;
	uint64_t size4 = size2<<nbits<<nbits;
	
	DataType* data_x = (DataType*) data_x_in.request().ptr;
	DataType* data_y = (DataType*) data_y_in.request().ptr;
	
	uint64_t N = (uint64_t) std::min(data_x_in.size(),data_y_in.size());
	if(N <= steps){throw std::runtime_error("U dumbdumb data must be larger than steps.");}
	if(steps >= (uint8_t) 8)
	{throw std::runtime_error("U dumbdumb too many bits will overflow ram.");}

	uint32_t* hist = (uint32_t*) malloc((2*steps*size4+size2)*sizeof(uint32_t));
	std::memset(hist,0,(2*steps*size4+size2)*sizeof(uint32_t));

	digitizer_histogram2D_steps<DataType>(hist,data_x,data_y,N,nbits,steps);

	std::vector<uint64_t> out_size = {(uint64_t)(2*steps*size2+1),size,size};
	py::capsule free_when_done(hist,free);
	return np_uint32
	(
		out_size,
		{size2*sizeof(uint32_t),size*sizeof(uint32_t),sizeof(uint32_t)},
		hist,
		free_when_done
	);
}

class cdigitizer_histogram2D_step_py: public cdigitizer_histogram2D_step
{
	private:

	public:
		cdigitizer_histogram2D_step_py(uint64_t nbits_in)
		: cdigitizer_histogram2D_step(nbits_in){}
		
		template<class DataType>
		void accumulate_py(py::array_t<DataType,py::array::c_style> x_in,
						py::array_t<DataType,py::array::c_style> y_in)
		{
			DataType* xdata = (DataType*) x_in.request().ptr; 
			DataType* ydata = (DataType*) y_in.request().ptr;
		   	uint64_t N = (uint64_t) std::min(x_in.size(),y_in.size());	
			accumulate(xdata,ydata,N);
		}

		np_uint64 getHistogram()
		{
			uint64_t* hist_out_py = (uint64_t*) malloc(sizeof(uint64_t)*total_size);
			std::memset(hist_out_py,0,total_size*sizeof(uint64_t));
			#pragma omp parallel for num_threads(N_t)
			for(uint64_t j=0;j<N_t;j++)
			{
				manage_thread_affinity();
				for(uint64_t i=0;i<total_size;i++)
				{
					#pragma omp atomic
					hist_out_py[i] += hist[j*total_size+i];
				}
			}
			std::vector<uint64_t> out_size = {(uint64_t)(size*size+1),size,size};
			py::capsule free_when_done(hist_out_py,free);
			return np_uint64(
				out_size,
				{size*size*sizeof(uint64_t),size*sizeof(uint64_t),sizeof(uint64_t)},
				hist_out_py,
				free_when_done);
		}
};

class cdigitizer_histogram2D_steps_py: public cdigitizer_histogram2D_steps
{
	private:

	public:
		cdigitizer_histogram2D_steps_py(uint64_t nbits_in, uint64_t steps_in)
		: cdigitizer_histogram2D_steps(nbits_in,steps_in){}
		
		template<class DataType>
		void accumulate_py(py::array_t<DataType,py::array::c_style> x_in,
						py::array_t<DataType,py::array::c_style> y_in)
		{
			DataType* xdata = (DataType*) x_in.request().ptr; 
			DataType* ydata = (DataType*) y_in.request().ptr;
		   	uint64_t N = (uint64_t) std::min(x_in.size(),y_in.size());	
			accumulate(xdata,ydata,N);
		}

		np_uint64 getHistogram()
		{
			uint64_t total_size = size*size*(2*steps*size*size+1);
			uint64_t* hist_out_py = (uint64_t*) malloc(sizeof(uint64_t)*total_size);
			std::memset(hist_out_py,0,total_size*sizeof(uint64_t));
			#pragma omp parallel for num_threads(N_t)
			for(uint64_t j=0;j<N_t;j++)
			{
				manage_thread_affinity();
				for(uint64_t i=0;i<total_size;i++)
				{
					#pragma omp atomic
					hist_out_py[i] += hist[j*total_size+i];
				}
			}
			std::vector<uint64_t> out_size = {(uint64_t)(2*steps*size*size+1),size,size};
			py::capsule free_when_done(hist_out_py,free);
			return np_uint64(
				out_size,
				{size*size*sizeof(uint64_t),size*sizeof(uint64_t),sizeof(uint64_t)},
				hist_out_py,
				free_when_done);
		}
};

