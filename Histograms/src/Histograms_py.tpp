#include <stdexcept>

////////////////////////////////////////////////////////////////
//  _     _   _   _ _     _                                   //
// / | __| | | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___   //
// | |/ _` | | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \  //
// | | (_| | |  _  | \__ \ || (_) | (_| | | | (_| | | | | | | //
// |_|\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_| //
//                                 |___/                      //
////////////////////////////////////////////////////////////////

template <class DataType>
std::tuple<np_uint64,py::array_t<DataType, py::array::c_style>>
histogram_py(py::array_t<DataType, py::array::c_style> py_in, uint64_t nbins)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	uint64_t N = buf_in.size;

	DataType* data = (DataType*) buf_in.ptr;
	uint32_t* hist = (uint32_t*) malloc(sizeof(uint32_t)*nbins);	
	std::memset(hist,0,sizeof(uint32_t)*nbins);

	DataType* edges = (DataType*) malloc(sizeof(DataType)*(nbins+1));
	get_edges<DataType>(data, N, nbins, edges);

	histogram<DataType>(hist, data, edges, N, nbins);

	py::capsule free_when_done1( hist, free );
	py::capsule free_when_done2( edges, free );

	np_uint32 hist_py;
	hist_py = np_uint32(
					{nbins},
					{sizeof(uint32_t)},
					hist,
					free_when_done1);

	py::array_t<DataType,py::array::c_style> edges_py;
	edges_py = py::array_t<DataType,py::array::c_style>(
					{nbins+1},
					{sizeof(DataType)},
					edges,
					free_when_done2);	

	std::tuple<np_uint32,py::array_t<DataType,py::array::c_style>> result; 
	result = std::make_tuple(hist_py, edges_py);

	return result; 
}

template <class DataType>
np_uint32 histogram_edges_py(py::array_t<DataType, py::array::c_style> py_in, 
				py::array_t<DataType, py::array::c_style> edges_py)
{
	py::buffer_info buf_in = py_in.request();
	py::buffer_info buf_edges = edges_py.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	uint64_t N = buf_in.size;
	uint64_t nbins = buf_edges.size-1;

	DataType* data = (DataType*) buf_in.ptr;
	DataType* edges = (DataType*) buf_edges.ptr;

	uint32_t* hist = (uint32_t*) malloc(sizeof(uint32_t)*nbins);
	std::memset(hist,0,sizeof(uint32_t)*nbins);

	histogram<DataType>(hist, data, edges, N, nbins);

	py::capsule free_when_done1( hist, free );

	np_uint32 hist_py = np_uint32(
					{nbins},
					{sizeof(uint32_t)},
					hist,
					free_when_done1);

	return hist_py; 
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

////////////////////////////////////////////////////////////////
//  _     _   _     _     _                                   //
// / | __| | | |__ (_)___| |_ ___   __ _ _ __ __ _ _ __ ___   //
// | |/ _` | | '_ \| / __| __/ _ \ / _` | '__/ _` | '_ ` _ \  //
// | | (_| | | | | | \__ \ || (_) | (_| | | | (_| | | | | | | //
// |_|\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_| //
//	                               |___/                      //
//			      _                _ _                        //
//             __| | ___ _ __  ___(_) |_ _   _                //
//			  / _` |/ _ \ '_ \/ __| | __| | | |               //
//			 | (_| |  __/ | | \__ \ | |_| |_| |               //
//			  \__,_|\___|_| |_|___/_|\__|\__, |               //
//			                             |___/                //
////////////////////////////////////////////////////////////////

template <class DataTypeIn, class DataTypeOut>
std::tuple<py::array_t<DataTypeOut, py::array::c_style>, 
		py::array_t<DataTypeIn, py::array::c_style>> 
histogram_density_py(py::array_t<DataTypeIn, py::array::c_style> py_in,
				uint64_t nbins, bool density)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}
	
	uint64_t N = buf_in.size;

	DataTypeIn* data = (DataTypeIn*) buf_in.ptr;
	
	DataTypeOut* hist = (DataTypeOut*) malloc(sizeof(DataTypeOut)*nbins);
	std::memset(hist,0,sizeof(DataTypeOut)*nbins);

	DataTypeIn* edges = (DataTypeIn*) malloc(sizeof(DataTypeIn)*(nbins+1));
	get_edges<DataTypeIn>(data, N, nbins, edges);

	histogram_density<DataTypeIn,DataTypeOut>(hist, data, edges, N, nbins, density);

	py::capsule free_when_done1( hist, free );
	py::capsule free_when_done2( edges, free );

	py::array_t<DataTypeOut,py::array::c_style> hist_py;
	hist_py = py::array_t<DataTypeOut,py::array::c_style>(
					{nbins},
					{sizeof(DataTypeOut)},
					hist,
					free_when_done1);

	py::array_t<DataTypeIn,py::array::c_style> edges_py;
	edges_py=py::array_t<DataTypeIn,py::array::c_style>(
					{nbins+1},
					{sizeof(DataTypeIn)},
					edges,
					free_when_done2);	

	std::tuple<py::array_t<DataTypeOut, py::array::c_style>, 
				py::array_t<DataTypeIn, py::array::c_style>> result;
	result = std::make_tuple(hist_py, edges_py);

	return result; 
}

template <class DataTypeIn, class DataTypeOut>
py::array_t<DataTypeOut, py::array::c_style> 
histogram_density_edges_py(py::array_t<DataTypeIn,py::array::c_style> py_in,
				py::array_t<DataTypeIn,py::array::c_style> edges_py, bool density)
{
	py::buffer_info buf_in = py_in.request();
	py::buffer_info buf_edges = edges_py.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}
	
	uint64_t N = buf_in.size;
	uint64_t nbins = buf_edges.size-1;

	DataTypeIn* data = (DataTypeIn*) buf_in.ptr;
	DataTypeIn* edges = (DataTypeIn*) buf_edges.ptr;
	
	DataTypeOut* hist = (DataTypeOut*) malloc(sizeof(DataTypeOut)*nbins);
	std::memset(hist,0,sizeof(DataTypeOut)*nbins);

	histogram_density<DataTypeIn,DataTypeOut>(hist, data, edges, N, nbins, density);

	py::capsule free_when_done1( hist, free );

	py::array_t<DataTypeOut,py::array::c_style> hist_py;
	hist_py = py::array_t<DataTypeOut,py::array::c_style>(
					{nbins},
					{sizeof(DataTypeOut)},
					hist,
					free_when_done1);
	
	return hist_py; 
}

///////////////////////////////////////////////////////////////////
//  ____     _   _   _ _     _                                   //
// |___ \ __| | | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___   //
//   __) / _` | | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \  //
//  / __/ (_| | |  _  | \__ \ || (_) | (_| | | | (_| | | | | | | //
// |_____\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_| //
//					                  |___/                      //
///////////////////////////////////////////////////////////////////

template <class DataType>
std::tuple<
np_uint32,
py::array_t<DataType,py::array::c_style>,
py::array_t<DataType,py::array::c_style>> 
histogram2D_py(py::array_t<DataType,py::array::c_style> py_x,
				py::array_t<DataType,py::array::c_style> py_y, uint64_t nbins)
{
	py::buffer_info buf_x = py_x.request();
	py::buffer_info buf_y = py_y.request();

	if (buf_x.ndim != 1 || buf_y.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb inputs dimension must be 1.");
	}	

	uint64_t N = std::min(buf_x.size,buf_y.size);
	uint64_t size = nbins*nbins;

	DataType* xdata = (DataType*) buf_x.ptr;
	DataType* ydata = (DataType*) buf_y.ptr;

	DataType* xedges = (DataType*) malloc(sizeof(DataType)*(nbins+1)); 
	DataType* yedges = (DataType*) malloc(sizeof(DataType)*(nbins+1));
	get_edges(xdata, N, nbins, xedges);
	get_edges(ydata, N, nbins, yedges);

	uint32_t* hist = (uint32_t*) malloc(sizeof(uint32_t)*size);
	std::memset(hist,0,sizeof(uint32_t)*size);

	histogram2D<DataType>(hist, xedges, yedges, xdata, ydata, N, nbins);

	py::capsule free_when_done1( hist, free );
	py::capsule free_when_done2( xedges, free );
	py::capsule free_when_done3( yedges, free );

	np_uint32 hist_py;
	hist_py = np_uint32(
					{nbins,nbins},
					{nbins*sizeof(uint32_t),sizeof(uint32_t)},
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

	std::tuple<
			np_uint32,
			py::array_t<DataType,py::array::c_style>,
			py::array_t<DataType,py::array::c_style>> result;
	result = std::make_tuple(hist_py, xedges_py, yedges_py);

	return result; 
}

template <class DataType>
np_uint32 
histogram2D_edges_py(
				py::array_t<DataType,py::array::c_style> py_x, 
				py::array_t<DataType,py::array::c_style> py_y, 
				std::tuple<py::array_t<DataType,py::array::c_style>,
				py::array_t<DataType,py::array::c_style>> edges)
{
	py::buffer_info buf_x = py_x.request();
	py::buffer_info buf_y = py_y.request();

	if (buf_x.ndim != 1 || buf_y.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb inputs dimension must be 1.");
	}	

	uint64_t N = std::min(buf_x.size,buf_y.size);
	uint64_t nbins = std::get<0>(edges).request().size-1;
	uint64_t size = nbins*nbins;

	DataType* xdata = (DataType*) buf_x.ptr;
	DataType* ydata = (DataType*) buf_y.ptr;
	DataType* xedges = (DataType*) std::get<0>(edges).request().ptr;
	DataType* yedges = (DataType*) std::get<1>(edges).request().ptr;

	uint32_t* hist = (uint32_t*) malloc(sizeof(uint32_t)*size);
	std::memset(hist,0,sizeof(uint32_t)*size);

	histogram2D<DataType>(hist, xedges, yedges, xdata, ydata, N, nbins);

	py::capsule free_when_done1( hist, free );

	np_uint32 hist_py;
	hist_py = np_uint32(
					{nbins,nbins},
					{nbins*sizeof(uint32_t),sizeof(uint32_t)},
					hist,
					free_when_done1);

	return hist_py; 
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
np_uint32 digitizer_histogram2D_10bits_py(
				py::array_t<DataType,py::array::c_style> data_x_in,
				py::array_t<DataType,py::array::c_style> data_y_in)
{
	
	#ifdef _WIN32_WINNT
		uint64_t nbgroups = GetActiveProcessorGroupCount();
		uint64_t N_t = std::min((uint64_t) 64, omp_get_max_threads()*nbgroups);
	#else
		uint64_t N_t = omp_get_max_threads();
	#endif

	uint16_t* hist = (uint16_t*) malloc(N_t*sizeof(uint16_t)*(1<<20));
	uint32_t* hist_out = (uint32_t*) malloc(sizeof(uint32_t)*(1<<20));
	std::memset(hist,0,N_t*sizeof(uint16_t)*(1<<20));
	std::memset(hist_out,0,sizeof(uint32_t)*(1<<20));

	DataType* data_x = (DataType*) data_x_in.request().ptr;
	DataType* data_y = (DataType*) data_y_in.request().ptr;
	uint64_t N = (uint64_t) std::min(data_x_in.size(),data_y_in.size());
	N = N/N_t;

	#pragma omp parallel for num_threads(N_t)
	for(uint64_t i=0;i<N_t;i++)
	{
		manage_thread_affinity();
		digitizer_histogram2D_10bits<DataType>(hist+(i<<20),data_x+(i*N),data_y+(i*N),N);
		for(uint64_t j=0;j<(1<<20);j++)
		{
			#pragma omp atomic
			hist_out[j] += hist[j+(i<<20)];
		}
	}

	free(hist);
	py::capsule free_when_done(hist_out,free);
	return np_uint32
	(
		{1<<10,1<<10},
		{sizeof(uint32_t)*(1<<10),sizeof(uint32_t)},
		hist_out,
		free_when_done
	);
}

///////////////////////////////////////////////////////////////////
//  ____     _   _   _ _     _                                   //
// |___ \ __| | | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___   //
//   __) / _` | | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \  //
//  / __/ (_| | |  _  | \__ \ || (_) | (_| | | | (_| | | | | | | //
// |_____\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_| //
//					                  |___/                      //
//			      _                _ _                           //
//             __| | ___ _ __  ___(_) |_ _   _                   //
//			  / _` |/ _ \ '_ \/ __| | __| | | |                  //
//			 | (_| |  __/ | | \__ \ | |_| |_| |                  //
//			  \__,_|\___|_| |_|___/_|\__|\__, |                  //
//			                             |___/                   //
///////////////////////////////////////////////////////////////////

template <class DataTypeIn, class DataTypeOut>
std::tuple<
py::array_t<DataTypeOut,py::array::c_style>,
py::array_t<DataTypeIn,py::array::c_style>,
py::array_t<DataTypeIn,py::array::c_style>> 
histogram2D_density_py(py::array_t<DataTypeIn,py::array::c_style> py_x, 
						py::array_t<DataTypeIn,py::array::c_style> py_y, 
						uint64_t nbins, bool density)
{	
	py::buffer_info buf_x = py_x.request();
	py::buffer_info buf_y = py_y.request();

	if (buf_x.ndim != 1 || buf_y.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb inputs dimension must be 1.");
	}	

	uint64_t N = buf_x.size;
	uint64_t size = nbins*nbins;

	DataTypeIn* xdata = (DataTypeIn*) buf_x.ptr;
	DataTypeIn* ydata = (DataTypeIn*) buf_y.ptr;
		
	DataTypeIn* xedges = (DataTypeIn*) malloc(sizeof(DataTypeIn)*(nbins+1));
	DataTypeIn* yedges = (DataTypeIn*) malloc(sizeof(DataTypeIn)*(nbins+1));
	get_edges(xdata, N, nbins, xedges);
	get_edges(ydata, N, nbins, yedges);

	DataTypeOut* hist = (DataTypeOut*) malloc(sizeof(DataTypeOut)*size);
	std::memset(hist,0,sizeof(DataTypeOut)*size);

	histogram2D_density<DataTypeIn,DataTypeOut>(hist, xedges, yedges, xdata, ydata, 
					N, nbins,density);

	py::capsule free_when_done1( hist, free );
	py::capsule free_when_done2( xedges, free );
	py::capsule free_when_done3( yedges, free );

	py::array_t<DataTypeOut,py::array::c_style> hist_py;
    hist_py = py::array_t<DataTypeOut,py::array::c_style>(
					{nbins,nbins},
					{nbins*sizeof(DataTypeOut),sizeof(DataTypeOut)},
					hist,
					free_when_done1);

	py::array_t<DataTypeIn,py::array::c_style> xedges_py;
   	xedges_py = py::array_t<DataTypeIn,py::array::c_style>(
					{nbins+1},
					{sizeof(DataTypeIn)},
					xedges,
					free_when_done2);	

	py::array_t<DataTypeIn,py::array::c_style> yedges_py;
   	yedges_py = py::array_t<DataTypeIn,py::array::c_style>(
					{nbins+1},
					{sizeof(DataTypeIn)},
					yedges,
					free_when_done3);	

	std::tuple<py::array_t<DataTypeOut,py::array::c_style>,
	py::array_t<DataTypeIn,py::array::c_style>,
	py::array_t<DataTypeIn,py::array::c_style>> result;
   	result = std::make_tuple(hist_py, xedges_py, yedges_py);
	return result; 
}

template <class DataTypeIn, class DataTypeOut>
py::array_t<DataTypeOut,py::array::c_style> 
histogram2D_density_edges_py(
				py::array_t<DataTypeIn,py::array::c_style> py_x, 
				py::array_t<DataTypeIn,py::array::c_style> py_y, 
				std::tuple<py::array_t<DataTypeIn,py::array::c_style>,
				py::array_t<DataTypeIn,py::array::c_style>> edges, bool density)
{	
	py::buffer_info buf_x = py_x.request();
	py::buffer_info buf_y = py_y.request();

	if (buf_x.ndim != 1 || buf_y.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb inputs dimension must be 1.");
	}	

	uint64_t nbins = std::get<0>(edges).request().size-1;
	uint64_t N = std::min(buf_x.size,buf_y.size);
	uint64_t size = nbins*nbins;

	DataTypeIn* xdata = (DataTypeIn*) buf_x.ptr;
	DataTypeIn* ydata = (DataTypeIn*) buf_y.ptr;
	DataTypeIn* xedges = (DataTypeIn*) std::get<0>(edges).request().ptr;
	DataTypeIn* yedges = (DataTypeIn*) std::get<1>(edges).request().ptr;

	DataTypeOut* hist = (DataTypeOut*) malloc(sizeof(DataTypeOut)*size);
	std::memset(hist,0,sizeof(DataTypeOut)*size);

	histogram2D_density<DataTypeIn,DataTypeOut>(hist, xedges, yedges, xdata, ydata, N, 
					nbins,density);

	py::capsule free_when_done1( hist, free );

	py::array_t<DataTypeOut,py::array::c_style> hist_py; 
	hist_py = py::array_t<DataTypeOut,py::array::c_style>(
					{nbins,nbins},
					{nbins*sizeof(DataTypeOut),sizeof(DataTypeOut)},
					hist,
					free_when_done1);

	return hist_py; 
}

///////////////////////////////////////////////////////////////////
//  ____     _   _   _ _     _                                   //
// |___ \ __| | | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___   //
//   __) / _` | | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \  //
//  / __/ (_| | |  _  | \__ \ || (_) | (_| | | | (_| | | | | | | //
// |_____\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_| //
//					                  |___/                      //
//                             _                                 //
//	                       ___| |_ ___ _ __                      //
//	                      / __| __/ _ \ '_ \                     //
//	                      \__ \ ||  __/ |_) |                    //
//	                      |___/\__\___| .__/                     //
//	                                  |_|                        //
///////////////////////////////////////////////////////////////////

template <class DataType>
std::tuple<np_uint32,
		py::array_t<DataType,py::array::c_style>,
		py::array_t<DataType,py::array::c_style>> histogram2D_step_py(
						py::array_t<DataType,py::array::c_style> py_x,
						py::array_t<DataType,py::array::c_style> py_y, uint64_t nbins)
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

	uint64_t N = buf_x.size;
	uint64_t size = (nbins*nbins)*(nbins*nbins+1);

	DataType* xdata = (DataType*) buf_x.ptr;
	DataType* ydata = (DataType*) buf_y.ptr;

	DataType* xedges = (DataType*) malloc(sizeof(DataType)*(nbins+1));
	DataType* yedges = (DataType*) malloc(sizeof(DataType)*(nbins+1));
	get_edges(xdata, N, nbins, xedges);
	get_edges(ydata, N, nbins, yedges);

	uint32_t* hist = (uint32_t*) malloc(sizeof(uint32_t)*size);
	std::memset(hist,0,sizeof(uint32_t)*size);
	
	histogram2D_step<DataType>(hist, xedges, yedges, xdata, ydata, N, nbins);

	py::capsule free_when_done1( hist, free );
	py::capsule free_when_done2( xedges, free );
	py::capsule free_when_done3( yedges, free );

	np_uint32 hist_py = np_uint32(
					{nbins*nbins+1,nbins,nbins},
					{nbins*nbins*sizeof(uint32_t),nbins*sizeof(uint32_t),sizeof(uint32_t)},
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

	std::tuple<np_uint32,py::array_t<DataType,py::array::c_style>,
			py::array_t<DataType,py::array::c_style>> result; 
	result = std::make_tuple(hist_py, xedges_py, yedges_py);

	return result; 
}

template <class DataType>
np_uint32 histogram2D_step_edges_py(
				py::array_t<DataType,py::array::c_style> py_x,
				py::array_t<DataType,py::array::c_style> py_y,
				std::tuple<py::array_t<DataType,py::array::c_style>,
				py::array_t<DataType,py::array::c_style>> edges)
{	
	py::buffer_info buf_x = py_x.request();
	py::buffer_info buf_y = py_y.request();
	py::buffer_info buf_xedges = std::get<0>(edges).request();
	py::buffer_info buf_yedges = std::get<1>(edges).request();

	if (buf_x.ndim != 1 || buf_y.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb inputs dimension must be 1.");
	}	

	uint32_t N = std::min(buf_x.size,buf_y.size);
	uint64_t nbins = buf_yedges.size-1;
	uint64_t size = (nbins*nbins)*(nbins*nbins+1);

	DataType* xedges = (DataType*) buf_xedges.ptr;
	DataType* yedges = (DataType*) buf_yedges.ptr;

	DataType* xdata = (DataType*) buf_x.ptr;
	DataType* ydata = (DataType*) buf_y.ptr;

	uint32_t* hist = (uint32_t*) malloc(sizeof(uint32_t)*N);
	std::memset(hist,0,sizeof(uint32_t)*size);

	histogram2D_step<DataType>(hist, xedges, yedges, xdata, ydata, N, nbins);

	py::capsule free_when_done1( hist, free );

	np_uint32 hist_py = np_uint32(
					{nbins*nbins+1,nbins,nbins},
					{nbins*nbins*sizeof(uint32_t),nbins*sizeof(uint32_t),sizeof(uint32_t)},
					hist,
					free_when_done1);

	return hist_py; 
}

template <class DataType>
std::tuple<np_uint32,
		py::array_t<DataType,py::array::c_style>,
		py::array_t<DataType,py::array::c_style>> histogram2D_steps_py(
						py::array_t<DataType,py::array::c_style> py_x,
						py::array_t<DataType,py::array::c_style> py_y, 
						uint64_t nbins, uint64_t steps)
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

	uint64_t N = buf_x.size;
	uint64_t size = (nbins*nbins)*(2*steps*nbins*nbins+1);

	DataType* xdata = (DataType*) buf_x.ptr;
	DataType* ydata = (DataType*) buf_y.ptr;

	DataType* xedges = (DataType*) malloc(sizeof(DataType)*(nbins+1));
	DataType* yedges = (DataType*) malloc(sizeof(DataType)*(nbins+1));
	get_edges(xdata, N, nbins, xedges);
	get_edges(ydata, N, nbins, yedges);

	uint32_t* hist = (uint32_t*) malloc(sizeof(uint32_t)*size);
	std::memset(hist,0,sizeof(uint32_t)*size);

	uint32_t* hist_after = hist;
	uint32_t* hist_before = hist+nbins*nbins*(1+steps*nbins*nbins);
	
	histogram2D_steps<DataType>(hist_after, hist_before, xedges, yedges, xdata, ydata, N, 
					nbins,steps);

	py::capsule free_when_done1( hist, free );
	py::capsule free_when_done2( xedges, free );
	py::capsule free_when_done3( yedges, free );

	np_uint32 hist_py = np_uint32(
					{2*steps*nbins*nbins+1,nbins,nbins},
					{nbins*nbins*sizeof(uint32_t),nbins*sizeof(uint32_t),sizeof(uint32_t)},
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

	std::tuple<np_uint32,py::array_t<DataType,py::array::c_style>,
			py::array_t<DataType,py::array::c_style>> result; 
	result = std::make_tuple(hist_py, xedges_py, yedges_py);

	return result; 
}

template <class DataType>
np_uint32 histogram2D_steps_edges_py(
				py::array_t<DataType,py::array::c_style> py_x,
				py::array_t<DataType,py::array::c_style> py_y,
				std::tuple<py::array_t<DataType,py::array::c_style>,
				py::array_t<DataType,py::array::c_style>> edges, uint64_t steps)
{	
	py::buffer_info buf_x = py_x.request();
	py::buffer_info buf_y = py_y.request();
	py::buffer_info buf_xedges = std::get<0>(edges).request();
	py::buffer_info buf_yedges = std::get<1>(edges).request();

	if (buf_x.ndim != 1 || buf_y.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb inputs dimension must be 1.");
	}	

	uint32_t N = std::min(buf_x.size,buf_y.size);
	uint64_t nbins = buf_yedges.size-1;
	uint64_t size = (nbins*nbins)*(nbins*nbins+1);

	DataType* xedges = (DataType*) buf_xedges.ptr;
	DataType* yedges = (DataType*) buf_yedges.ptr;

	DataType* xdata = (DataType*) buf_x.ptr;
	DataType* ydata = (DataType*) buf_y.ptr;

	uint32_t* hist = (uint32_t*) malloc(sizeof(uint32_t)*N);
	std::memset(hist,0,sizeof(uint32_t)*size);

	uint32_t* hist_after = hist;
	uint32_t* hist_before = hist+nbins*nbins*(1+steps*nbins*nbins);

	histogram2D_steps<DataType>(hist_after, hist_before, xedges, yedges, xdata, ydata, N, 
					nbins,steps);

	py::capsule free_when_done1( hist, free );

	np_uint32 hist_py = np_uint32(
					{2*steps*nbins*nbins+1,nbins,nbins},
					{nbins*nbins*sizeof(uint32_t),nbins*sizeof(uint32_t),sizeof(uint32_t)},
					hist,
					free_when_done1);

	return hist_py; 
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

//////////////////////////////////////////////////
//   __                  _   _                  //
//  / _|_   _ _ __   ___| |_(_) ___  _ __  ___  //
// | |_| | | | '_ \ / __| __| |/ _ \| '_ \/ __| //
// |  _| |_| | | | | (__| |_| | (_) | | | \__ \ //
// |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|___/ //
//                                              //
//////////////////////////////////////////////////

template <class DataType>
uint64_t find_first_in_bin_py(py::array_t<DataType,py::array::c_style> py_data,
				py::array_t<DataType,py::array::c_style> py_edges,uint64_t bin)
{
	DataType* data = (DataType*) py_data.request().ptr;
	DataType* edges = (DataType*) py_edges.request().ptr;

	return find_first_in_bin<DataType>(data,edges,(uint64_t) py_data.request().size,bin);
}

template <class DataType>
uint64_t find_first_in_bin2D_py(py::array_t<DataType,py::array::c_style> py_xdata,
				py::array_t<DataType,py::array::c_style> py_ydata, 
				py::array_t<DataType,py::array::c_style> py_xedges, 
				py::array_t<DataType,py::array::c_style> py_yedges, uint64_t xbin, uint64_t ybin)
{
	DataType* xdata = (DataType*) py_xdata.request().ptr;
	DataType* xedges = (DataType*) py_xedges.request().ptr;
	DataType* ydata = (DataType*) py_ydata.request().ptr;
	DataType* yedges = (DataType*) py_yedges.request().ptr;

	uint64_t n = std::min(py_xdata.request().size,py_ydata.request().size);

	return find_first_in_bin2D<DataType>(xdata,ydata,xedges,yedges,n,xbin,ybin);
}

template<class DataType>
py::array_t<DataType,py::array::c_style> histogram_vectorial_average_py(
				py::array_t<DataType,py::array::c_style> py_in,
				uint64_t row, uint64_t col)
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
	
	uint64_t nbins = py_in.shape(0);
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
				DataType dt, uint64_t m, uint64_t n)
{
	if(data_after_py.ndim() != 5 || data_before_py.ndim() != 5)
	{
		throw std::runtime_error("U dumbdumb ndim must be 5.");
	}
	if((uint64_t) data_after_py.shape(0) < m || (uint64_t) data_before_py.shape(0) < m)
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
	
	uint64_t nbins = data_after_py.shape(4);
	uint64_t nbins2 = data_after_py.shape(4);
	uint64_t size = nbins*nbins*nbins*nbins;
	
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
				uint64_t time_index)
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
	
	if(time_index >= (uint64_t) gamma_py.shape(0))
	{throw std::runtime_error("U dumbdumb time_index too large.");}
	
	uint64_t bins = gamma_py.shape(1);
	uint64_t size = bins*bins*bins*bins;
	uint64_t offset = time_index*size;
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

///////////////////////////////////////
//       _                           //
//   ___| | __ _ ___ ___  ___  ___   //
//  / __| |/ _` / __/ __|/ _ \/ __|  //
// | (__| | (_| \__ \__ \  __/\__ \  //
//  \___|_|\__,_|___/___/\___||___/  //
//                                   //
/////////////////////////////////////// 

class Histogram2D_py: public Histogram2D
{
	public:
		Histogram2D_py(uint64_t nbins_in):Histogram2D(nbins_in){}
	
		template <class DataType>
		void initialize_py(py::array_t<DataType,py::array::c_style> xdata_py,
						py::array_t<DataType,py::array::c_style> ydata_py)
		{
			py::buffer_info buf_x = xdata_py.request();
			py::buffer_info buf_y = ydata_py.request();
			
			if (buf_x.ndim != 1 || buf_y.ndim != 1)
			{
				throw std::runtime_error("U dumbdumb inputs dimension must be 1.");
			}	

			uint64_t N = std::min(buf_x.size,buf_y.size);

			DataType* xdata = (DataType*) buf_x.ptr;
			DataType* ydata = (DataType*) buf_y.ptr;
			initialize(xdata,ydata,N);
		}
		
		template <class DataType>
		void accumulate_py(py::array_t<DataType,py::array::c_style> xdata_py,
						py::array_t<DataType,py::array::c_style> ydata_py)
		{
			py::buffer_info buf_x = xdata_py.request();
			py::buffer_info buf_y = xdata_py.request();
			
			if (buf_x.ndim != 1 || buf_y.ndim != 1)
			{
				throw std::runtime_error("U dumbdumb inputs dimension must be 1.");
			}	

			uint64_t N = std::min(buf_x.size,buf_y.size);

			DataType* xdata = (DataType*) buf_x.ptr;
			DataType* ydata = (DataType*) buf_y.ptr;
			accumulate(xdata,ydata,N);
		}

		np_uint64 getHistogram_py()
		{
			uint64_t* hist_out = (uint64_t*) malloc(sizeof(uint64_t)*size);
			std::memset(hist_out,0,sizeof(uint64_t)*size);

			#pragma omp parallel for
			for(uint64_t i=0;i<size;i++){hist_out[i] += hist[i];}	

			py::capsule free_when_done_out( hist_out, free );

			np_uint64 hist_py = np_uint64(
							{nbins,nbins},
							{nbins*sizeof(uint64_t),sizeof(uint64_t)},
							hist_out,
							free_when_done_out);
			return hist_py;
		}

		std::tuple<np_double,np_double> getEdges_py()
		{
			double* xedges_out = (double*) malloc(sizeof(double)*(nbins+1));
			double* yedges_out = (double*) malloc(sizeof(double)*(nbins+1));
			
			std::memcpy(xedges_out,xedges,sizeof(double)*(nbins+1));
			std::memcpy(yedges_out,yedges,sizeof(double)*(nbins+1));
			
			py::capsule free_when_done_xout( xedges_out, free );
			py::capsule free_when_done_yout( yedges_out, free );

			np_double xedges_py;
			xedges_py = np_double(
							{nbins+1},
							{sizeof(double)},
							xedges_out,
							free_when_done_xout);	

			np_double yedges_py;
			yedges_py = np_double(
							{nbins+1},
							{sizeof(double)},
							yedges_out,
							free_when_done_yout);
			return std::make_tuple(xedges_py,yedges_py);
		}

		template <class DataType>
		void setEdges(py::array_t<DataType,py::array::c_style> xe,
						py::array_t<DataType,py::array::c_style> ye)
		{
			py::buffer_info buf_xe = xe.request();
			py::buffer_info buf_ye = ye.request();

			if((uint64_t) buf_xe.size != nbins || (uint64_t) buf_ye.size != nbins)
			{
				throw std::runtime_error("U dumbdumb edges must be size nbins+1.");
			}

			resetEdges();
			double* xedges_in = (double*) buf_xe.ptr;
			double* yedges_in = (double*) buf_ye.ptr;
			
			for(uint64_t i=0;i<(nbins+1);i++)
			{
				xedges[i] = (double) xedges_in[i];
				yedges[i] = (double) yedges_in[i];
			}
		}
};

class Histogram2D_Density_py: public Histogram2D_Density
{
	public:
		Histogram2D_Density_py(uint64_t nbins_in):Histogram2D_Density(nbins_in){}
	
		template <class DataType>
		void initialize_py(py::array_t<DataType,py::array::c_style> xdata_py,
						py::array_t<DataType,py::array::c_style> ydata_py)
		{
			py::buffer_info buf_x = xdata_py.request();
			py::buffer_info buf_y = ydata_py.request();
			
			if (buf_x.ndim != 1 || buf_y.ndim != 1)
			{
				throw std::runtime_error("U dumbdumb inputs dimension must be 1.");
			}	

			uint64_t N = std::min(buf_x.size,buf_y.size);

			DataType* xdata = (DataType*) buf_x.ptr;
			DataType* ydata = (DataType*) buf_y.ptr;
			initialize(xdata,ydata,N);
		}
		
		template <class DataType>
		void accumulate_py(py::array_t<DataType,py::array::c_style> xdata_py,
						py::array_t<DataType,py::array::c_style> ydata_py)
		{
			py::buffer_info buf_x = xdata_py.request();
			py::buffer_info buf_y = xdata_py.request();
			
			if (buf_x.ndim != 1 || buf_y.ndim != 1)
			{
				throw std::runtime_error("U dumbdumb inputs dimension must be 1.");
			}	

			uint64_t N = std::min(buf_x.size,buf_y.size);

			DataType* xdata = (DataType*) buf_x.ptr;
			DataType* ydata = (DataType*) buf_y.ptr;
			accumulate(xdata,ydata,N);
		}

		np_double getHistogram_py()
		{
			double* hist_out = (double*) malloc(sizeof(double)*size);

			#pragma omp parallel for
			for(uint64_t i=0;i<size;i++){hist_out[i] = hist[i];}	

			py::capsule free_when_done_out( hist_out, free );

			np_double hist_py = np_double(
							{nbins,nbins},
							{nbins*sizeof(double),sizeof(double)},
							hist_out,
							free_when_done_out);
			return hist_py;
		}

		std::tuple<np_double,np_double> getEdges_py()
		{
			double* xedges_out = (double*) malloc(sizeof(double)*(nbins+1));
			double* yedges_out = (double*) malloc(sizeof(double)*(nbins+1));
			
			std::memcpy(xedges_out,xedges,sizeof(double)*(nbins+1));
			std::memcpy(yedges_out,yedges,sizeof(double)*(nbins+1));
			
			py::capsule free_when_done_xout( xedges_out, free );
			py::capsule free_when_done_yout( yedges_out, free );

			np_double xedges_py;
			xedges_py = np_double(
							{nbins+1},
							{sizeof(double)},
							xedges_out,
							free_when_done_xout);	

			np_double yedges_py;
			yedges_py = np_double(
							{nbins+1},
							{sizeof(double)},
							yedges_out,
							free_when_done_yout);
			return std::make_tuple(xedges_py,yedges_py);
		}

		template <class DataType>
		void setEdges(py::array_t<DataType,py::array::c_style> xe,
						py::array_t<DataType,py::array::c_style> ye)
		{
			py::buffer_info buf_xe = xe.request();
			py::buffer_info buf_ye = ye.request();

			if((uint64_t) buf_xe.size != nbins || (uint64_t) buf_ye.size != nbins)
			{
				throw std::runtime_error("U dumbdumb edges must be size nbins+1.");
			}

			resetEdges();
			double* xedges_in = (double*) buf_xe.ptr;
			double* yedges_in = (double*) buf_ye.ptr;
			
			for(uint64_t i=0;i<(nbins+1);i++)
			{
				xedges[i] = (double) xedges_in[i];
				yedges[i] = (double) yedges_in[i];
			}
		}
};

class Histogram2D_Step_py: public Histogram2D_Step
{
	public:
		Histogram2D_Step_py(uint64_t nbins_in):Histogram2D_Step(nbins_in){}
	
		template <class DataType>
		void initialize_py(py::array_t<DataType,py::array::c_style> xdata_py,
						py::array_t<DataType,py::array::c_style> ydata_py)
		{
			py::buffer_info buf_x = xdata_py.request();
			py::buffer_info buf_y = ydata_py.request();
			
			if (buf_x.ndim != 1 || buf_y.ndim != 1)
			{
				throw std::runtime_error("U dumbdumb inputs dimension must be 1.");
			}	

			uint64_t N = std::min(buf_x.size,buf_y.size);

			DataType* xdata = (DataType*) buf_x.ptr;
			DataType* ydata = (DataType*) buf_y.ptr;
			initialize(xdata,ydata,N);
		}
		
		template <class DataType>
		void accumulate_py(py::array_t<DataType,py::array::c_style> xdata_py,
						py::array_t<DataType,py::array::c_style> ydata_py)
		{
			py::buffer_info buf_x = xdata_py.request();
			py::buffer_info buf_y = xdata_py.request();
			
			if (buf_x.ndim != 1 || buf_y.ndim != 1)
			{
				throw std::runtime_error("U dumbdumb inputs dimension must be 1.");
			}	

			uint64_t N = std::min(buf_x.size,buf_y.size);

			DataType* xdata = (DataType*) buf_x.ptr;
			DataType* ydata = (DataType*) buf_y.ptr;
			accumulate(xdata,ydata,N);
		}

		np_uint64 getHistogram_py()
		{
			uint64_t* hist_out = (uint64_t*) malloc(sizeof(uint64_t)*size);
			std::memset(hist_out,0,sizeof(uint64_t)*size);

			#pragma omp parallel for
			for(uint64_t i=0;i<size;i++){hist_out[i] += hist[i];}	

			py::capsule free_when_done_out( hist_out, free );

			np_uint64 hist_py = np_uint64(
							{(nbins*nbins)+1,nbins,nbins},
							{nbins*nbins*sizeof(uint64_t),nbins*sizeof(uint64_t),
							sizeof(uint64_t)},
							hist_out,
							free_when_done_out);
			return hist_py;
		}

		std::tuple<np_double,np_double> getEdges_py()
		{
			double* xedges_out = (double*) malloc(sizeof(double)*(nbins+1));
			double* yedges_out = (double*) malloc(sizeof(double)*(nbins+1));
			
			std::memcpy(xedges_out,xedges,sizeof(double)*(nbins+1));
			std::memcpy(yedges_out,yedges,sizeof(double)*(nbins+1));
			
			py::capsule free_when_done_xout( xedges_out, free );
			py::capsule free_when_done_yout( yedges_out, free );

			np_double xedges_py;
			xedges_py = np_double(
							{nbins+1},
							{sizeof(double)},
							xedges_out,
							free_when_done_xout);	

			np_double yedges_py;
			yedges_py = np_double(
							{nbins+1},
							{sizeof(double)},
							yedges_out,
							free_when_done_yout);
			return std::make_tuple(xedges_py,yedges_py);
		}

		template <class DataType>
		void setEdges(py::array_t<DataType,py::array::c_style> xe,
						py::array_t<DataType,py::array::c_style> ye)
		{
			py::buffer_info buf_xe = xe.request();
			py::buffer_info buf_ye = ye.request();

			if((uint64_t) buf_xe.size != nbins || (uint64_t) buf_ye.size != nbins)
			{
				throw std::runtime_error("U dumbdumb edges must be size nbins+1.");
			}

			resetEdges();
			double* xedges_in = (double*) buf_xe.ptr;
			double* yedges_in = (double*) buf_ye.ptr;
			
			for(uint64_t i=0;i<(nbins+1);i++)
			{
				xedges[i] = (double) xedges_in[i];
				yedges[i] = (double) yedges_in[i];
			}
		}
};

class Histogram2D_Steps_py: public Histogram2D_Steps
{
	public:
		Histogram2D_Steps_py(uint64_t nbins_in, uint64_t steps_in):
				Histogram2D_Steps(nbins_in,steps_in){}
	
		template <class DataType>
		void initialize_py(py::array_t<DataType,py::array::c_style> xdata_py,
						py::array_t<DataType,py::array::c_style> ydata_py)
		{
			py::buffer_info buf_x = xdata_py.request();
			py::buffer_info buf_y = ydata_py.request();
			
			if (buf_x.ndim != 1 || buf_y.ndim != 1)
			{
				throw std::runtime_error("U dumbdumb inputs dimension must be 1.");
			}	

			uint64_t N = std::min(buf_x.size,buf_y.size);

			DataType* xdata = (DataType*) buf_x.ptr;
			DataType* ydata = (DataType*) buf_y.ptr;
			initialize(xdata,ydata,N);
		}
		
		template <class DataType>
		void accumulate_py(py::array_t<DataType,py::array::c_style> xdata_py,
						py::array_t<DataType,py::array::c_style> ydata_py)
		{
			py::buffer_info buf_x = xdata_py.request();
			py::buffer_info buf_y = xdata_py.request();
			
			if (buf_x.ndim != 1 || buf_y.ndim != 1)
			{
				throw std::runtime_error("U dumbdumb inputs dimension must be 1.");
			}	

			uint64_t N = std::min(buf_x.size,buf_y.size);

			DataType* xdata = (DataType*) buf_x.ptr;
			DataType* ydata = (DataType*) buf_y.ptr;
			accumulate(xdata,ydata,N);
		}

		np_uint64 getHistogram_py()
		{
			uint64_t* hist_out = (uint64_t*) malloc(sizeof(uint64_t)*size);
			std::memset(hist_out,0,sizeof(uint64_t)*size);

			#pragma omp parallel for
			for(uint64_t i=0;i<size;i++){hist_out[i] += hist[i];}	

			py::capsule free_when_done_out( hist_out, free );

			np_uint64 hist_py = np_uint64(
							{(2*steps*nbins*nbins)+1,nbins,nbins},
							{nbins*nbins*sizeof(uint64_t),nbins*sizeof(uint64_t),
							sizeof(uint64_t)},
							hist_out,
							free_when_done_out);
			return hist_py;
		}

		std::tuple<np_double,np_double> getEdges_py()
		{
			double* xedges_out = (double*) malloc(sizeof(double)*(nbins+1));
			double* yedges_out = (double*) malloc(sizeof(double)*(nbins+1));
			
			std::memcpy(xedges_out,xedges,sizeof(double)*(nbins+1));
			std::memcpy(yedges_out,yedges,sizeof(double)*(nbins+1));
			
			py::capsule free_when_done_xout( xedges_out, free );
			py::capsule free_when_done_yout( yedges_out, free );

			np_double xedges_py;
			xedges_py = np_double(
							{nbins+1},
							{sizeof(double)},
							xedges_out,
							free_when_done_xout);	

			np_double yedges_py;
			yedges_py = np_double(
							{nbins+1},
							{sizeof(double)},
							yedges_out,
							free_when_done_yout);
			return std::make_tuple(xedges_py,yedges_py);
		}

		template <class DataType>
		void setEdges(py::array_t<DataType,py::array::c_style> xe,
						py::array_t<DataType,py::array::c_style> ye)
		{
			py::buffer_info buf_xe = xe.request();
			py::buffer_info buf_ye = ye.request();

			if((uint64_t) buf_xe.size != nbins || (uint64_t) buf_ye.size != nbins)
			{
				throw std::runtime_error("U dumbdumb edges must be size nbins+1.");
			}

			resetEdges();
			double* xedges_in = (double*) buf_xe.ptr;
			double* yedges_in = (double*) buf_ye.ptr;
			
			for(uint64_t i=0;i<(nbins+1);i++)
			{
				xedges[i] = (double) xedges_in[i];
				yedges[i] = (double) yedges_in[i];
			}
		}
};


class Digitizer_histogram2D_step_py: public Digitizer_histogram2D_step
{
	private:

	public:
		Digitizer_histogram2D_step_py(uint64_t nbits_in)
		: Digitizer_histogram2D_step(nbits_in){}
		
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

class Digitizer_histogram2D_steps_py: public Digitizer_histogram2D_steps
{
	private:

	public:
		Digitizer_histogram2D_steps_py(uint64_t nbits_in, uint64_t steps_in)
		: Digitizer_histogram2D_steps(nbits_in,steps_in){}
		
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
