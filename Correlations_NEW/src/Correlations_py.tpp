///////////////////////////////////////////////////////////////////
//                       _    ____                               //
//                      / \  / ___|___  _ __ _ __                //
//                     / _ \| |   / _ \| '__| '__|               //
//                    / ___ \ |__| (_) | |  | |                  //
//                   /_/   \_\____\___/|_|  |_|                  //
///////////////////////////////////////////////////////////////////

/*
template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style>
rfft_py(py::array_t<DataType,py::array::c_style> py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	uint64_t N = buf_in.size;

	DataType* in = (DataType*) buf_in.ptr;
	std::complex<DataType>* out = (std::complex<DataType>*) fftw_malloc((N+2)*sizeof(DataType));
	std::memcpy((void*) out,in,N*sizeof(DataType));
	out[N/2] = (std::complex<DataType>) 0.0;

	rfft<DataType>(N, reinterpret_cast<DataType*>(out), out);

	py::capsule free_when_done( out, fftw_free );
	return py::array_t<std::complex<DataType>, py::array::c_style>
	(
		{N/2+1},
		{2*sizeof(DataType)},
		out,
		free_when_done
	);
}*/

///////////////////////////////////////////////////////////////////
//                      __  ______                               //
//                      \ \/ / ___|___  _ __ _ __                //
//                       \  / |   / _ \| '__| '__|               //
//                       /  \ |__| (_) | |  | |                  //
//                      /_/\_\____\___/|_|  |_|                  //
///////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////
//                       _____ ____                              //
//                      |  ___/ ___|___  _ __ _ __               //
//                      | |_ | |   / _ \| '__| '__|              //
//                      |  _|| |__| (_) | |  | |                 //
//                      |_|   \____\___/|_|  |_|                 //
///////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////
//               ___ _____ _   _ _____ ____  ____                //
//				/ _ \_   _| | | | ____|  _ \/ ___|               //
//			   | | | || | | |_| |  _| | |_) \___ \               //
//			   | |_| || | |  _  | |___|  _ < ___) |              //
//				\___/ |_| |_| |_|_____|_| \_\____/               //
///////////////////////////////////////////////////////////////////

template<class DataType>
DataType reduceAVX_py(py::array_t<DataType,py::array::c_style> py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	uint64_t N = buf_in.size;
	uint64_t size = std::max((uint64_t) 22,N/4);

	DataType* in = (DataType*) buf_in.ptr;
	DataType* out = (DataType*) malloc(size*sizeof(DataType));
	//std::memset((void*) out,0,size*sizeof(DataType));

	reduceAVX<DataType>(N, in, out);
	
	DataType result = out[0];
	free(out);

	return result;
}

template<class DataType>
py::array_t<uint8_t,py::array::c_style> 
base16_py(DataType in)
{
	uint64_t N = in;
	uint8_t* out = (uint8_t*) malloc(16*sizeof(uint8_t));

	for(int i=15;i>=0;i--){out[i]=N>>(4*i);N^=(N>>(4*i)<<(4*i));}	
	
	py::capsule free_when_done( out, free );
	return py::array_t<uint8_t, py::array::c_style>
	(
		{16},
		{sizeof(uint8_t)},
		out,
		free_when_done
	);
}

template<class DataType>
py::array_t<uint8_t,py::array::c_style> 
base8_py(DataType in)
{
	uint64_t N = in;
	uint8_t* out = (uint8_t*) malloc(22*sizeof(uint8_t));

	for(int i=21;i>=0;i--){out[i]=N>>(3*i);N^=(N>>(3*i)<<(3*i));}	
	
	py::capsule free_when_done( out, free );
	return py::array_t<uint8_t, py::array::c_style>
	(
		{22},
		{sizeof(uint8_t)},
		out,
		free_when_done
	);
}

template<class DataType>
py::array_t<uint64_t,py::array::c_style> 
base128_py(DataType in)
{
	uint64_t N = in;
	uint64_t* out = (uint64_t*) malloc(10*sizeof(uint64_t));

	for(int i=10;i>=0;i--){out[i]=N>>(7*i);N^=(N>>(7*i)<<(7*i));}	
	
	py::capsule free_when_done( out, free );
	return py::array_t<uint64_t, py::array::c_style>
	(
		{10},
		{sizeof(uint64_t)},
		out,
		free_when_done
	);
}
