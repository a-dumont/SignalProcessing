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
	//uint64_t n = (uint64_t) std::log2(N); 

	DataType* in = (DataType*) buf_in.ptr;
	//DataType* out = (DataType*) malloc((1<<(n-1))*sizeof(DataType));

	reduceAVX<DataType>(N, in, in, 0.0);

	DataType result = 0;
	result += in[0];
	//free(out);
	
	return result;
}
