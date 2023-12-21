///////////////////////////////////////////////////////////////////
//                       _    ____                               //
//                      / \  / ___|___  _ __ _ __                //
//                     / _ \| |   / _ \| '__| '__|               //
//                    / ___ \ |__| (_) | |  | |                  //
//                   /_/   \_\____\___/|_|  |_|                  //
///////////////////////////////////////////////////////////////////
template<class DataType>
py::array_t<DataType,py::array::c_style>
aCorrCircularFreqAVX_py(py::array_t<DataType,py::array::c_style> py_in, uint64_t size)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	uint64_t N = buf_in.size;
	uint64_t howmany = N/size;
	if(size*howmany != N){howmany+=1;}

	// Retreive all pointers
	DataType* in = (DataType*) buf_in.ptr;
	
	DataType* out;
	out = (DataType*) fftw_malloc(2*(howmany+1)*(size/2+1)*sizeof(DataType));
	std::memset((void*) out, 0, 2*(howmany+1)*(size/2+1)*sizeof(DataType));
	
	DataType* result;
   	result = (DataType*) malloc((size/2+1)*sizeof(DataType));

	// Compute rFFT blocks
	rfftBlock<DataType>((int) N, (int) size, in, 
					reinterpret_cast<std::complex<DataType>*>(out)+(size/2+1));
	
	// Compute product
	aCorrCircularFreqAVX<DataType>(2*howmany*(size/2+1),out+2*(size/2+1),out+2*(size/2+1));
	
	// Sum all blocks
	reduceBlockAVX<DataType>(2*howmany*(size/2+1),2*(size/2+1),
					out+2*(size/2+1),
					out);

	// Divide the sum by the number of blocks
	for(uint64_t i=0;i<(size/2+1);i++){result[i]=(out[2*i]+out[2*i+1])/howmany;}
	
	// Free intermediate buffer
	fftw_free(out);

	py::capsule free_when_done( result, free );
	return py::array_t<DataType, py::array::c_style>
	(
		{(size/2+1)},
		{sizeof(DataType)},
		result,
		free_when_done
	);
}

///////////////////////////////////////////////////////////////////
//                      __  ______                               //
//                      \ \/ / ___|___  _ __ _ __                //
//                       \  / |   / _ \| '__| '__|               //
//                       /  \ |__| (_) | |  | |                  //
//                      /_/\_\____\___/|_|  |_|                  //
///////////////////////////////////////////////////////////////////
template<class DataType>
py::array_t<std::complex<DataType>,py::array::c_style>
xCorrCircularFreqAVX_py(py::array_t<DataType,py::array::c_style> py_in1, 
				py::array_t<DataType,py::array::c_style> py_in2, uint64_t size)
{
	py::buffer_info buf_in1 = py_in1.request();
	py::buffer_info buf_in2 = py_in2.request();

	if (buf_in1.ndim != 1 || buf_in2.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	uint64_t N = std::min(buf_in1.size,buf_in2.size);
	uint64_t howmany = N/size;
	if(size*howmany != N){howmany+=1;}

	// Retreive all pointers
	DataType* in1 = (DataType*) buf_in1.ptr;
	DataType* in2 = (DataType*) buf_in2.ptr;
	
	DataType *out1, *out2;
	out1 = (DataType*) fftw_malloc(2*(howmany+1)*(size/2+1)*sizeof(DataType));
	out2 = (DataType*) fftw_malloc(2*(howmany+1)*(size/2+1)*sizeof(DataType));
	std::memset((void*) out1, 0, 2*(howmany+1)*(size/2+1)*sizeof(DataType));
	std::memset((void*) out2, 0, 2*(howmany+1)*(size/2+1)*sizeof(DataType));
	
	DataType* result;
   	result = (DataType*) malloc(2*(size/2+1)*sizeof(DataType));

	// Compute rFFT blocks
	rfftBlock<DataType>((int) N, (int) size, in1, 
					reinterpret_cast<std::complex<DataType>*>(out1)+(size/2+1));
	rfftBlock<DataType>((int) N, (int) size, in2, 
					reinterpret_cast<std::complex<DataType>*>(out2)+(size/2+1));

	// Compute product
	xCorrCircularFreqAVX<DataType>(2*howmany*(size/2+1),out1+2*(size/2+1),
					out2+2*(size/2+1), out1+2*(size/2+1));
	
	// Sum all blocks
	reduceBlockAVX<DataType>(2*howmany*(size/2+1),2*(size/2+1),
					out1+2*(size/2+1),
					out1);

	// Divide the sum by the number of blocks
	for(uint64_t i=0;i<(2*(size/2+1));i++)
	{
		result[i]=out1[i]/howmany;
	}
	
	// Free intermediate buffer
	fftw_free(out1);
	fftw_free(out2);

	py::capsule free_when_done(result, free);
	return py::array_t<std::complex<DataType>, py::array::c_style>
	(
		{size/2+1},
		{2*sizeof(DataType)},
		reinterpret_cast<std::complex<DataType>*>(result),
		free_when_done
	);
}

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
	uint64_t size = std::max((uint64_t) 256, N*sizeof(DataType)/32);

	DataType* in = (DataType*) buf_in.ptr;
	DataType* out = (DataType*) malloc(size*sizeof(DataType));

	reduceAVX<DataType>(N, in, out);
	DataType result = out[0];
	free(out);

	return result;
}

template<class DataType>
py::array_t<DataType,py::array::c_style>
reduceBlockAVX_py(py::array_t<DataType,py::array::c_style> py_in, uint64_t size)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	uint64_t N = buf_in.size;

	DataType* in = (DataType*) buf_in.ptr;
	DataType* out = (DataType*) malloc(N*sizeof(DataType));

	std::memset((void*) out,0,N*sizeof(DataType));
	
	reduceBlockAVX<DataType>(N, size, in, out);

	py::capsule free_when_done( out, free );
	return py::array_t<DataType, py::array::c_style>
	(
		{size},
		{sizeof(DataType)},
		out,
		free_when_done
	);
}
