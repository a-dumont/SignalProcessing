template<class DataType>
np_uint32 digitizer_histogram_filter_py(py::array_t<DataType,py::array::c_style> data_in, 
				py::array_t<float,py::array::c_style> filter_py, DataType offset)
{
	uint64_t size = (1<<(8*sizeof(DataType)));
	uint32_t* hist = (uint32_t*) malloc(size*sizeof(uint32_t));
	std::memset(hist,0,size*sizeof(uint32_t));

	DataType* data = (DataType*) data_in.request().ptr;
	float* filter = (float*) filter_py.request().ptr;
	uint64_t N = (uint64_t) data_in.size();

	filter_single(N,data,filter,offset);
	
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
	uint32_t size = 1<<(sizeof(DataType)*8);
	uint32_t size2 = size<<(sizeof(DataType)*8);
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


