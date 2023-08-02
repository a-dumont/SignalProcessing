template<class DataTypeIn, class DataTypeOut>
py::array_t<DataTypeOut,py::array::c_style>
boxcarFilter_py(py::array_t<DataTypeIn,py::array::c_style> data_py, uint64_t order)
{
	py::buffer_info data_buf = data_py.request();
	DataTypeIn* data = (DataTypeIn*) data_buf.ptr;
	uint64_t Ndata = data_buf.size;

	DataTypeOut* out = (DataTypeOut*) malloc(Ndata*sizeof(DataTypeOut));
	std::memset(out,0,Ndata*sizeof(DataTypeOut));

	DataTypeIn* filter = (DataTypeIn*) malloc(order*sizeof(DataTypeIn));
	generateBoxcar<DataTypeIn>(order,filter);

	applyFilter<DataTypeIn,DataTypeOut>(Ndata,order,data,out,filter);
	
	free(filter);
	py::capsule free_when_done(out,free);
	return py::array_t<DataTypeOut,py::array::c_style>
			({Ndata},
			 {sizeof(DataTypeOut)},
			 out,
			 free_when_done);
}

template<class DataTypeIn, class DataTypeOut>
py::array_t<DataTypeOut,py::array::c_style>
boxcarFilterAVX_py(py::array_t<DataTypeIn,py::array::c_style> data_py, uint64_t order)
{
	py::buffer_info data_buf = data_py.request();
	DataTypeIn* data = (DataTypeIn*) data_buf.ptr;
	uint64_t Ndata = data_buf.size;

	DataTypeOut* out = (DataTypeOut*) malloc(Ndata*sizeof(DataTypeOut));
	std::memset(out,0,Ndata*sizeof(DataTypeOut));

	DataTypeIn* filter = (DataTypeIn*) malloc(order*sizeof(DataTypeIn));
	generateBoxcar<DataTypeIn>(order,filter);

	applyFilterAVX2<DataTypeIn,DataTypeOut>(Ndata,order,data,out,filter);
	
	free(filter);
	py::capsule free_when_done(out,free);
	return py::array_t<DataTypeOut,py::array::c_style>
			({Ndata},
			 {sizeof(DataTypeOut)},
			 out,
			 free_when_done);
}
