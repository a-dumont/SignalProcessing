#include "Histograms_py.h"

void init_histograms(py::module &m)
{
	//1D Histogram
	m.def("histogram",&histogram_py<double>, "data"_a.noconvert(),"nbins"_a);
	m.def("histogram",&histogram_py<float>, "data"_a.noconvert(),"nbins"_a);
	
	m.def("histogram",&histogram_edges_py<double>, "data"_a.noconvert(),"edges"_a.noconvert());
	m.def("histogram",&histogram_edges_py<float>, "data"_a.noconvert(),"edges"_a.noconvert());
	
	m.def("digitizer_histogram",&digitizer_histogram_py<uint8_t>,"data"_a.noconvert());
	m.def("digitizer_histogram",&digitizer_histogram_py<uint16_t>,"data"_a.noconvert());
	m.def("digitizer_histogram",&digitizer_histogram_py<int16_t>,"data"_a.noconvert());
	
	m.def("digitizer_histogram",&digitizer_histogram_subbyte_py<uint8_t>,
					"data"_a.noconvert(),"nbits"_a);
	m.def("digitizer_histogram",&digitizer_histogram_subbyte_py<uint16_t>,
					"data"_a.noconvert(),"nbits"_a);
	m.def("digitizer_histogram",&digitizer_histogram_subbyte_py<int16_t>,
					"data"_a.noconvert(),"nbits"_a);

    
	//1D Histogram density
	m.def("histogram",&histogram_density_py<double,double>, "data"_a.noconvert(),
				"bins"_a,"density"_a);
	m.def("histogram",&histogram_density_py<float,double>, "data"_a.noconvert(),
				"bins"_a,"density"_a);
	m.def("histogram",&histogram_density_py<double,float>, "data"_a.noconvert(),
				"bins"_a,"density"_a);
	m.def("histogram",&histogram_density_py<float,float>, "data"_a.noconvert(),
				"bins"_a,"density"_a);
	
	m.def("histogram",&histogram_density_edges_py<double,double>, "data"_a.noconvert(),
				"edges"_a.noconvert(),"density"_a);
	m.def("histogram",&histogram_density_edges_py<float,double>, "data"_a.noconvert(),
				"edges"_a.noconvert(),"density"_a);
	m.def("histogram",&histogram_density_edges_py<double,float>, "data"_a.noconvert(),
				"edges"_a.noconvert(),"density"_a);
	m.def("histogram",&histogram_density_edges_py<float,float>, "data"_a.noconvert(),
				"edges"_a.noconvert(),"density"_a);
	
	
	//2D Histogram
	m.def("histogram2d",&histogram2D_py<double>, "xdata"_a.noconvert(),
					"ydata.noconvert()"_a,"nbins"_a);
	m.def("histogram2d",&histogram2D_py<float>, "xdata"_a.noconvert(),
					"ydata.noconvert()"_a,"nbins"_a);

	m.def("histogram2d",&histogram2D_edges_py<double>, "xdata"_a.noconvert(),
					"ydata.noconvert()"_a,"edges"_a.noconvert());
	m.def("histogram2d",&histogram2D_edges_py<float>, "xdata"_a.noconvert(),
					"ydata.noconvert()"_a,"edges"_a.noconvert());

	m.def("digitizer_histogram2D",&digitizer_histogram2D_py<uint8_t>,
					"data_x"_a.noconvert(),"data_y"_a.noconvert());

	m.def("digitizer_histogram2D",&digitizer_histogram2D_subbyte_py<uint8_t>,
					"data_x"_a.noconvert(),"data_y"_a.noconvert(),"nbits"_a);
	
	m.def("digitizer_histogram2D_10bits",&digitizer_histogram2D_10bits_py<uint16_t>,
					"data_x"_a.noconvert(),"data_y"_a.noconvert());
	m.def("digitizer_histogram2D_10bits",&digitizer_histogram2D_10bits_py<int16_t>,
					"data_x"_a.noconvert(),"data_y"_a.noconvert());

	//2D Histogram density	
	m.def("histogram2d",&histogram2D_density_py<double,double>,"xdata"_a.noconvert(),
					"ydata"_a.noconvert(),"nbins"_a,"density"_a);
	m.def("histogram2d",&histogram2D_density_py<float,double>,"xdata"_a.noconvert(),
					"ydata"_a.noconvert(),"nbins"_a,"density"_a);
	m.def("histogram2d",&histogram2D_density_py<double,float>,"xdata"_a.noconvert(),
					"ydata"_a.noconvert(),"nbins"_a,"density"_a);
	m.def("histogram2d",&histogram2D_density_py<float,float>,"xdata"_a.noconvert(),
					"ydata"_a.noconvert(),"nbins"_a,"density"_a);

	m.def("histogram2d",&histogram2D_density_edges_py<double,double>,"xdata"_a.noconvert(),
					"ydata"_a.noconvert(),"edges"_a.noconvert(),"density"_a);
	m.def("histogram2d",&histogram2D_density_edges_py<float,double>,"xdata"_a.noconvert(),
					"ydata"_a.noconvert(),"edges"_a.noconvert(),"density"_a);
	m.def("histogram2d",&histogram2D_density_edges_py<double,float>,"xdata"_a.noconvert(),
					"ydata"_a.noconvert(),"edges"_a.noconvert(),"density"_a);
	m.def("histogram2d",&histogram2D_density_edges_py<float,float>,"xdata"_a.noconvert(),
					"ydata"_a.noconvert(),"edges"_a.noconvert(),"density"_a);
	
	//2D Histogram step
	m.def("histogram2d_step",&histogram2D_step_py<double>,
					"xdata"_a.noconvert(),"ydata"_a.noconvert(),"nbins"_a);
	m.def("histogram2d_step",&histogram2D_step_py<float>,
					"xdata"_a.noconvert(),"ydata"_a.noconvert(),"nbins"_a);

	m.def("histogram2d_step",&histogram2D_step_edges_py<double>,
					"xdata"_a.noconvert(),"ydata"_a.noconvert(),"edges"_a.noconvert());
	m.def("histogram2d_step",&histogram2D_step_edges_py<float>,
					"xdata"_a.noconvert(),"ydata"_a.noconvert(),"edges"_a.noconvert());

	m.def("histogram2d_steps",&histogram2D_steps_py<double>,
					"xdata"_a.noconvert(),"ydata"_a.noconvert(),"nbins"_a,"steps"_a);
	m.def("histogram2d_steps",&histogram2D_steps_py<float>,
					"xdata"_a.noconvert(),"ydata"_a.noconvert(),"nbins"_a,"steps"_a);

	m.def("histogram2d_steps",&histogram2D_steps_edges_py<double>,
					"xdata"_a.noconvert(),"ydata"_a.noconvert(),"edges"_a.noconvert(),"steps"_a);
	m.def("histogram2d_steps",&histogram2D_steps_edges_py<float>,
					"xdata"_a.noconvert(),"ydata"_a.noconvert(),"edges"_a.noconvert(),"steps"_a);

	m.def("digitizer_histogram2D_step",&digitizer_histogram2D_step_py<uint8_t>,
					"data_x"_a.noconvert(),"data_y"_a.noconvert(),"nbits"_a);

	m.def("digitizer_histogram2D_steps",&digitizer_histogram2D_steps_py<uint8_t>,
					"data_x"_a.noconvert(),"data_y"_a.noconvert(),"nbits"_a,"steps"_a);
	
	//Functions	
	m.def("find_first_in_bin",&find_first_in_bin_py<double>,"data"_a.noconvert(),
					"edges"_a.noconvert(),"bin"_a);
	m.def("find_first_in_bin",&find_first_in_bin_py<float>,"data"_a.noconvert(),
					"edges"_a.noconvert(),"bin"_a);
	
	m.def("find_first_in_bin2d",&find_first_in_bin2D_py<double>,
					"xdata"_a.noconvert(),"ydata"_a.noconvert(),
					"xedges"_a.noconvert(),"yedges"_a.noconvert(),"xbin"_a,"ybin"_a);
	m.def("find_first_in_bin2d",&find_first_in_bin2D_py<float>,
					"xdata"_a.noconvert(),"ydata"_a.noconvert(),
					"xedges"_a.noconvert(),"yedges"_a.noconvert(),"xbin"_a,"ybin"_a);
	
	m.def("histogram_vectorial_average",&histogram_vectorial_average_py<double>,
					"hist"_a.noconvert(),"row"_a,"col"_a);
	m.def("histogram_vectorial_average",&histogram_vectorial_average_py<float>,
					"hist"_a.noconvert(),"row"_a,"col"_a);
		
	m.def("histogram_nth_order_derivative",&histogram_nth_order_derivative_py<double>,
					"hist_after"_a,"hist_before"_a,"dt"_a,"n"_a,"m"_a);
	m.def("histogram_nth_order_derivative",&histogram_nth_order_derivative_py<float>,
					"hist_after"_a,"hist_before"_a,"dt"_a,"n"_a,"m"_a);

	m.def("detailed_balance",&detailed_balance_py<double>,"p_density"_a.noconvert(),
					"gamma"_a.noconvert(),"time_index"_a);
	m.def("detailed_balance",&detailed_balance_py<float>,"p_density"_a.noconvert(),
					"gamma"_a.noconvert(),"time_index"_a);

	
	//Classes
	py::class_<Histogram2D_py>(m,"Histogram2D")
			.def(py::init<uint64_t>())
			.def("initialize",&Histogram2D_py::initialize_py<double>)
			.def("accumulate",&Histogram2D_py::accumulate_py<double>)
			.def("getHistogram",&Histogram2D_py::getHistogram_py)
			.def("getEdges",&Histogram2D_py::getEdges_py)
			.def("getCount",&Histogram2D_py::getCount)
			.def("getNbins",&Histogram2D_py::getNbins)
			.def("setEdges",&Histogram2D_py::setEdges<double>)
			.def("resetHistogram",&Histogram2D_py::resetHistogram);

	py::class_<Histogram2D_Density_py>(m,"Histogram2D_Density")
			.def(py::init<uint64_t>())
			.def("initialize",&Histogram2D_Density_py::initialize_py<double>)
			.def("accumulate",&Histogram2D_Density_py::accumulate_py<double>)
			.def("getHistogram",&Histogram2D_Density_py::getHistogram_py)
			.def("getEdges",&Histogram2D_Density_py::getEdges_py)
			.def("getCount",&Histogram2D_Density_py::getCount)
			.def("getNbins",&Histogram2D_Density_py::getNbins)
			.def("setEdges",&Histogram2D_Density_py::setEdges<double>)
			.def("resetHistogram",&Histogram2D_Density_py::resetHistogram);

	py::class_<Histogram2D_Step_py>(m,"Histogram2D_Step")
			.def(py::init<uint64_t>())
			.def("initialize",&Histogram2D_Step_py::initialize_py<double>)
			.def("accumulate",&Histogram2D_Step_py::accumulate_py<double>)
			.def("getHistogram",&Histogram2D_Step_py::getHistogram_py)
			.def("getEdges",&Histogram2D_Step_py::getEdges_py)
			.def("getCount",&Histogram2D_Step_py::getCount)
			.def("getNbins",&Histogram2D_Step_py::getNbins)
			.def("setEdges",&Histogram2D_Step_py::setEdges<double>)
			.def("resetHistogram",&Histogram2D_Step_py::resetHistogram);

	py::class_<Histogram2D_Steps_py>(m,"Histogram2D_Steps")
			.def(py::init<uint64_t,uint64_t>())
			.def("initialize",&Histogram2D_Steps_py::initialize_py<double>)
			.def("accumulate",&Histogram2D_Steps_py::accumulate_py<double>)
			.def("getHistogram",&Histogram2D_Steps_py::getHistogram_py)
			.def("getEdges",&Histogram2D_Steps_py::getEdges_py)
			.def("getCount",&Histogram2D_Steps_py::getCount)
			.def("getNbins",&Histogram2D_Steps_py::getNbins)
			.def("setEdges",&Histogram2D_Steps_py::setEdges<double>)
			.def("resetHistogram",&Histogram2D_Steps_py::resetHistogram);

	py::class_<Digitizer_histogram2D_step_py>(m,"Digitizer_Step2D")
			.def(py::init<uint64_t>())
			.def("getHistogram",&Digitizer_histogram2D_step_py::getHistogram)
			.def("getCount",&Digitizer_histogram2D_step_py::getCount)
			.def("getNbits",&Digitizer_histogram2D_step_py::getNbits)
			.def("getSize",&Digitizer_histogram2D_step_py::getSize)
			.def("resetHistogram",&Digitizer_histogram2D_step_py::resetHistogram)
			.def("accumulate",&Digitizer_histogram2D_step_py::accumulate_py<uint8_t>)
			.def("getThreads",&Digitizer_histogram2D_step_py::getThreads);

	py::class_<Digitizer_histogram2D_steps_py>(m,"Digitizer_Steps2D")
			.def(py::init<uint64_t,uint64_t>())
			.def("getHistogram",&Digitizer_histogram2D_steps_py::getHistogram)
			.def("getCount",&Digitizer_histogram2D_steps_py::getCount)
			.def("getSteps",&Digitizer_histogram2D_steps_py::getSteps)
			.def("getNbits",&Digitizer_histogram2D_steps_py::getNbits)
			.def("getSize",&Digitizer_histogram2D_steps_py::getSize)
			.def("resetHistogram",&Digitizer_histogram2D_steps_py::resetHistogram)
			.def("accumulate",&Digitizer_histogram2D_steps_py::accumulate_py<uint8_t>)
			.def("getThreads",&Digitizer_histogram2D_steps_py::getThreads);
}

PYBIND11_MODULE(libhistograms, m)
{
	m.doc() = "Computes Histograms and is meant to be a replacement for numpy in some simple scenarios.";
	init_histograms(m);
}
