#include "Histograms_py.h"

void init_histograms(py::module &m)
{
	//Returns count
	m.def("histogram",&Histogram_py<long long int>, "x"_a,"nbins"_a);
	m.def("histogram",&Histogram_py<float>, "x"_a,"nbins"_a);
	m.def("histogram",&Histogram_py<double>, "x"_a,"nbins"_a);
	m.def("histogram",&Histogram_py<long long int, long long int>, "x"_a,"edges"_a);
	m.def("histogram",&Histogram_py<float, float>, "x"_a,"edges"_a);
	m.def("histogram",&Histogram_py<double, double>, "x"_a,"edges"_a);

	//Returns normalized histograms
	m.def("histogram",&Histogram_Density_py<float>, "x"_a,"nbins"_a,"density"_a);
	m.def("histogram",&Histogram_Density_py<double>, "x"_a,"nbins"_a,"density"_a);	
	m.def("histogram",&Histogram_Density_py<float,float>, "x"_a,"edges"_a,"density"_a);
	m.def("histogram",&Histogram_Density_py<double,double>, "x"_a,"edges"_a,"density"_a);
	
	//Returns count 2D
	m.def("histogram2d",&Histogram_2D_py<long long int>, "x"_a,"y"_a,"nbins"_a);
	m.def("histogram2d",&Histogram_2D_py<float>, "x"_a,"y"_a,"nbins"_a);
	m.def("histogram2d",&Histogram_2D_py<double>, "x"_a,"y"_a,"nbins"_a);
	m.def("histogram2d",&Histogram_2D_py<long long int, long long int>, "x"_a,"y"_a,"edges"_a);
	m.def("histogram2d",&Histogram_2D_py<float,float>, "x"_a,"y"_a,"edges"_a);
	m.def("histogram2d",&Histogram_2D_py<double,double>, "x"_a,"y"_a,"edges"_a);

	m.def("histogram2d",&Histogram_2D_Density_py<float>,"x"_a,"y"_a,"nbins"_a,"density"_a);
	m.def("histogram2d",&Histogram_2D_Density_py<double>,"x"_a,"y"_a,"nbins"_a,"density"_a);
	m.def("histogram2d",&Histogram_2D_Density_py<float,float>,"x"_a,"y"_a,"edges"_a,"density"_a);
	m.def("histogram2d",&Histogram_2D_Density_py<double,double>,"x"_a,"y"_a,"edges"_a,"density"_a);
		
	m.def("find_first_in_bin",&Find_First_In_Bin_py<long long int>,"data"_a,"edges"_a);
	m.def("find_first_in_bin",&Find_First_In_Bin_py<float>,"data"_a,"edges"_a);
	m.def("find_first_in_bin",&Find_First_In_Bin_py<double>,"data"_a,"edges"_a);
	
	m.def("find_first_in_bin_2d",&Find_First_In_Bin_2D_py<long long int>,
					"xdata"_a,"ydata"_a,"xedges"_a,"yedges"_a);
	m.def("find_first_in_bin_2d",&Find_First_In_Bin_2D_py<float>,
					"xdata"_a,"ydata"_a,"xedges"_a,"yedges"_a);
	m.def("find_first_in_bin_2d",&Find_First_In_Bin_2D_py<double>,
					"xdata"_a,"ydata"_a,"xedges"_a,"yedges"_a);

	m.def("histogram_displacement2d",&Histogram_And_Displacement_2D_py<long long int>,
					"x"_a,"y"_a,"nbins"_a);
	m.def("histogram_displacement2d",&Histogram_And_Displacement_2D_py<float>,
					"x"_a,"y"_a,"nbins"_a);
	m.def("histogram_displacement2d",&Histogram_And_Displacement_2D_py<double>,
					"x"_a,"y"_a,"nbins"_a);
	
	m.def("histogram_displacement2d",&Histogram_And_Displacement_2D_py<long long int,long long int>
					,"x"_a,"y"_a,"edges"_a);	
	m.def("histogram_displacement2d",&Histogram_And_Displacement_2D_py<float,float>, 
					"x"_a,"y"_a,"edges"_a);	
	m.def("histogram_displacement2d",&Histogram_And_Displacement_2D_py<double,double>,
					"x"_a,"y"_a,"edges"_a);	
	
	m.def("histogram_vectorial_average",&histogram_vectorial_average_py<float>,
					"hist"_a,"row"_a,"col"_a);
	m.def("histogram_vectorial_average",&histogram_vectorial_average_py<double>,
					"hist"_a,"row"_a,"col"_a);
	
	m.def("histogram_nth_order_derivative",&histogram_nth_order_derivative_py<float>,
					"hist_after"_a,"hist_before"_a,"dt"_a,"n"_a,"m"_a);
	m.def("histogram_nth_order_derivative",&histogram_nth_order_derivative_py<double>,
					"hist_after"_a,"hist_before"_a,"dt"_a,"n"_a,"m"_a);

	m.def("digitizer_histogram",&digitizer_histogram_py<uint8_t>,"data"_a.noconvert());
	m.def("digitizer_histogram",&digitizer_histogram_py<uint16_t>,"data"_a.noconvert());

	m.def("digitizer_histogram",&digitizer_histogram_subbyte_py<uint8_t>,
					"data"_a.noconvert(),"nbits"_a);
	m.def("digitizer_histogram",&digitizer_histogram_subbyte_py<uint16_t>,
					"data"_a.noconvert(),"nbits"_a);
	
	m.def("digitizer_histogram2D",&digitizer_histogram2D_py<uint8_t>,
					"data_x"_a.noconvert(),"data_y"_a.noconvert());

	m.def("digitizer_histogram2D",&digitizer_histogram2D_subbyte_py<uint8_t>,
					"data_x"_a.noconvert(),"data_y"_a.noconvert(),"nbits"_a);

	m.def("digitizer_histogram2D",&digitizer_histogram2D_subbyte_py<uint16_t>,
					"data_x"_a.noconvert(),"data_y"_a.noconvert(),"nbits"_a);

	m.def("digitizer_histogram2D_steps",&digitizer_histogram2D_steps_py<uint8_t>,
					"data_x"_a.noconvert(),"data_y"_a.noconvert(),"nbits"_a,"steps"_a);

	py::class_<cdigitizer_histogram2D_steps_py>(m,"Digitizer_Steps2D")
			.def(py::init<uint64_t,uint64_t>())
			.def("getHistogram",&cdigitizer_histogram2D_steps_py::getHistogram)
			.def("getCount",&cdigitizer_histogram2D_steps_py::getCount)
			.def("getSteps",&cdigitizer_histogram2D_steps_py::getSteps)
			.def("getNbits",&cdigitizer_histogram2D_steps_py::getNbits)
			.def("getSize",&cdigitizer_histogram2D_steps_py::getSize)
			.def("resetHistogram",&cdigitizer_histogram2D_steps_py::resetHistogram)
			.def("accumulate",&cdigitizer_histogram2D_steps_py::accumulate<uint8_t>)
			.def("accumulate",&cdigitizer_histogram2D_steps_py::accumulate<uint16_t>);

	py::class_<cHistogram2D_py<double>>(m,"Histogram2D")
			.def(py::init<np_double,np_double,int>())
			.def("getHistogram",&cHistogram2D_py<double>::getHistogram)
			.def("getEdges",&cHistogram2D_py<double>::getEdges)
			.def("accumulate",&cHistogram2D_py<double>::accumulate)
			.def("getCount",&cHistogram2D_py<double>::getCount)
			.def("getNbins",&cHistogram2D_py<double>::getNbins)
			.def("setEdges",&cHistogram2D_py<double>::setEdges)
			.def("resetHistogram",&cHistogram2D_py<double>::resetHistogram);

	py::class_<cHistogram_2D_Density_py<double>>(m,"HistogramDensity2D")
			.def(py::init<np_double,np_double,int>())
			.def("getHistogram",&cHistogram_2D_Density_py<double>::getHistogram)
			.def("getEdges",&cHistogram_2D_Density_py<double>::getEdges)
			.def("accumulate",&cHistogram_2D_Density_py<double>::accumulate)
			.def("getCount",&cHistogram_2D_Density_py<double>::getCount)
			.def("getNbins",&cHistogram_2D_Density_py<double>::getNbins)	
			.def("setEdges",&cHistogram_2D_Density_py<double>::setEdges)
			.def("resetHistogram",&cHistogram_2D_Density_py<double>::resetHistogram);

	py::class_<cHistogram_And_Displacement_2D_py<double>>(m,"HistogramAndDisplacement2D")
			.def(py::init<np_double,np_double,int>())
			.def("getHistogram",&cHistogram_And_Displacement_2D_py<double>::getHistogram)
			.def("getEdges",&cHistogram_And_Displacement_2D_py<double>::getEdges)
			.def("accumulate",&cHistogram_And_Displacement_2D_py<double>::accumulate)
			.def("getCount",&cHistogram_And_Displacement_2D_py<double>::getCount)
			.def("getNbins",&cHistogram_And_Displacement_2D_py<double>::getNbins)
			.def("setEdges",&cHistogram_And_Displacement_2D_py<double>::setEdges)
			.def("resetHistogram",&cHistogram_And_Displacement_2D_py<double>::resetHistogram);

	py::class_<cHistogram_And_Displacement_2D_steps_py<double>>(m,"HistogramAndDisplacementSteps2D")
			.def(py::init<np_double,np_double,int,int>())
			.def("getHistogram",&cHistogram_And_Displacement_2D_steps_py<double>::getHistogram)
			.def("getEdges",&cHistogram_And_Displacement_2D_steps_py<double>::getEdges)
			.def("accumulate",&cHistogram_And_Displacement_2D_steps_py<double>::accumulate)
			.def("getCount",&cHistogram_And_Displacement_2D_steps_py<double>::getCount)
			.def("getNbins",&cHistogram_And_Displacement_2D_steps_py<double>::getNbins)
			.def("setEdges",&cHistogram_And_Displacement_2D_steps_py<double>::setEdges)
			.def("resetHistogram",&cHistogram_And_Displacement_2D_steps_py<double>::resetHistogram);
}

PYBIND11_MODULE(libhistograms, m)
{
	m.doc() = "Computes Histograms and is meant to be a replacement for numpy in some simple scenarios.";
	init_histograms(m);
}
