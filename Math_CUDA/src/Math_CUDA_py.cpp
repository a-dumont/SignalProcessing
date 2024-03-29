#include "Math_CUDA_py.h"

void init_math(py::module &m)
{
	m.def("vAddCUDA",&vector_sum_py<float>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vAddCUDA",&vector_sum_py<double>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vAddCUDA",&vector_sum_py<int8_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vAddCUDA",&vector_sum_py<int16_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vAddCUDA",&vector_sum_py<int32_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vAddCUDA",&vector_sum_py<int64_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vAddCUDA",&vector_sum_py<uint8_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vAddCUDA",&vector_sum_py<uint16_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vAddCUDA",&vector_sum_py<uint32_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vAddCUDA",&vector_sum_py<uint64_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vAddCUDA",&vector_sum_complex_py<float>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vAddCUDA",&vector_sum_complex_py<double>, "In1"_a.noconvert(),"In2"_a.noconvert());
	
	m.def("vProdCUDA",&vector_product_py<float>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vProdCUDA",&vector_product_py<double>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vProdCUDA",&vector_product_py<int8_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vProdCUDA",&vector_product_py<int16_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vProdCUDA",&vector_product_py<int32_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vProdCUDA",&vector_product_py<int64_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vProdCUDA",&vector_product_py<uint8_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vProdCUDA",&vector_product_py<uint16_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vProdCUDA",&vector_product_py<uint32_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vProdCUDA",&vector_product_py<uint64_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vProdCUDA",&vector_product_complex_py<float>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vProdCUDA",&vector_product_complex_py<double>, "In1"_a.noconvert(),"In2"_a.noconvert());
	
	m.def("vDiffCUDA",&vector_diff_py<float>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vDiffCUDA",&vector_diff_py<double>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vDiffCUDA",&vector_diff_py<int8_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vDiffCUDA",&vector_diff_py<int16_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vDiffCUDA",&vector_diff_py<int32_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vDiffCUDA",&vector_diff_py<int64_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vDiffCUDA",&vector_diff_py<uint8_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vDiffCUDA",&vector_diff_py<uint16_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vDiffCUDA",&vector_diff_py<uint32_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vDiffCUDA",&vector_diff_py<uint64_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vDiffCUDA",&vector_diff_complex_py<float>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vDiffCUDA",&vector_diff_complex_py<double>, "In1"_a.noconvert(),"In2"_a.noconvert());
	
	m.def("vDivCUDA",&vector_div_py<float>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vDivCUDA",&vector_div_py<double>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vDivCUDA",&vector_div_py<int8_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vDivCUDA",&vector_div_py<int16_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vDivCUDA",&vector_div_py<int32_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vDivCUDA",&vector_div_py<int64_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vDivCUDA",&vector_div_py<uint8_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vDivCUDA",&vector_div_py<uint16_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vDivCUDA",&vector_div_py<uint32_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vDivCUDA",&vector_div_py<uint64_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vDivCUDA",&vector_div_complex_py<float>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("vDivCUDA",&vector_div_complex_py<double>, "In1"_a.noconvert(),"In2"_a.noconvert());
	/*	
	m.def("mAddCUDA",&matrix_sum_py<float>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mAddCUDA",&matrix_sum_py<double>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mAddCUDA",&matrix_sum_py<int8_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mAddCUDA",&matrix_sum_py<int16_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mAddCUDA",&matrix_sum_py<int32_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mAddCUDA",&matrix_sum_py<int64_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mAddCUDA",&matrix_sum_py<uint8_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mAddCUDA",&matrix_sum_py<uint16_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mAddCUDA",&matrix_sum_py<uint32_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mAddCUDA",&matrix_sum_py<uint64_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mAddCUDA",&matrix_sum_complex_py<float>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mAddCUDA",&matrix_sum_complex_py<double>, "In1"_a.noconvert(),"In2"_a.noconvert());
	
	m.def("mProdCUDA",&matrix_prod_py<float>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mProdCUDA",&matrix_prod_py<double>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mProdCUDA",&matrix_prod_py<int8_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mProdCUDA",&matrix_prod_py<int16_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mProdCUDA",&matrix_prod_py<int32_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mProdCUDA",&matrix_prod_py<int64_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mProdCUDA",&matrix_prod_py<uint8_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mProdCUDA",&matrix_prod_py<uint16_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mProdCUDA",&matrix_prod_py<uint32_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mProdCUDA",&matrix_prod_py<uint64_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mProdCUDA",&matrix_prod_complex_py<float>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mProdCUDA",&matrix_prod_complex_py<double>, "In1"_a.noconvert(),"In2"_a.noconvert());
		
	m.def("mDiffCUDA",&matrix_diff_py<float>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mDiffCUDA",&matrix_diff_py<double>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mDiffCUDA",&matrix_diff_py<int8_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mDiffCUDA",&matrix_diff_py<int16_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mDiffCUDA",&matrix_diff_py<int32_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mDiffCUDA",&matrix_diff_py<int64_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	//m.def("mDiffCUDA",&matrix_diff_py<uint8_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	//m.def("mDiffCUDA",&matrix_diff_py<uint16_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	//m.def("mDiffCUDA",&matrix_diff_py<uint32_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	//m.def("mDiffCUDA",&matrix_diff_py<uint64_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mDiffCUDA",&matrix_diff_complex_py<float>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mDiffCUDA",&matrix_diff_complex_py<double>, "In1"_a.noconvert(),"In2"_a.noconvert());
	
	m.def("mDivCUDA",&matrix_div_py<float>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mDivCUDA",&matrix_div_py<double>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mDivCUDA",&matrix_div_py<int8_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mDivCUDA",&matrix_div_py<int16_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mDivCUDA",&matrix_div_py<int32_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mDivCUDA",&matrix_div_py<int64_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mDivCUDA",&matrix_div_py<uint8_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mDivCUDA",&matrix_div_py<uint16_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mDivCUDA",&matrix_div_py<uint32_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mDivCUDA",&matrix_div_py<uint64_t>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mDivCUDA",&matrix_div_complex_py<float>, "In1"_a.noconvert(),"In2"_a.noconvert());
	m.def("mDivCUDA",&matrix_div_complex_py<double>, "In1"_a.noconvert(),"In2"_a.noconvert());
	*/
}

PYBIND11_MODULE(libmathcuda, m)
{
	m.doc() = "Useful math using CUDA";
	init_math(m);
}
