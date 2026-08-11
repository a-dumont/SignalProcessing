// Gradient of array dx(t)/dt with arrays x and t
template <class Datatype, class Datatype2>
py::array_t<Datatype, py::array::c_style> gradient_py(
	py::array_t<Datatype, py::array::c_style> py_in1,
	py::array_t<Datatype2, py::array::c_style> py_in2)
{
		// Get buffers from python
		py::buffer_info buf_x = py_in1.request();
		py::buffer_info buf_t = py_in2.request();

		// Check that arrays are 1D
		if ((buf_x.ndim != 1) | (buf_t.ndim != 1)) {
				throw std::runtime_error("U dumbdumb dimension must be 1.");
		}

		// Check that array are the same size
		if (buf_x.size != buf_t.size) {
				throw std::runtime_error("U dumbdumb size must be same.");
		}

		// Array length
		uint64_t n = buf_x.size;

		// Get pointers
		Datatype* x = (Datatype*)buf_x.ptr;
		Datatype2* t = (Datatype2*)buf_t.ptr;

		// Create output array and get pointer
		py::array_t<Datatype, py::array::c_style> result(buf_x.shape);
		Datatype* out = (Datatype*)result.request().ptr;

		// Compute gradient in-place in out
		gradient<Datatype, Datatype2>(n, x, t, out);

		return result;
}

// Gradient dx(t)/dt with constant dt
template <class Datatype, class Datatype2>
py::array_t<Datatype, py::array::c_style> gradient2_py(
	py::array_t<Datatype, py::array::c_style> py_in1, Datatype2 dt)
{
		// Get buffer
		py::buffer_info buf_x = py_in1.request();

		// Check array is 1D
		if (buf_x.ndim != 1) {
				throw std::runtime_error("U dumbdumb dimension must be 1.");
		}

		// Get array size
		uint64_t n = buf_x.size;

		// Get pointer
		Datatype* x = (Datatype*)buf_x.ptr;
		
		// Create output array and get pointer
		py::array_t<Datatype, py::array::c_style> result(buf_x.shape);
		Datatype* out = (Datatype*)result.request().ptr;

		// Compute gradient
		gradient2<Datatype, Datatype2>(n, x, dt, out);

		return result;
}

template <class Datatype>
py::array_t<Datatype, py::array::c_style> finite_difference_coefficients_py(uint64_t M, uint64_t N)
{
		// Allocate memory for coeffs computation
		Datatype* coeff = (Datatype*) malloc((M + 1)*(2*N+1)*(2*N+1)*sizeof(Datatype));
		
		// Ensure memory is all zeros
		std::memset(coeff, 0, (M+1)*(2*N+1)*(2*N+1)*sizeof(Datatype));
		
		// Compute coeffs
		finite_difference_coefficients(M, N, coeff);
		
		// Useful to avoid typing
		N = 2 * N;
		uint64_t n = N + 1;

		// Create output array and get pointer
		py::array_t<Datatype, py::array::c_style> result(n);
		Datatype* out = (Datatype*)result.request().ptr;
		
		// Copy coeffs to out
		std::memcpy(out,coeff+(M*n*n+N*n),n*sizeof(Datatype));

		// Fixes rounding error when coeff is supposed to be 0
		if(M%2 != 0){out[0] *= 0.0;}
		
		// Free coeffs
		free(coeff);
		
		return result;
}

template <class Datatype, class Datatype2>
py::array_t<Datatype, py::array::c_style> nth_order_gradient_py(
	py::array_t<Datatype, py::array::c_style> py_in, Datatype2 dt, uint64_t M, uint64_t N)
{
		// get buffer from python
		py::buffer_info buf_x = py_in.request();

		// Check array is 1D
		if (buf_x.ndim != 1) {
				throw std::runtime_error("U dumbdumb dimension must be 1.");
		}
		
		// Get array size
		uint64_t n = buf_x.size;

		// Get input pointer
		Datatype* x = (Datatype*)buf_x.ptr;
		
		// Create output array and get pointer
		py::array_t<Datatype, py::array::c_style> result(n-2*N);
		Datatype* out = (Datatype*)result.request().ptr;
		//Datatype* out = (Datatype*)malloc(sizeof(Datatype) * (n - 2 * N));

		// Allocate memory and compute coeffs
		Datatype* coeff = (Datatype*) malloc((M+1)*(2*N+1)*(2*N+1)*sizeof(Datatype));
		std::memset(coeff, 0, (M+1)*(2*N+1)*(2*N+1)*sizeof(Datatype));
		finite_difference_coefficients(M, N, coeff);

		// Wipe output memory
		std::memset(out, 0, sizeof(Datatype)*(n-2*N));

		// Compute finite difference
		nth_order_gradient(n, x, dt, out, M, N, coeff);
		
		// Free coeffs
		free(coeff);

		return result;
}

template <class Datatype>
np_uint64 continuous_max_py(py::array_t<Datatype, py::array::c_style> py_in)
{
		// get buffer from python
		py::buffer_info buf_in = py_in.request();
		
		// Check array is 1D
		if (buf_in.ndim != 1) {throw std::runtime_error("U dumbdumb dimension must be 1.");}

		// Create output array and get pointer
		py::array_t<uint64_t, py::array::c_style> result(buf_in.shape);
		uint64_t* out = (uint64_t*) result.request().ptr;

		// Compute
		continuous_max(buf_in.size, (Datatype*) buf_in.ptr, out);

		return result;
}

template <class Datatype>
np_uint64 continuous_min_py(py::array_t<Datatype, py::array::c_style> py_in)
{
		// get buffer from python
		py::buffer_info buf_in = py_in.request();
		
		// Check array is 1D
		if (buf_in.ndim != 1) {throw std::runtime_error("U dumbdumb dimension must be 1.");}

		// Create output array and get pointer
		py::array_t<uint64_t, py::array::c_style> result(buf_in.shape);
		uint64_t* out = (uint64_t*) result.request().ptr;
		
		// Compute
		continuous_min(buf_in.size, (Datatype*) buf_in.ptr, out);
		
		return result;
}

template <class Datatype>
Datatype sum_py(py::array_t<Datatype, py::array::c_style>& py_in1)
{
		py::buffer_info buf1 = py_in1.request();
		uint64_t n = buf1.size;
		Datatype* in = (Datatype*) buf1.ptr;
		return sum_pairwise(n,in);
}

template <class Datatype>
double mean_py(py::array_t<Datatype, py::array::c_style> py_in1)
{
		py::buffer_info buf1 = py_in1.request();
		return (double) sum_pairwise(buf1.size, (Datatype*) buf1.ptr) / buf1.size;
}

template <class Datatype>
Datatype mean_complex_py(py::array_t<Datatype, py::array::c_style> py_in1)
{
		py::buffer_info buf1 = py_in1.request();
		Datatype res = sum_pairwise(buf1.size, (Datatype*) buf1.ptr);
		return Datatype(std::real(res) / buf1.size, std::imag(res) / buf1.size);
}

template <class Datatype>
Datatype variance_py(py::array_t<Datatype, py::array::c_style> py_in1)
{
		py::buffer_info buf1 = py_in1.request();
		Datatype* ptr = (Datatype*) buf1.ptr;
		return variance_pairwise(buf1.size, ptr);
}

template <class Datatype>
Datatype skewness_py(py::array_t<Datatype, py::array::c_style> py_in1)
{
		py::buffer_info buf1 = py_in1.request();
		return skewness_pairwise(buf1.size, (Datatype*) buf1.ptr);
}

template <class Datatype>
Datatype max_py(py::array_t<Datatype, py::array::c_style> py_in1)
{
		py::buffer_info buf1 = py_in1.request();
		Datatype* ptr = (Datatype*) buf1.ptr;
		return max(buf1.size, ptr);
}

template <class Datatype>
Datatype min_py(py::array_t<Datatype, py::array::c_style> py_in1)
{
		py::buffer_info buf1 = py_in1.request();
		Datatype* ptr = (Datatype*) buf1.ptr;
		return min(buf1.size, ptr);
}

template <class Datatype, class Datatype2>
py::array_t<Datatype, py::array::c_style> 
product_py(py::array_t<Datatype, py::array::c_style> py_in1, 
py::array_t<Datatype2, py::array::c_style> py_in2)
{
		// Get buffers from python
		py::buffer_info buf1 = py_in1.request();
		py::buffer_info buf2 = py_in2.request();

		// Check dimensions and size
		if (buf1.ndim != buf2.ndim) {
				throw std::runtime_error("U dumbdumb dimension must be same.");
		}
		if (buf1.size != buf2.size) {
				throw std::runtime_error("U dumbdumb size must be same.");
		}

		// Create output array and get pointer
		py::array_t<Datatype, py::array::c_style> result(buf1.shape);
		Datatype* out = (Datatype*) result.request().ptr;
		
		// Compute
		product(buf1.size, (Datatype*) buf1.ptr, (Datatype2*) buf2.ptr, out);

		return result;
}

template <class Datatype, class Datatype2>
py::array_t<Datatype, py::array::c_style> 
sum_py(py::array_t<Datatype, py::array::c_style> py_in1, 
py::array_t<Datatype2, py::array::c_style> py_in2)
{
		// Get buffers from python
		py::buffer_info buf1 = py_in1.request();
		py::buffer_info buf2 = py_in2.request();

		// Check dimensions
		if (buf1.ndim != buf2.ndim) {
				throw std::runtime_error("U dumbdumb dimension must be same.");
		}
		if (buf1.size != buf2.size) {
				throw std::runtime_error("U dumbdumb size must be same.");
		}
		
		// Create output array and get pointer
		py::array_t<Datatype, py::array::c_style> result(buf1.shape);
		Datatype* out = (Datatype*) result.request().ptr;
	
		// Compute	
		sum(buf1.size, (Datatype*) buf1.ptr, (Datatype2*) buf2.ptr, out);

		return result;
}

template <class Datatype, class Datatype2>
py::array_t<Datatype, py::array::c_style> 
difference_py(py::array_t<Datatype, py::array::c_style> py_in1, 
py::array_t<Datatype2, py::array::c_style> py_in2)
{
		// Get buffers from python
		py::buffer_info buf1 = py_in1.request();
		py::buffer_info buf2 = py_in2.request();

		// Check dimensions
		if (buf1.ndim != buf2.ndim) {
				throw std::runtime_error("U dumbdumb dimension must be same.");
		}
		if (buf1.size != buf2.size) {
				throw std::runtime_error("U dumbdumb size must be same.");
		}
		
		// Create output array and get pointer
		py::array_t<Datatype, py::array::c_style> result(buf1.shape);
		Datatype* out = (Datatype*) result.request().ptr;
		
		// Compute
		difference(buf1.size, (Datatype*) buf1.ptr, (Datatype*) buf2.ptr, out);

		return result;
}

template <class Datatype, class Datatype2>
py::array_t<Datatype, py::array::c_style> 
division_py(py::array_t<Datatype, py::array::c_style> py_in1, 
py::array_t<Datatype2, py::array::c_style> py_in2)
{
		// Get buffers from python
		py::buffer_info buf1 = py_in1.request();
		py::buffer_info buf2 = py_in2.request();

		// Check dimensions
		if (buf1.ndim != buf2.ndim) {
				throw std::runtime_error("U dumbdumb dimension must be same.");
		}
		if (buf1.size != buf2.size) {
				throw std::runtime_error("U dumbdumb size must be same.");
		}

		// Create output array and get pointer
		py::array_t<Datatype, py::array::c_style> result(buf1.shape);
		Datatype* out = (Datatype*) result.request().ptr;
		
		// Compute
		division(buf1.size, (Datatype*) buf1.ptr, (Datatype2*) buf2.ptr, out);

		return result;
}

class DigitizerBlockMaxPy : public DigitizerBlockMax {
	private:
	public:
		DigitizerBlockMaxPy(uint64_t N_in, uint64_t min_size_in,
			uint64_t max_size_in, uint64_t resolution_in)
			: DigitizerBlockMax(N_in, min_size_in, max_size_in, resolution_in)
		{
		}

		template <class Datatype>
		void accumulate_py(py::array_t<Datatype, py::array::c_style> py_in)
		{
				py::buffer_info buf = py_in.request();
				accumulate((Datatype*)buf.ptr);
		}

		py::array_t<uint64_t, py::array::c_style> get_max_hists_py()
		{
				uint64_t* out = (uint64_t*)malloc(sizeof(uint64_t) * hist_size);
				#pragma omp parallel for
				for (uint64_t i = 0; i < hist_size; i++) {
						out[i] = max_hists[i];
				}
				std::vector<uint64_t> out_size = 
				{ (uint64_t)(log2(n_max / n_min) + 1), (uint64_t) (1<<resolution) };
				py::capsule free_when_done(out, free);
				return py::array_t<uint64_t, py::array::c_style>(
					out_size,
					{ (1 << resolution) * sizeof(uint64_t), sizeof(uint64_t) },
					out,
					free_when_done);
		}
};

class DigitizerBlockMinPy : public DigitizerBlockMin {
	private:
	public:
		DigitizerBlockMinPy(uint64_t N_in, uint64_t min_size_in,
			uint64_t max_size_in, uint64_t resolution_in)
			: DigitizerBlockMin(N_in, min_size_in, max_size_in, resolution_in)
		{
		}

		template <class Datatype>
		void accumulate_py(py::array_t<Datatype, py::array::c_style> py_in)
		{
				py::buffer_info buf = py_in.request();
				accumulate((Datatype*)buf.ptr);
		}

		py::array_t<uint64_t, py::array::c_style> get_min_hists_py()
		{
				uint64_t* out = (uint64_t*)malloc(sizeof(uint64_t) * hist_size);
				#pragma omp parallel for
				for (uint64_t i = 0; i < hist_size; i++) {
						out[i] = min_hists[i];
				}
				std::vector<uint64_t> out_size = 
				{ (uint64_t)(log2(n_max / n_min) + 1), (uint64_t) (1<<resolution) };
				py::capsule free_when_done(out, free);
				return py::array_t<uint64_t, py::array::c_style>(
					out_size,
					{ (1 << resolution) * sizeof(uint64_t), sizeof(uint64_t) },
					out,
					free_when_done);
		}
};

class DigitizerBlockMinMaxPy : public DigitizerBlockMinMax {
	private:
	public:
		DigitizerBlockMinMaxPy(uint64_t N_in, uint64_t min_size_in,
			uint64_t max_size_in, uint64_t resolution_in)
			: DigitizerBlockMinMax(N_in, min_size_in, max_size_in, resolution_in)
		{
		}

		template <class Datatype>
		void accumulate_py(py::array_t<Datatype, py::array::c_style> py_in)
		{
				py::buffer_info buf = py_in.request();
				accumulate((Datatype*)buf.ptr);
		}

		py::array_t<uint64_t, py::array::c_style> get_min_hists_py()
		{
				uint64_t* out = (uint64_t*)malloc(sizeof(uint64_t) * hist_size);
				#pragma omp parallel for
				for (uint64_t i = 0; i < hist_size; i++) {
						out[i] = min_hists[i];
				}
				std::vector<uint64_t> out_size = 
				{ (uint64_t)(log2(n_max / n_min) + 1), (uint64_t) (1<<resolution) };
				py::capsule free_when_done(out, free);
				return py::array_t<uint64_t, py::array::c_style>(
					out_size,
					{ (1 << resolution) * sizeof(uint64_t), sizeof(uint64_t) },
					out,
					free_when_done);
		}
		py::array_t<uint64_t, py::array::c_style> get_max_hists_py()
		{
				uint64_t* out = (uint64_t*)malloc(sizeof(uint64_t) * hist_size);
				#pragma omp parallel for
				for (uint64_t i = 0; i < hist_size; i++) {
						out[i] = max_hists[i];
				}
				std::vector<uint64_t> out_size = 
				{ (uint64_t)(log2(n_max / n_min) + 1), (uint64_t) (1<<resolution) };
				py::capsule free_when_done(out, free);
				return py::array_t<uint64_t, py::array::c_style>(
					out_size,
					{ (1 << resolution) * sizeof(uint64_t), sizeof(uint64_t) },
					out,
					free_when_done);
		}
};
