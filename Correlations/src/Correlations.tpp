template <class DatatypeIn, class DatatypeOut>
void autocorrelation(int n, DatatypeIn* in, DatatypeOut* out)
{
	// Compute the FFT
	rFFT(n, in, out);

	// Compute the correlation
	for (int i=0; i<(n/2+1); i++)
	{
		out[i] *= std::conj(out[i]);	
	}
}


template <class DatatypeIn, class DatatypeOut>
void xcorrelation(int n, DatatypeIn* in1, DatatypeIn* in2, DatatypeOut* out)
{
	// Create a temporary buffer
	DatatypeOut* temp_out = (DatatypeOut*) fftw_malloc(sizeof(DatatypeOut)*(n/2+1));
	
	// Compute both FFTs
	rFFT(n, in1, out);
	rFFT(n, in2, temp_out);

	// Compute the correlation
	for (int i=0; i<(n/2+1); i++)
	{
		out[i] *= std::conj(temp_out[i]);	
	}

	// Free the temporary buffer
	fftw_free(temp_out);
}

template <class DatatypeIn, class DatatypeOut>
void autocorrelation_Block(int n, int N, DatatypeIn* in, DatatypeOut* out)
{	
	// Make sure the output is empty
	for (int i=0; i<(N/2+1); i++)
	{
		out[i] = 0;
	}

	int h = n/N;
	// Create a temporary buffer
	DatatypeOut* temp_out = (DatatypeOut*) fftw_malloc(sizeof(DatatypeOut)*h*(N/2+1));

	// Compute and store the FFT
	rFFT_Block(n, N, in, temp_out);

	// Compute the correlation and reduce the blocks
	for (int i=0; i<(h*(N/2+1)); i++)
	{
		out[i%(N/2+1)] += std::norm(temp_out[i]);	
	}

	// Divide by the number of blocks
	for (int i=0; i<(N/2+1); i++)
	{
		out[i] /= h;
	}

	// Free the temporary buffer
	fftw_free(temp_out);
}

template <class DatatypeIn, class DatatypeOut>
void xcorrelation_Block(int n, int N, DatatypeIn* in1, DatatypeIn* in2, DatatypeOut* out)
{	
	// Make sure the output buffer is empty
	for (int i=0; i<(N/2+1); i++)
	{
		out[i] = 0;
	}

	int h = n/N;
	// Create 2 temporary buffers
	DatatypeOut* temp_out1 = (DatatypeOut*) fftw_malloc(sizeof(DatatypeOut)*h*(N/2+1));
	DatatypeOut* temp_out2 = (DatatypeOut*) fftw_malloc(sizeof(DatatypeOut)*h*(N/2+1));
	
	// Compute the FFT of the first input and store in temp_out
	rFFT_Block(n, N, in1, temp_out1);
	rFFT_Block(n, N, in2, temp_out2);
	
	// Compute correlation and reduce the blocks
	for (int i=0; i<(h*(N/2+1)); i++)
	{
		out[i%(N/2+1)] += temp_out1[i]*std::conj(temp_out2[i]);	
	}

	// Divide by the number of blocks
	for (int i=0; i<(N/2+1); i++)
	{
		out[i] /= h;
	}

	//Free the temporary buffers
	fftw_free(temp_out1);
	fftw_free(temp_out2);
}


