template<class DataType>
void gradient(int n, DataType* x, DataType* t, DataType* out)
{
	out[0] = (x[1]-x[0])/(t[1]-t[0]);
	out[n-1] = (x[n-1]-x[n-2])/(t[n-1]-t[n-2]);
	for (int i=1; i<(n-1); i++)
	{
			DataType hd = t[i+1]-t[i];
			DataType hs = t[i]-t[i-1];
			out[i] = (hs*hs*x[i+1]+(hd*hd-hs*hs)*x[i]-hd*hd*x[i-1])/(hs*hd*(hd+hs));
	}
}

template<class DataType, class DataType2>
void gradient(int n, DataType* x, DataType2 dt, DataType* out)
{
	DataType2 h = 1/(2*dt);
	out[0] = 2*h*(x[1]-x[0]);
	out[n-1] = 2*h*(x[n-1]-x[n-2]);
	for (int i=1; i<(n-1); i++)
	{
			out[i] = h*(x[i+1]-x[i-1]);
	}
}

template<class DataType>
double* finite_difference_coefficients(DataType M, int N)
{
	double* alpha = (double*) malloc((2*N+1)*sizeof(double));
	N = 2*N;
	double* coeff = (double*) malloc((M+1)*(N+1)*(N+1)*sizeof(double));
	std::memset(coeff,0,(M+1)*(N+1)*(N+1)*sizeof(double));
	alpha[0] = 0;
	alpha[1] = 1;
	double a; double b; double c;
	for(int i=2;i<(N+1);i++)
	{
		if(alpha[i-1] > 0)
		{
			alpha[i] = -alpha[i-1];
		}
		else
		{
			alpha[i] = -1*alpha[i-1]+1;
		}
	}
	coeff[0] = 1;
	a = 1;
	for(int n=1;n<(N+1);n++)
	{
		b = 1;
		for(int v=0;v<n;v++)
		{
			c = alpha[n]-alpha[v];
			b = b*c;
			for(int m=0;m<(std::min(M,n)+1);m++)
			{
				if (m != 0)
				{
					coeff[m*(N+1)*(N+1)+n*(N+1)+v] = (alpha[n]*coeff[m*(N+1)*(N+1)+(n-1)*(N+1)+v]-m*coeff[(m-1)*(N+1)*(N+1)+(n-1)*(N+1)+v])/c;
				}
				else
				{
					coeff[m*(N+1)*(N+1)+n*(N+1)+v] = (alpha[n]*coeff[m*(N+1)*(N+1)+(n-1)*(N+1)+v])/c;
				}
			}
		}
		for(int m=0;m<(std::min(M,n)+1);m++)
		{
			if (m != 0)
			{
				coeff[m*(N+1)*(N+1)+n*(N+1)+n] = a/b*(m*coeff[(m-1)*(N+1)*(N+1)+(n-1)*(N+1)+(n-1)]-alpha[n-1]*coeff[(m)*(N+1)*(N+1)+(n-1)*(N+1)+(n-1)]);
			}
			else
			{
				coeff[m*(N+1)*(N+1)+n*(N+1)+n] = -a/b*(alpha[n-1]*coeff[(m)*(N+1)*(N+1)+(n-1)*(N+1)+(n-1)]);
			}
		}
		a = b;
	}
	free(alpha);
	return coeff;
}

template<class DataType>
void nth_order_gradient(int n, DataType* x, DataType dt, DataType* out,int M, int N)
{
	double* coeff = finite_difference_coefficients(M,N)+(M*(2*N+1)*(2*N+1)+2*N*(2*N+1));
	double norm = 1/dt;
	int k;
	for(int i=N;i<(n-N);i++)
	{
		out[i-N] = coeff[0]*x[i];
		k = 1;
		for(int j=0;j<N;j++)
		{
			out[i-N] += coeff[k]*x[i+j+1]+coeff[k+1]*x[i-(j+1)];
			k += 2;
		}
		for(int l=0;l<M;l++){out[i-N] *= norm;}
	}	
}

template<class DataType>
void rolling_average(int n, DataType* in, DataType* out, int size)
{	
	std::memset(out,0,(n-size)*sizeof(DataType));
	DataType norm = 1.0/size;
	for(int i=0;i<(n-size+1);i++)
	{	
		for(int j=0;j<size;j++)
		{
				out[i] += in[i+j];
		}
		out[i] *= norm;
	}
}

template<class DataType>
void histogram_vectorial_average(int nbins, DataType* hist, DataType* out, int row, int col)
{
	for(int i=0;i<nbins;i++)
	{
		for(int j=0;j<nbins;j++)
		{
			out[0] += hist[i*nbins+j]*(i-row);
			out[1] += hist[i*nbins+j]*(j-col);
		}
	}
}

template<class DataType> 
void inverse_probability2D(int nbins, DataType* in, DataType* out,DataType* P)
{
		# pragma omp parallel for
		for(int i=0;i<(nbins*nbins);i++)
		{
			double temp = 0;
			for(int j=0;j<(nbins*nbins);j++)
			{	
				temp += in[j*(nbins*nbins)+i]*P[j];
				out[i*(nbins*nbins)+j] = in[j*(nbins*nbins)+i]*P[j];
			}
			if(temp != 0.0)
			{
				# pragma omp parallel for
				for(int j=0;j<(nbins*nbins);j++)
				{	
					out[i*(nbins*nbins)+j] *= 1/temp;
				}
			}
		}
}
