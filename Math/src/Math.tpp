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
	double alpha[2*N+1];//double* alpha = (double*) malloc((2*N+1)*sizeof(double));
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
	//free(alpha);
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
	free(coeff-(M*(2*N+1)*(2*N+1)+2*N*(2*N+1)));
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
void continuous_max(long int* out, DataType* in, int n)
{
	out[0] = 0;
	for(long int i=1;i<n;i++)
	{
		if(in[i] > in[out[i-1]])
		{
			out[i] = i;
		}
		else 
		{
			out[i] = out[i-1];	
		}
	}
}

template<class DataType>
void continuous_min(long int* out, DataType* in, int n)
{
	out[0] = 0;
	for(long int i=1;i<n;i++)
	{
		if(in[i] < in[out[i-1]])
		{
			out[i] = i;
		}
		else 
		{
			out[i] = out[i-1];	
		}
	}
}

template<class DataType>
DataType sum_pairwise(DataType* in, int n)
{
	if(n<8)
	{
		DataType res = 0.0;
		for(int i=0;i<n;i++)
		{
			res += in[i];
		}
		return res;
	}
	else if (n<=8096)
	{
		int N = n-n%128;
		int m = N/128;
		DataType res[8] = {};
		DataType out = (DataType) 0;
		for(int j=0;j<m;j++)
		{
			std::memset(res,0,8*sizeof(DataType));
			for(int i=0;i<128;i+=8)
			{
				res[0] += in[128*j+i];
				res[1] += in[128*j+i+1];
				res[2] += in[128*j+i+2]; 
				res[3] += in[128*j+i+3]; 
				res[4] += in[128*j+i+4]; 
				res[5] += in[128*j+i+5]; 
				res[6] += in[128*j+i+6]; 
				res[7] += in[128*j+i+7]; 
			}
			out += std::accumulate(res,res+8,0.0);
		}
		return std::accumulate(res,res+8,0.0)+std::accumulate(in+N,in+n,0.0);
	}
	else
	{
		int N = n-n%128;
		DataType res[8] = {};
		#pragma omp parallel for default(shared) reduction(+:res[:8])
		for(int i=0;i<N;i+=8)
		{
			res[0] += in[i];
			res[1] += in[i+1];
			res[2] += in[i+2]; 
			res[3] += in[i+3]; 
			res[4] += in[i+4]; 
			res[5] += in[i+5]; 
			res[6] += in[i+6]; 
			res[7] += in[i+7]; 
		}
		return std::accumulate(res,res+8,0.0)+std::accumulate(in+N,in+n,0.0);
	}
}

template<class DataType>
DataType sum_pairwise_complex(DataType* in, int n)
{
	return DataType (sum_pairwise(std::real(in),n),sum_pairwise(std::imag(in),n));
}

template<class DataType>
double variance(DataType* in, int n)
{
	double var = 0;
	DataType _mean = sum_pairwise(in,n)/n;
	#pragma omp parallel for default(shared) reduction(+:var)
	for(int i=0;i<n;i++)
	{
		var += (in[i]-_mean)*(in[i]-_mean);
	}
	return var/n;
}

template<class DataType>
double skewness(DataType* in, int n)
{
	double poisson = 0;
	DataType _mean = sum_pairwise(in,n)/n;
	#pragma omp parallel for default(shared) reduction(+:poisson)
	for(int i=0;i<n;i++)
	{
		poisson += (in[i]-_mean)*(in[i]-_mean)*(in[i]-_mean);
	}
	return poisson/n;
}

template<class DataType, class DataType2>
void product(DataType* in1, DataType* in2, DataType2* out, int n)
{
	#pragma omp parallel for
	for(int i=0;i<n;i++)
	{
		out[i] = in1[i]*in2[i];
	}
}

template<class DataType, class DataType2>
void sum(DataType* in1, DataType* in2, DataType2* out, int n)
{
	#pragma omp parallel for
	for(int i=0;i<n;i++)
	{
		out[i] = in1[i]+in2[i];
	}
}

template<class DataType, class DataType2>
void difference(DataType* in1, DataType* in2, DataType2* out, int n)
{
	#pragma omp parallel for
	for(int i=0;i<n;i++)
	{
		out[i] = in1[i]-in2[i];
	}
}

template<class DataType, class DataType2>
void division(DataType* in1, DataType* in2, DataType2* out, int n)
{
	#pragma omp parallel for
	for(int i=0;i<n;i++)
	{
		out[i] = in1[i]/in2[i];
	}
}

template<class DataType>
DataType max(DataType* in, int n)
{
	DataType _max = in[0];
	#pragma omp parallel for default(shared) reduction(max:_max)
	for(int i=1;i<n;i++)
	{
		_max = _max > in[i] ? _max : in[i];
	}
	return _max;
}

template<class DataType>
DataType min(DataType* in, int n)
{
	DataType _min = in[0];
	#pragma omp parallel for default(shared) reduction(min:_min)
	for(int i=1;i<n;i++)
	{
		_min = _min < in[i] ? _min : in[i];
	}
	return _min;
}