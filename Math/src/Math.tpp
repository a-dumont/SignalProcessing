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
DataType sum_pairwise(DataType* in, long int n)
{
	if (n<=128)
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
		else 
		{
			long int N = n-n%8;
			DataType remainder = 0.0;
			DataType out =  0.0;
			DataType res[8] = {};
			for(long int i=0;i<N;i+=8)
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
			out = std::accumulate(res,res+8,remainder)+std::accumulate(in+N,in+n,remainder);
			return out;
		}
	}
	else
	{
		long int N = n-n%128;
		long int m = N/128;
		DataType remainder = 0;
		DataType* out = (DataType*) malloc(sizeof(DataType)*m);
		#pragma omp parallel for
		for(long int j=0;j<m;j++)
		{
			DataType res[8] = {};
			for(long int i=0;i<128;i+=8)
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
			out[j] = std::accumulate(res,res+8,remainder);
		}
		DataType res = sum_pairwise<DataType>(out,m)+std::accumulate(in+N,in+n,remainder);
		free(out);
		return res;
	}
}

template<class DataType>
double variance(DataType* in, long int n)
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
DataType variance_pairwise(DataType* in, long int n)
{
	DataType _mean = sum_pairwise(in,n)/n;
	if (n<=128)
	{
		if(n<8)
		{
			DataType var = 0.0;
			for(int i=0;i<n;i++)
			{
				var += (in[i])*(in[i]);
			}
			return var/n-_mean*_mean;
		}
		else 
		{
			long int N = n-n%8;
			DataType remainder = 0.0;
			DataType out =  0.0;
			DataType res[8] = {};
			for(long int i=0;i<N;i+=8)
			{
				res[0] += (in[i])*(in[i]);
				res[1] += (in[i+1])*(in[i+1]);
				res[2] += (in[i+2])*(in[i+2]);
				res[3] += (in[i+3])*(in[i+3]); 
				res[4] += (in[i+4])*(in[i+4]);
				res[5] += (in[i+5])*(in[i+5]);
				res[6] += (in[i+6])*(in[i+6]); 
				res[7] += (in[i+7])*(in[i+7]);
			}
			out = std::accumulate(res,res+8,remainder);
			for(int i=N;i<n;i++)
			{
				remainder += (in[i])*(in[i]);
			}
			return (out+remainder)/n-_mean*_mean;
		}
	}
	else
	{
		long int N = n-n%128;
		long int m = N/128;
		DataType remainder = 0.0;
		DataType* out = (DataType*) malloc(sizeof(DataType)*m);
		#pragma omp parallel for
		for(long int j=0;j<m;j++)
		{
			DataType res[8] = {};
			for(long int i=0;i<128;i+=8)
			{
				res[0] += (in[128*j+i])*(in[128*j+i]);
				res[1] += (in[128*j+i+1])*(in[128*j+i+1]);
				res[2] += (in[128*j+i+2])*(in[128*j+i+2]);
				res[3] += (in[128*j+i+3])*(in[128*j+i+3]); 
				res[4] += (in[128*j+i+4])*(in[128*j+i+4]);
				res[5] += (in[128*j+i+5])*(in[128*j+i+5]);
				res[6] += (in[128*j+i+6])*(in[128*j+i+6]); 
				res[7] += (in[128*j+i+7])*(in[128*j+i+7]);
			}
			out[j] = std::accumulate(res,res+8,remainder);
		}
		for(int i=N;i<n;i++)
		{
			remainder += (in[i])*(in[i]);
		}
		DataType res = sum_pairwise<DataType>(out,m)+remainder;
		free(out);
		return res/n-_mean*_mean;
	}
}

template<class DataType>
double skewness(DataType* in, long int n)
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

template<class DataType>
DataType skewness_pairwise(DataType* in, long int n)
{
	DataType _mean = sum_pairwise(in,n)/n;
	if (n<=128)
	{
		if(n<8)
		{
			DataType skew = 0.0;
			for(int i=0;i<n;i++)
			{
				skew += (in[i]-_mean)*(in[i]-_mean)*(in[i]-_mean);
			}
			return skew/n;
		}
		else 
		{
			long int N = n-n%8;
			DataType remainder = 0.0;
			DataType out =  0.0;
			DataType res[8] = {};
			for(long int i=0;i<N;i+=8)
			{
				res[0] += (in[i]-_mean)*(in[i]-_mean)*(in[i]-_mean);
				res[1] += (in[i+1]-_mean)*(in[i+1]-_mean)*(in[i+1]-_mean);
				res[2] += (in[i+2]-_mean)*(in[i+2]-_mean)*(in[i+2]-_mean);
				res[3] += (in[i+3]-_mean)*(in[i+3]-_mean)*(in[i+3]-_mean); 
				res[4] += (in[i+4]-_mean)*(in[i+4]-_mean)*(in[i+4]-_mean);
				res[5] += (in[i+5]-_mean)*(in[i+5]-_mean)*(in[i+5]-_mean);
				res[6] += (in[i+6]-_mean)*(in[i+6]-_mean)*(in[i+6]-_mean); 
				res[7] += (in[i+7]-_mean)*(in[i+7]-_mean)*(in[i+7]-_mean);
			}
			out = std::accumulate(res,res+8,remainder);
			for(int i=N;i<n;i++)
			{
				remainder += (in[i]-_mean)*(in[i]-_mean)*(in[i]-_mean);
			}
			return (out+remainder)/n;
		}
	}
	else
	{
		long int N = n-n%128;
		long int m = N/128;
		DataType remainder = 0.0;
		DataType* out = (DataType*) malloc(sizeof(DataType)*m);
		#pragma omp parallel for
		for(long int j=0;j<m;j++)
		{
			DataType res[8] = {};
			for(long int i=0;i<128;i+=8)
			{
				res[0] += (in[128*j+i]-_mean)*(in[128*j+i]-_mean)*(in[128*j+i]-_mean);
				res[1] += (in[128*j+i+1]-_mean)*(in[128*j+i+1]-_mean)*(in[128*j+i+1]-_mean);
				res[2] += (in[128*j+i+2]-_mean)*(in[128*j+i+2]-_mean)*(in[128*j+i+2]-_mean);
				res[3] += (in[128*j+i+3]-_mean)*(in[128*j+i+3]-_mean)*(in[128*j+i+3]-_mean); 
				res[4] += (in[128*j+i+4]-_mean)*(in[128*j+i+4]-_mean)*(in[128*j+i+4]-_mean);
				res[5] += (in[128*j+i+5]-_mean)*(in[128*j+i+5]-_mean)*(in[128*j+i+5]-_mean);
				res[6] += (in[128*j+i+6]-_mean)*(in[128*j+i+6]-_mean)*(in[128*j+i+6]-_mean); 
				res[7] += (in[128*j+i+7]-_mean)*(in[128*j+i+7]-_mean)*(in[128*j+i+7]-_mean);
			}
			out[j] = std::accumulate(res,res+8,remainder);
		}
		for(int i=N;i<n;i++)
		{
			remainder += (in[i]-_mean)*(in[i]-_mean)*(in[i]-_mean);
		}
		DataType res = sum_pairwise<DataType>(out,m)+remainder;
		free(out);
		return res/n;
	}
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