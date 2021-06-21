#!/bin/bash

make -C FFT/
make -C Correlations/

mv FFT/fft/libfft.so SignalProcessing/fft/.
mv Correlations/correlations/libcorrelations.so SignalProcessing/correlations/.
