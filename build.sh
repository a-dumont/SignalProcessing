#!/bin/bash

make -C FFT/
make -C Correlations/

mv FFT/obj/libfft.so SignalProcessing/fft/.
mv Correlations/obj/libcorrelations.so SignalProcessing/correlations/.
