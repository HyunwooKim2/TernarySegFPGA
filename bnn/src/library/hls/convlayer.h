/******************************************************************************
 *  Copyright (c) 2017, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *****************************************************************************/

/******************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *           Thomas B. Preusser <thomas.preusser@utexas.edu>
 *             Marie-Curie Fellow, Xilinx Ireland, Grant Agreement No. 751339
 *           Christoph Doehring <cdoehrin@xilinx.com>
 *
 *  @file convlayer.h
 *
 *  Library of templated HLS functions for BNN deployment.
 *  This file lists a set of convenience funtions used to implement
 *  convolutional layers.
 *
 *****************************************************************************/

#ifndef CONVLAYER_H
#define CONVLAYER_H

#include <ap_int.h>
#include <hls_stream.h>

#include "streamtools.h"
#include "mvau.hpp"

// hwkim modified for debug
//#define ACTIVATION_LOG
#ifdef ACTIVATION_LOG
#include <fstream>
#endif

template<
		unsigned int ConvKernelDim,		// e.g 3 for a 3x3 conv kernel (assumed square)
		unsigned int IFMChannels,		// number of input feature maps
		unsigned int IFMDim,			// width of input feature map (assumed square)
		unsigned int OFMChannels,		// number of output feature maps
		unsigned int OFMDim,			// IFMDim-ConvKernelDim+1 or less
		
		// hwkim added for segmentation
		unsigned int IFMHeight,
		unsigned int OFMHeight,
		unsigned int Stride,

		unsigned int SIMD, 				// number of SIMD lanes
		unsigned int PE,				// number of PEs
		
		typename TSrcI = Identity,      // redefine I/O interpretation as needed for input activations
		typename TDstI = Identity,		// redefine I/O interpretation as needed for output activations
		typename TWeightI = Identity,	// redefine I/O interpretation as needed for weigths

		int InStreamW, int OutStreamW,  // safely deducible (stream width must be int though!)
		typename TW,   typename TA,  typename R
>
void ConvLayer_Batch(hls::stream<ap_uint<InStreamW>>  &in,
			    hls::stream<ap_uint<OutStreamW>> &out,
#ifdef ACTIVATION_LOG
				hls::stream<ap_uint<OutStreamW>> &out_log,
#endif
			    TW const        &weights,
			    TA const        &activation,
			    unsigned const   reps,
				R const &r) {
#pragma HLS INLINE
  unsigned const MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
  unsigned const MatrixH = OFMChannels;

  // hwkim modified for debug - consider fractal
  //unsigned const InpPerImage = IFMDim*IFMDim*IFMChannels/InStreamW * TSrcI::width;
  //unsigned const InpPerImage = (float)(IFMDim*IFMDim*IFMChannels)/InStreamW * TSrcI::width;
  // hwkim modified for segmentation
  unsigned const InpPerImage = (float)(IFMDim*IFMHeight*IFMChannels)/InStreamW * TSrcI::width;

  WidthAdjustedInputStream <InStreamW, SIMD*TSrcI::width, InpPerImage>  wa_in (in,  reps);


  WidthAdjustedOutputStream <PE*TDstI::width, OutStreamW,
  // hwkim modified for segmentation
  	  //OFMDim * OFMDim * (OFMChannels / PE)> mvOut (out,  reps);
  	  OFMDim * OFMHeight * (OFMChannels / PE)> mvOut (out,  reps);
#ifdef ACTIVATION_LOG
  WidthAdjustedOutputStream <PE*TDstI::width, OutStreamW,
  // hwkim modified for segmentation
	  //OFMDim * OFMDim * (OFMChannels / PE)> mvOut_log (out_log,  reps);
  	  OFMDim * OFMHeight * (OFMChannels / PE)> mvOut_log (out_log,  reps);
#endif

  hls::stream<ap_uint<SIMD*TSrcI::width> > convInp("StreamingConvLayer_Batch.convInp");
  // hwkim modified for debug
#ifdef ACTIVATION_LOG
  hls::stream<ap_uint<SIMD*TSrcI::width> > convInp_log("StreamingConvLayer_Batch.convInp_log");
#endif

  ConvolutionInputGenerator
  	  <ConvKernelDim,
	  IFMChannels,
	  TSrcI::width,
	  IFMDim,
	  OFMDim,
  	  OFMHeight,	// hwkim added for segmentation
  	  SIMD,
	  Stride>	//1>	// hwkim modified for segmentation
  (wa_in, convInp,
// hwkim modified for debug
#ifdef ACTIVATION_LOG
					convInp_log,
#endif
					reps);

  // hwkim modified for debug
#ifdef ACTIVATION_LOG
            ofstream conv_in_gen_log_file("conv_in_gen_log.txt");
            if(!conv_in_gen_log_file.is_open()){
            	cout << "conv_in_gen_log_file open error!!" << endl;
              }
            for(int y=0; y<OFMHeight; y++){
            	for(int x=0; x<OFMDim; x++){
            		for(int ky=0; ky<3; ky++){
            			for(int kx=0; kx<3; kx++){;
            				for(int in_ch=0; in_ch<IFMChannels/SIMD; in_ch++){
            					conv_in_gen_log_file << setw(20) << hex << (unsigned int)convInp_log.read() << " ";
            				}
            				conv_in_gen_log_file << " | ";
            			}
            			conv_in_gen_log_file << endl;
            		}
            		conv_in_gen_log_file << "x,y=" << dec << x << "," << y << endl;
            	}
            	conv_in_gen_log_file << endl;
              }
            conv_in_gen_log_file.close();
//            cout << "convInp.size = " << convInp.size() << endl;
#endif

  // hwkim modified for padding
//  Matrix_Vector_Activate_Batch<MatrixW, MatrixH, SIMD, PE, TSrcI, TDstI, TWeightI>
//    (static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
//     static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut),
//     weights, activation, reps* OFMDim * OFMDim, r);
	Matrix_Vector_Activate_Batch_Padding<MatrixW, MatrixH, SIMD, PE, OFMDim,
	// hwkim modified for segmentation
	OFMHeight,

	TSrcI, TDstI, TWeightI>
		(static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
		static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut),
#ifdef ACTIVATION_LOG
		static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut_log),
#endif
		// hwkim modified for segmentation
//		weights, activation, reps* OFMDim * OFMDim, r);
		weights, activation, reps* OFMDim * OFMHeight, r);
}

#endif
