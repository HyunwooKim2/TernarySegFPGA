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
		unsigned int Top,
		unsigned int Bottom,
		unsigned int Left,
		unsigned int Right,
#ifdef ACTIVATION_LOG
		unsigned int LayerCnt,
#endif

		unsigned int SIMD, 				// number of SIMD lanes
		unsigned int PE,				// number of PEs
		
		// hwkim added for batch norm scale
		typename TDstElem,

		typename TSrcI = Identity,      // redefine I/O interpretation as needed for input activations
		typename TDstI = Identity,		// redefine I/O interpretation as needed for output activations
		typename TWeightI = Identity,	// redefine I/O interpretation as needed for weigths

		int InStreamW, int OutStreamW,  // safely deducible (stream width must be int though!)
		typename TW,   typename TA,  typename R
>
void ConvLayer_Batch(hls::stream<ap_uint<InStreamW>>  &in,
			    hls::stream<ap_uint<OutStreamW>> &out,
//#ifdef ACTIVATION_LOG
//				hls::stream<ap_uint<OutStreamW>> &out_log,
//#endif
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
//#ifdef ACTIVATION_LOG
//  WidthAdjustedOutputStream <PE*TDstI::width, OutStreamW,
//  	  // hwkim modified for segmentation
//	  //OFMDim * OFMDim * (OFMChannels / PE)> mvOut_log (out_log,  reps);
//  	  OFMDim * OFMHeight * (OFMChannels / PE)> mvOut_log (out_log,  reps);
//#endif

  hls::stream<ap_uint<SIMD*TSrcI::width> > convInp("StreamingConvLayer_Batch.convInp");

  ConvolutionInputGenerator<ConvKernelDim, IFMChannels, TSrcI::width, IFMDim, OFMDim,
	  IFMHeight, OFMHeight, Top, Bottom, Left, Right,	// hwkim added for segmentation
  	  SIMD,
	  Stride>	//1>	// hwkim modified for segmentation
  (wa_in, convInp, reps);

  // hwkim modified for padding
//  Matrix_Vector_Activate_Batch<MatrixW, MatrixH, SIMD, PE, TSrcI, TDstI, TWeightI>
//    (static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
//     static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut),
//     weights, activation, reps* OFMDim * OFMDim, r);
	Matrix_Vector_Activate_Batch_Padding<MatrixW, MatrixH, SIMD, PE, OFMDim,
		OFMHeight, Top, Bottom, Left, Right,	// hwkim modified for segmentation
#ifdef ACTIVATION_LOG
		LayerCnt, (PE*TDstI::width),
#endif
		// hwkim added for batch norm scale
		TDstElem,

		TSrcI, TDstI, TWeightI>
			(static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
			static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut),
#ifdef ACTIVATION_LOG
//			static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut_log),
#endif
			// hwkim modified for segmentation
//			weights, activation, reps* OFMDim * OFMDim, r);
			weights, activation, reps* OFMDim * OFMHeight, r);
}



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
		unsigned int Top,
		unsigned int Bottom,
		unsigned int Left,
		unsigned int Right,
#ifdef ACTIVATION_LOG
		unsigned int LayerCnt,
#endif

		unsigned int SIMD, 				// number of SIMD lanes
		unsigned int PE,				// number of PEs
		typename TSrcI = Identity,      // redefine I/O interpretation as needed for input activations
		typename TDstI = Identity,		// redefine I/O interpretation as needed for output activations
		typename TWeightI = Identity,	// redefine I/O interpretation as needed for weigths
		int InStreamW, int OutStreamW,  // safely deducible (stream width must be int though!)
		typename TW,   typename TA,  typename R
>
void UpConvLayer_Batch(hls::stream<ap_uint<InStreamW>>  &in,
			    hls::stream<ap_uint<OutStreamW>> &out,
//#ifdef ACTIVATION_LOG
//				hls::stream<ap_uint<OutStreamW>> &out_log,
//#endif
			    TW const        &weights,
			    TA const        &activation,
			    unsigned const   reps,
				R const &r) {
#pragma HLS INLINE
  unsigned const MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
  unsigned const MatrixH = OFMChannels;
  unsigned const InpPerImage = (float)(IFMDim*IFMHeight*IFMChannels)/InStreamW * TSrcI::width;

  WidthAdjustedInputStream <InStreamW, SIMD*TSrcI::width, InpPerImage>  wa_in (in,  reps);
  WidthAdjustedOutputStream <PE*TDstI::width, OutStreamW, OFMDim * OFMHeight * (OFMChannels / PE)> mvOut (out,  reps);
//#ifdef ACTIVATION_LOG
//  WidthAdjustedOutputStream <PE*TDstI::width, OutStreamW, OFMDim * OFMHeight * (OFMChannels / PE)> mvOut_log (out_log,  reps);
//#endif

  hls::stream<ap_uint<SIMD*TSrcI::width> > convInp("StreamingConvLayer_Batch.convInp");

  TConvolutionInputGenerator<ConvKernelDim, IFMChannels, TSrcI::width, IFMDim, OFMDim, IFMHeight, OFMHeight,
  	  //Top, Bottom, Left, Right,
	  SIMD>
  	  (wa_in, convInp, reps);

	Matrix_Vector_Activate_Batch_Skipping<IFMChannels, MatrixH, SIMD, PE, OFMDim,
		OFMHeight, Top, Bottom, Left, Right,	// hwkim modified for segmentation
#ifdef ACTIVATION_LOG
		LayerCnt,
		(PE*TDstI::width),
#endif
		TSrcI, TDstI, TWeightI>
		(static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
		static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut),
//#ifdef ACTIVATION_LOG
//		static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut_log),
//#endif
		weights, activation, reps* OFMDim * OFMHeight, r);
}





#endif
