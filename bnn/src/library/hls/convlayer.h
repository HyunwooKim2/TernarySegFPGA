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
		unsigned int IFMHeight,	// hwkim added for segmentation
		unsigned int OFMHeight,	// hwkim added for segmentation
		unsigned int Stride,	// hwkim added for segmentation
		unsigned int Top,	// hwkim added for segmentation
		unsigned int Bottom,	// hwkim added for segmentation
		unsigned int Left,	// hwkim added for segmentation
		unsigned int Right,	// hwkim added for segmentation
#ifdef ACTIVATION_LOG
		unsigned int LayerCnt,
#endif
		unsigned int SIMD,	// number of SIMD lanes
		unsigned int PE,		// number of PEs
		typename TDstElem,	// hwkim added for batch norm scale
		typename TSrcI = Identity,      // redefine I/O interpretation as needed for input activations
		typename TDstI = Identity,		// redefine I/O interpretation as needed for output activations
		typename TWeightI = Identity,	// redefine I/O interpretation as needed for weigths
		int InStreamW, int OutStreamW,  // safely deducible (stream width must be int though!)
//		int InMaskStreamW,	// hwkim added for ternary
		typename TW,
//		typename TM,	// hwkim added for ternary
		typename TA,
		typename R>
void ConvLayer_Batch(
				hls::stream<ap_uint<InStreamW>> &in,
//				hls::stream<ap_uint<InMaskStreamW>> &in_mask,	// hwkim added for ternary
				hls::stream<ap_uint<OutStreamW>> &out,
				TW const &weights,
//				TM const &wmasks,	// hwkim added for ternary
				TA const &activation,
				unsigned const   reps,
				R const &r
) {
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

  hls::stream<ap_uint<SIMD*TSrcI::width> > convInp("StreamingConvLayer_Batch.convInp");

  // hwkim added for ternary
//  WidthAdjustedInputStream <InMaskStreamW, SIMD, InpPerImage>  wa_in_mask (in_mask,  reps);
//  hls::stream<ap_uint<SIMD>> convInp_mask("StreamingConvLayer_Batch.convInp_mask");

  ConvolutionInputGenerator<ConvKernelDim, IFMChannels, TSrcI::width, IFMDim, OFMDim,
	  IFMHeight, OFMHeight, Top, Bottom, Left, Right,	// hwkim added for segmentation
  	  SIMD,
	  Stride>	//1>	// hwkim modified for segmentation
  	  	  (wa_in,
//		  wa_in_mask,	// hwkim added for ternary
		  convInp,
//		  convInp_mask,	// hwkim added for ternary
		  reps);

  // hwkim added for ternary
	hls::stream<ap_uint<SIMD*TSrcI::width>> packed_input[PE];
#pragma HLS STREAM variable=packed_input //depth=256
	hls::stream<ap_uint<SIMD>> packed_weight[PE];
#pragma HLS STREAM variable=packed_weight //depth=256
  nonzero_activation_weight_stream_gen<MatrixW, MatrixH, SIMD, PE, OFMDim,
  	  OFMHeight,
  	  Top, Bottom, Left, Right>
  	  (
		  static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
		  weights,
		  static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>*>(packed_input),
		  static_cast<hls::stream<ap_uint<SIMD>>*>(packed_weight),
		  reps * OFMDim * OFMHeight
  	  );


  // hwkim modified for padding
  /*
  Matrix_Vector_Activate_Batch<MatrixW, MatrixH, SIMD, PE, TSrcI, TDstI, TWeightI>
    (static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
     static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut),
     weights, activation, reps* OFMDim * OFMDim, r);
     */
  // hwkim modified for ternary
  /*
	Matrix_Vector_Activate_Batch_Padding<MatrixW, MatrixH, SIMD, PE, OFMDim,
		OFMHeight, Top, Bottom, Left, Right,	// hwkim modified for segmentation
#ifdef ACTIVATION_LOG
		LayerCnt, (PE*TDstI::width),
#endif
		TDstElem,	// hwkim added for batch norm scale
		TSrcI, TDstI, TWeightI>
			(static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
			static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>(mvOut),
			weights,
			activation,
			reps* OFMDim * OFMHeight, r);	//reps* OFMDim * OFMDim, r);	// hwkim modified for segmentation
			*/
	Matrix_Vector_Activate_Batch_SkipSeparately<MatrixW, MatrixH, SIMD, PE, OFMDim,
		OFMHeight, Top, Bottom, Left, Right,	// hwkim modified for segmentation
#ifdef ACTIVATION_LOG
		LayerCnt, (PE*TDstI::width),
#endif
		TDstElem,	// hwkim added for batch norm scale
		TSrcI, TDstI, TWeightI>
			(static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
//			static_cast<hls::stream<ap_uint<SIMD>>&>(convInp_mask),	// hwkim added for ternary
			static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>(mvOut),

			// hwkim modifeid for ternary
			//weights,
			static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>*>(packed_input),
			static_cast<hls::stream<ap_uint<SIMD>>*>(packed_weight),
			//wmasks,	// hwkim added for ternary

			activation,
			reps * OFMDim * OFMHeight, r);	//reps* OFMDim * OFMDim, r);	// hwkim modified for segmentation
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
		weights, activation, reps* OFMDim * OFMHeight, r);
}

#endif
