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

template<
		unsigned int ConvKernelDim,		// e.g 3 for a 3x3 conv kernel (assumed square)
		unsigned int IFMChannels,		// number of input feature maps
		unsigned int IFMDim,			// width of input feature map (assumed square)
		unsigned int OFMChannels,		// number of output feature maps
		unsigned int OFMDim,			// IFMDim-ConvKernelDim+1 or less
		
		unsigned int SIMD, 				// number of SIMD lanes
		unsigned int PE,				// number of PEs
		
		typename TSrcI = Identity,      // redefine I/O interpretation as needed for input activations
		typename TDstI = Identity,		// redefine I/O interpretation as needed for output activations
		typename TWeightI = Identity,	// redefine I/O interpretation as needed for weigths

		int InStreamW, int OutStreamW,  // safely deducible (stream width must be int though!)
		/* hwkim commented
		 * argument에서 InStreamW와 OutStreamW를 알아냄
		 */
		typename TW,   typename TA,  typename R
>
void ConvLayer_Batch(hls::stream<ap_uint<InStreamW>>  &in,
			    hls::stream<ap_uint<OutStreamW>> &out,
			    TW const        &weights,
			    TA const        &activation,
			    unsigned const   reps,
				R const &r) {
#pragma HLS INLINE
  unsigned const MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
  /* hwkim commented
   * kernel 1개 크기 convolution을 한 번에 함?
   */
  unsigned const MatrixH = OFMChannels;
  /* hwkim commented
   * kernel 개수와 동일
   */
  unsigned const InpPerImage = IFMDim*IFMDim*IFMChannels/InStreamW * TSrcI::width;
  /* hwkim commented
   * InPerImage -> input word 개수
   * 	-> TSrcI/InStreamW -> layer 0의 경우 8/24 - 의미는?
   * 	-> InStreamW는 IFMChannels*TSrcI와 같음 (만약 InStreamW가 모든 input channel을 포함한다면)
   * 	-> TSrcI는 data 1개의 width를 의미하는 듯
   */
  WidthAdjustedInputStream <InStreamW, SIMD*TSrcI::width, InpPerImage>  wa_in (in,  reps);
  /* hwkim commented
   *
   * wa_in은 WidthAdjustedInputStream class로써, m_target stream을 가지고 있음
   * 	wa_in class에서 생성자 호출하면서 in의 width(InStreamW)를 SIMD*TSrcI::width로 바꿔서
   * 	m_target stream에 채워줌
   *
   * layer 0의 경우 24-bit -> 24 bit 변환 (SIMD 3 * TSrcI::width 8)
   *
   * InStreamW -> 모든 input channel 다 합친 width?
   * SIMD * TSrcI::width -> 한 번에 처리할 input channel data 개수(SIMD) * data 1개 width
   *
   * template<unsigned IW, unsigned OW, unsigned N>
		 class WidthAdjustedInputStream {
		  hls::stream<ap_uint<OW>>  m_target;
		   public:
		  WidthAdjustedInputStream(hls::stream<ap_uint<IW> >&  source, unsigned const  reps) {
			StreamingDataWidthConverter_Batch<IW, OW, N>(source, m_target, reps);
  }
   */
  WidthAdjustedOutputStream <PE*TDstI::width, OutStreamW,
  	  OFMDim * OFMDim * (OFMChannels / PE)> mvOut (out,  reps);
  /* hwkim commented
   * mvOut은 WidthAdjustedOutputStream class
   * 	WidthAdjustedOutputStream은 ~~InputStream과 달리 값을 채워주지 않고,
   * 	그냥 buffer 할당만 해 줌
   *
   * PE*TDstI::width의 out stream을 OutStream width로 변환하는 것이 아님!!!
   * 	layer 0의 경우,
   * 		PE*TDstI::width = 16*1 = 16-bit
   * 		OutStream = 64-bit (out channel 수)
   *
   * mvOut은 생성자 호출 시 out을 WidthAdjustedOutputStream 내 m_target buffer에 할당만 함
   * 소멸자 호출 시, WidthAdjustedOutputStream mvOut class 내 m_buffer의 width를 OutStreamW로 변환해서
   * 넣어줌
   * 	WidthAdjustedOutputStream mvOut class의 m_buffer는 convolution 연산 결과를 채울 듯
   */

  hls::stream<ap_uint<SIMD*TSrcI::width> > convInp("StreamingConvLayer_Batch.convInp");


  ConvolutionInputGenerator<ConvKernelDim, IFMChannels, TSrcI::width, IFMDim,
			OFMDim, SIMD,1>(wa_in, convInp, reps);
  /* hwkim commented
   * wa_in -> SIMD*TSrcI::width의 width를 갖는 m_target stream을 class member로 갖음
   * convInp -> SIMD*TSrcI::width의 stream
   * reps -> input image 장 수
   *
   * ConvKernelDim -> 3
   */

  Matrix_Vector_Activate_Batch<MatrixW, MatrixH, SIMD, PE, TSrcI, TDstI, TWeightI>
    (static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
     static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut),
     weights, activation, reps* OFMDim * OFMDim, r);
}

#endif
