/******************************************************************************
 *  Copyright (c) 2016, Xilinx, Inc.
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
 
/*****************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *           Thomas B. Preusser <thomas.preusser@utexas.edu>
 *             Marie-Curie Fellow, Xilinx Ireland, Grant Agreement No. 751339
 *           Christoph Doehring <cdoehrin@xilinx.com>
 *
 *  @file fclayer.h
 *
 *  Library of templated HLS functions for BNN deployment. 
 *  This file lists a set of convenience funtions used to implement fully 
 *  connected layers
 *
 *****************************************************************************/
 
#ifndef FCLAYER_H
#define FCLAYER_H

#include <ap_int.h>
#include <hls_stream.h>

#include "streamtools.h"
#include "mvau.hpp"

template<
  unsigned int MatrixW, unsigned int MatrixH, // geometry must be specified
  unsigned int SIMD,    unsigned int PE,

  typename TSrcI = Identity,      // redefine I/O interpretation as needed
  typename TDstI = Identity,
  typename TWeightI = Identity,	  // redefine I/O interpretation as needed for weigths

  int InStreamW, int OutStreamW,  // safely deducible (stream width must be int though!)
  typename TW,   typename TA, typename R
>
/* hwkim commented
 * MatrixW -> 256 -> input channel 수
 * 		layer 0의 경우에, 3 x 3 x input channel 수 였음
 * 		fc layer는 (1 x 1 x) 256
 * MatrixH -> 512 -> output channel 수
 * SIMD -> 4 -> 한 번에 처리할 input channel 개수
 * PE -> 1 -> 한 번에 처리할 kernel(output channel) 개수
 *
 * TSrcI -> Recast<XnorMul>
 * TDstI -> Identity 또는 ap_unit<16>(마지막 layer)
 *
 * InStreamW -> 256-bit -> # of input channel
 * OutStreamW -> 64-bit
 * 		# of output channel? -> X
 * 		output stream의 width
 *
 * TW -> BinaryWeights class
 * TA -> ThresholdsActivation class
 * R -> LUT
 */
void StreamingFCLayer_Batch(hls::stream<ap_uint<InStreamW>>  &in,
			    hls::stream<ap_uint<OutStreamW>> &out,
			    TW const        &weights,
			    TA const        &activation,
			    unsigned const   reps,
				R const &r) {
#pragma HLS INLINE
  unsigned const  InpPerImage = MatrixW / InStreamW * TSrcI::width;
  /* hwkim commented
   * TSrcI::width -> 1-bit(XnorMul)
   * InpPerImage -> 결국 1
   * convolution layer(layer 0)에서는
   * 	InpPerImage = IFMDim*IFMDim*(IFMChannels/InStreamW * TSrcI::width);
   * 		괄호 안은 결국 channel 방향으로 input word가 몇 개 있느냐
   * 		channel 방향으로 예를 들어 1-bit(TSrcI::Width) 64(IFMChannels)개가 있어도,
   * 		InStreamW가 32라면 input word는 2개임
   * 	즉, InpPerImage는 image 1장에 input "word"가 몇 개 있느냐
   * fc layer에서는 IFMDim=1이므로 IFMChannels/InStreamW*TSrcI::width
   * 	fc layer에서는 IFMChannels==MatrixW
   */
  unsigned const  OutPerImage = MatrixH / PE;
  /* hwkim commented
   * PE가 많아지면, OutPerImage가 줄어듬?
   * PE개를 병렬로 하기 때문에, PE개의 output이 동시에 나오므로
   * X PE만큼 output stream width가 늘어나고, OutPerImage는 줄어듦?
   */

  WidthAdjustedInputStream <InStreamW, SIMD*TSrcI::width, InpPerImage>  wa_in (in,  reps);
  /* hwkim commented
   * <IW, OW, N>
   * IW -> input stream의 width
   * OW -> SIMD로 한 번에 처리할 input의 width
   * N -> image 당 input word 개수
   * 생성자 호출될 때 width 변경하여 m_target에 채움
   */
  WidthAdjustedOutputStream<PE*TDstI::width,  OutStreamW, OutPerImage>  wa_out(out, reps);
  /* hwkim commented
   * <IW, OW, N>
   * IW -> input stream width
   * 	class 내 m_buffer memory에 input 할당
   * OW -> output stream width
   * 	class 내 m_target 참조자에 할당
   * 	wa_out(out, rep) argument(초기값) out stream에 참조자를 연결
   * 	소멸자 호출될 때, stream width 변경
   * N -> OW width로 변환할 word 개수인 듯?
   */

  /* hwkim commented
   * 결국 위의 width adjust는 아래 matrix vector activate batch할 때의
   * 	width로 바꿔주는 것
   * 	in -> SIMD*TSrcI::width
   * 		layer 6의 경우
   * 			256-bit에서 4-bit(SIMD 4 * XnorMul 1) stream으로 바꿈
   *				64개를 stream으로 잇는 것
   *			in stream width를 SIMD개만큼 변경하는 것은,
   *				stream에서 SIMD개 input 동시에 읽어오기 위해
   * 	out -> PE*TDstI::width
   * 		layer 6의 경우
   * 			64-bit에서 1-bit stream으로 바꿈
   * 			Matrix_Vector_Activate_Batch 함수에서 output을
   * 			PE*TDstI::width==1-bit으로 쓰기 때문
   * 			WidthAdjustedOutputStream 소멸자 호출될 때,
   * 				1-bit(PE*TDstI::width) -> 64-bit(OutStreamW)으로 변환해줌
   */

  Matrix_Vector_Activate_Batch<MatrixW, MatrixH, SIMD, PE, TSrcI, TDstI, TWeightI>
    (static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(wa_in),
     static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (wa_out),
     weights, activation, reps, r);
  /* hwkim commented
   * 여기서 WidthAdjustedInputStream class wa_in의 operator&가 호출되고,
   * 	WidthAdjustedOutputSream class wa_out의 operator&가 호출되어,
   * 	wa_in과 wa_out의 m_target 및 m_buffer가 연결되는 듯
   * reps가 1로, Matrix_Vector_Activate_Batch 수행 시,
   * 	1 pixel에 대해서만 수행
   */
}

#endif
