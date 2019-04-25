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
 *******************************************************************************/

/*******************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *           Thomas B. Preusser <thomas.preusser@utexas.edu>
 *             Marie-Curie Fellow, Xilinx Ireland, Grant Agreement No. 751339
 *           Christoph Doehring <cdoehrin@xilinx.com>
 *
 *  @file mvau.hpp
 *
 *  This file lists a templated funtion used to implement  
 *  Matrix-Vector-Activation Unit
 *
 *  This project has received funding from the European Union's Framework
 *  Programme for Research and Innovation Horizon 2020 (2014-2020) under
 *  the Marie Skłodowska-Curie Grant Agreement No. 751339.
 *
 *******************************************************************************/

#ifndef MVAU_HPP
#define MVAU_HPP

#include "hls_stream.h"

#include "mac.hpp"
#include "interpret.hpp"

template<
  unsigned MatrixW, unsigned MatrixH, unsigned SIMD, unsigned PE,
  typename TSrcI = Identity, typename TDstI = Identity, typename TWeightI = Identity,
  typename TI, typename TO, typename TW, typename TA, typename R
>
/* hwkim commented
 * MatrixW -> ConvKernelDim * ConvKernelDim * IFMChannels -> 3*3*32
 * MatrixH -> OFMChannels
 * TSrcI -> Slice<ap_fixed<8, 1, AP_TRN, AP_SAT>>
 * TDstI -> Identity
 * TWeightI -> Recast<Binary>
 * TI -> 24-bit
 * TO -> 64-bit
 * TW -> SIMD -> 3
 * TA -> 24-bit fixed point
 * R -> ?
 */
void Matrix_Vector_Activate_Batch(hls::stream<TI> & in,
				  hls::stream<TO> &out,
				  TW  const &weights,
				  TA  const &activation,
				  int const  reps,
				  R const &r) {
/* hwkim commented
 * in (hls::stream<ap_uint<SIMD*TSrcI::width>>)
 * 		-> convInp -> sliding 순서로 저장된 input stream
 * out (hls::stream<ap_uint<PE*TDstI::width>>)
 * 		-> mvOut -> WidthAdjustedOutputStream class로 그냥 할당만 된 m_buffer stream
 * weights (BinaryWeights)
 * 		-> BlackBoxJam의 mem init에서 load한 weights memory(not stream)
 * activation (ThresholdsActivation)
 * 		-> BlackBoxJam의 mem init에서 load한 threshold memory(activation function을 위한)
 * reps -> reps*OFDim*OFDim -> 1*30*30
 */
  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  unsigned const  NF = MatrixH / PE;
  /* hwkim commented
   * PE -> output neuron 중 몇 개를 동시에 계산할 것인가
   * NF -> input 가로 한 줄을 PE개 씩 몇 번 해야하는지
   */

  // how many synapse groups each row is split into
  // alternatively: number of horizontal matrix chunks
  unsigned const  SF = MatrixW / SIMD;
  /* hwkim commented
   * SF -> input 채널 방향 한 줄을 SIMD개 씩 몇 번 해야하는지
   */

  // input vector buffers
  TI  inputBuf[SF];
  /* hwkim commented
   * TI -> input bit width -> 24-bit
   * 	-> SIMD개 input과 같음
   * inputBuf[SF] -> input data 한개의 모든 channel 저장하는 memory
   */
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1

  decltype(activation.init(0,0))  accu[PE];
  /* hwkim commented
   * activation.init(0,0)과 동일한 type으로 accu(accumulation buf?) 선언
   * PE 별로 accum 하므로 PE 개 배열 선언
   * activation의 type은 ThresholdsActivation class
   * 	-> layer 0의 경우 24-bit fixed point
   * 	-> 나머지 layer는 16-bit integer
   */
#pragma HLS ARRAY_PARTITION variable=accu complete dim=0

  unsigned  nf   = 0;
  unsigned  sf   = 0;
  unsigned  tile = 0; // invariant: tile = nf*SF + sf

  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelinening the way we want
  unsigned const TOTAL_FOLD = NF * SF;
  for(unsigned  i = 0; i < reps * TOTAL_FOLD; i++) {
#pragma HLS PIPELINE II=1
    TI  inElem;
    if(nf == 0) {
    	/* hwkim commented
    	 * PE 단위로 끝까지 연산 다했을 때만 새로운 input을 read
    	 * 	PE가 output(feature map) xy 방향? c 방향으로 병렬 연산?
    	 * 		-> c 방향일 듯. c 방향은 같은 input에 대해 kernel만 바뀌면 되므로
    	 */
      // read input from stream
      inElem = in.read();
      /* hwkim commented
       * convInp -> sliding 순으로 정렬된 input
       */
      // store in appropriate buffer for reuse
      inputBuf[sf] = inElem;
    }
    else {
      // reuse buffered input
    	/* hwkim commented
    	 * nf가 증가하면 바뀌는 것은 PE개의 kernel만 바뀜
    	 * 같은 input에 대해서 연산하므로 reuse 가능
    	 */
      inElem = inputBuf[sf];
    }

    // Threshold Initialisation
    /* hwkim commented
     * output channel(kernel) 별로 threshold가 다름
     * 	-> SIMD 입장에서는 모두 같은 threshold가 적용됨
     */
    if(sf == 0) {
    	/* hwkim commented
    	 * input 모든 channel에 대해 연산이 끝났으면, accu[] 초기화
    	 */
      for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
    	  /* hwkim commented
    	   * PE 모두 병렬로 실행
    	   */
	    accu[pe] = activation.init(nf, pe);
	    /* hwkim commented
	     * 단순히 TA width의 0을 accu[pe]에 저장
	     * 	-> 즉, 단순히 accu[pe]를 0으로 초기화
	     * 	인자로 들어가는 nf, pe는 의미 없는 듯
	     */
      }
    }

    // compute matrix-vector product for each processing element
    auto const &w = weights.weights(tile);
    /* hwkim commented
     * BinaryWeights class의 member weights(TileIndex class 타입)의
     * 	-> m_par(BinaryWeights class) member에 현재 BinaryWeights 참조자 연결
     * 	-> m_idx member를 tile로 초기화
     */
    for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
    	/* hwkim commented
    	 * PE들 병렬 실행
    	 */
      auto const  wgt = TWeightI()(w[pe]);
      /* hwkim commented
       * auto로 인해 wgt는 TWeightI의 반환 값(w[pe] == weights == BinaryWeights class)으로 type 결정
       * TWeights == Recast<Binary>
       * 	-> Recast class의 ()(w[pe]) operator는 Recast class의 member인 Container class의
       * 	   m_val member를 w[pe]로 초기화
       * 	-> w의 [] operator
       *
       *
       *
       *
       *
       *
       * cf) Identity는 기존 type 유지하는 것인 듯
       */
      auto const  act = TSrcI()(inElem);
      /* hwkim commented
       * TSrcI == Slice<ap_fixed<8, 1, AP_TRN, AP_SAT>>
       * TSrcI()(inElem)
       * 	-> Slice class의 contrainer의 m_val을 inElem으로 채움(type도 동일하게)
       */
      accu[pe] = mac<SIMD>(accu[pe], wgt, act, r);
      /* hwkim commented
       * type
       * 	accu[pe] -> 24-bit fixed (==ThresholdsActivaiton)
       * 	wgt ->
       * 	act -> 24-bit fixed (ThresholdsActivaiton)
       */
    }

    // keep track of which folded synapse/neuron we are processing
    ++tile;
    if(++sf == SF) {
      // produce output and clear accumulators
      auto  outElem = TDstI().template operator()<TO>();
      for (unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
	    outElem[pe] = activation.activate(nf, pe, accu[pe]);
      }

      out.write(outElem);

      // next folded neuron or image
      sf = 0;
      if(++nf == NF) {
	    nf   = 0;
	    tile = 0;
      }
    }
  }
}
#endif
