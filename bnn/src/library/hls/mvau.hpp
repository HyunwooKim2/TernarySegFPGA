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

// hwkim modified for debug
//#define ACTIVATION_LOG
#ifdef ACTIVATION_LOG
#include <fstream>
extern int weighted_layer_cnt;
#endif

template<
  unsigned MatrixW, unsigned MatrixH, unsigned SIMD, unsigned PE,
  typename TSrcI = Identity, typename TDstI = Identity, typename TWeightI = Identity,
  typename TI, typename TO, typename TW, typename TA, typename R
>
/* hwkim commented
 * 	MatrixW -> 3x3x3
 * 		ConvKernelDim * ConvKernelDim * IFMChannels
 *	MatrixH -> 64
 *		OFMChannels -> output neuron channels
 *	SIMD -> 3
 *	PE -> 16
 *	TSrcI -> 8-bit
 *	TDstI -> 1-bit
 *	TWeightI -> 1-bit
 *	TI -> 24-bit
 *		static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp)
 *	TO -> 16-bit
 *		static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut)
 *	TW -> BinaryWeights<L0_SIMD, L0_PE, L0_WMEM>
 *		해당 class type 임
 *	TA -> ThresholdsActivation<L0_TMEM, L0_PE, L0_API, ap_fixed<24, 16>, ap_uint<L0_API> >
 *		해당 class type 임
 *	R -> LUT
*/
void Matrix_Vector_Activate_Batch(hls::stream<TI> & in,
				  hls::stream<TO> &out,
				  TW  const &weights,
				  TA  const &activation,
				  int const  reps,
				  R const &r) {
/* hwkim commented
 * in (stream)
 * 		-> hls::stream<ap_uint<SIMD*TSrcI::width>>
 * 		-> convInp -> sliding 순서로 저장된 input stream
 * out (stream)
 * 		-> hls::stream<ap_uint<PE*TDstI::width>>
 * 		-> mvOut -> WidthAdjustedOutputStream class로 그냥 할당만 된 m_buffer stream
 * weights (stream)
 * 		-> BinaryWeights
 * 		-> BlackBoxJam의 mem init에서 load한 weights memory(not stream)
 * 		-> weight 값이 저장되어 있음
 * activation (stream)
 * 		-> ThresholdsActivation
 * 		-> BlackBoxJam의 mem init에서 load한 threshold memory(activation function을 위한)
 * 		-> threshold 값이 저장되어 있음
 * reps -> reps*OFDim*OFDim -> 1*30*30
 */
  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  unsigned const  NF = MatrixH / PE;
  /* hwkim commented
   * MatrixH -> output channel 수
   * layer 0(convolution layer)의 경우
   * 	NF = 64 / 16 = 4;
   * 	즉, 16개 PE로 4번 수행해야 output channel 64개를 완료
   * fc layer들의 경우,
   * 	PE가 1 or 4
   * 	왜 굳이 PE를 작게 설정했는지?
   * 	PE가 1인 경우, NF가 256 or 512정도로 많음
   * NF = neuron fold
   */

  // how many synapse groups each row is split into
  // alternatively: number of horizontal matrix chunks
  unsigned const  SF = MatrixW / SIMD;
  /* hwkim commented
   * MatrixW
   * 	convolution layer의 경우 ->  ConvKernelDim * ConvKernelDim * IFMChannels;
   * 	fc layer의 경우 -> input channel 수
   * layer 0(convolution)의 경우
   * 	SF = 3x3x3/3 = 9 = kernel size x input channel # / SIMD
   * 	3x3 kernel을 한 방에 하는 것이 아님
   * 		-> pipeline으로 처리
   * fc layer들의 경우, SIMD가 1 or 4 or 8
   * 	SIMD 1 -> SF = 512/1 = 512
   * 	SIMD 1개로 512번 연산함...
   * SF = synapse fold
   */

  // input vector buffers
  TI  inputBuf[SF];
  /* hwkim commented
   * memory (for reuse)
   * layer 0의 경우
   * 	TI -> 24-bit
   * 	SF -> 3x3
   * 	inputBuf[9]는 3x3 kernel에 해당하는 각 input(3-ch)을 가지고 있음
   * fc layer들의 경우
   * 	TI -> 1-bit
   * 	SF -> input channel 수 / SIMD -> ex) 64
   * 즉, inputBuf의 다음 element는 다음 SIMD(sf) 계산을 위한
   * 	input이 저장되어 있음
  */
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1

  decltype(activation.init(0,0))  accu[PE];
  /* hwkim commented
   * memory
   * ThresholdsActivation class 타입의 PE개 원소를 가지는 배열
   * 	convolution layers
   * 	-> layer 0의 경우 24-bit fixed point
   * 	-> 나머지 layer는 16-bit integer
   * 	fc layers
   * 	-> layer 6/7 - 16-bit integer
   * 	-> layer 8(마지막) - 16-bit integer
   * PE 당 1개의 threshold를 가짐
   * accumulation 값은 output channel(PE) 당 하나
   * 	-> 1 pixel씩(만) 계산하므로
   * 	-> 원래 accumulation은 kernel window 하나 단위
   * 	-> 따라서 SIMD들에서는 같은 accumuulation이 공통으로 사용됨
   * PE 별로 accum 하므로 PE 개 배열 선언
   */
#pragma HLS ARRAY_PARTITION variable=accu complete dim=0

  unsigned  nf   = 0;
  /* hwkim commented
   * 16개의 PE가 몇 번 수행되었는지를 나타내는 index
   * output 전체를 다 계산하기 위해서는 layer 0의 경우, 4번만 수행되면 됨
   * 	즉, nf가 3이면 끝
   */
  unsigned  sf   = 0;
  /* hwkim commented
   * SIMD 단위 몇 번 수행되었는지를 나타내는 index
   * 즉, input channel 방향으로 끝까지 다 계산했는지를 의미
   * layer 0의 경우, sf는 0이면 끝
   */
  unsigned  tile = 0; // invariant: tile = nf*SF + sf

  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelinening the way we want
  unsigned const TOTAL_FOLD = NF * SF;
  /* hwkim commented
   * TOTAL_FOLD는 PE 및 SIMD개의 연산을 동시에 수행했을 때,
   * output 전체 channel 1 pixel을 다 계산하는데 걸리는 횟수
   */
  for(unsigned  i = 0; i < reps * TOTAL_FOLD; i++) {
      /* hwkim commented
       * 여기에서의 reps는
       * 	convolution layer의 경우
       * 		reps* OFMDim * OFMDim
       * 		즉, output pixel 총 개수
       * 	fc layer의 경우 1
       * 		즉, 1 pixel
       */
#pragma HLS PIPELINE II=1
    TI  inElem;
    if(nf == 0) {
    	/* hwkim commented
    	 * PE 단위로 끝까지 연산 다했을 때만 새로운 input을 read
    	 * 	-> 즉, 현재 input pixel 위치에서
    	 * 		모든 output channel을 다 연산했다는 말
    	 */
      // read input from stream
      inElem = in.read();
      /* hwkim commented
       * in -> convInp
       * 	sliding 순으로 정렬된 input
       */
      // store in appropriate buffer for reuse
      inputBuf[sf] = inElem;
      /* hwkim commented
       * 3x3 kernel (1,1)~(3,3)까지 sf증가하면서 순차 실행
       * inputBuf에 sf증가시키면서 3x3 input(SIMD 채널 개) 순차적으로 저장
       */
    }
    else {
      // reuse buffered input
      inElem = inputBuf[sf];
      /* hwkim commented
       * nf 0빼고 나머지는, 같은 input에 대해 kernel만 다름
       * 따라서 같은 input을 reuse
       */
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
	     * 		-> 즉, 단순히 accu[pe]를 0으로 초기화
	     * 			인자로 들어가는 nf, pe는 의미 없는 듯
	     */
      }
    }

    // compute matrix-vector product for each processing element
    auto const &w = weights.weights(tile);
    /* hwkim commented
     * BinaryWeights class의 member
     * 	TileIndex weights(TileIndex class 타입)의
     * 		-> m_par(BinaryWeights class 참조자 타임) member에
     * 			현재 BinaryWeights 참조자 연결
     * 		-> m_idx member를 tile로 초기화
     */
    for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
    	/* hwkim commented
    	 * PE들 병렬 실행
    	 */
      auto const  wgt = TWeightI()(w[pe]);
      /* hwkim commented
       * auto로 인해 wgt는 TWeightI의 반환 값(w[pe] == weights
       * 	== BinaryWeights class)으로 type 결정
       * TWeights == Recast<Binary>
       * 	-> Recast class의 ()(w[pe]) operator는 Recast class의 member인
       * 		Container class의 m_val member를 w[pe]로 초기화
       * w[pe]
       * 	-> w(BinaryWeights class)의 [] operator는
       * 		m_par.m_weights[pe][m_idx] 값을 return
       * 		>> m_par는 위에서 연결한 BinaryWeights class 참조자이므로
       * 			m_par는 BinaryWeights class로 볼 수 있음
       * 		>> m_idx는 위에서 설정한 tile 값으로 index
       * 	-> 따라서 위 구문은 해당 pe 및 위의 tile을 index로
       * 		m_weights[][]를 접근해서 해당 참조자를 return
       * 		(값을 대입하는 것이 아니라 참조자를 연결해서
       * 		해당 데이터에 접근할 수 있게 함)
       */
      auto const  act = TSrcI()(inElem);
      /* hwkim commented
       * TSrcI
       * 	layer 0 -> Slice<ap_fixed<8, 1, AP_TRN, AP_SAT>>
       * 	other layers -> Recast<XnorMul>
       * TSrcI()(inElem)
       * 	-> Slice class의 contrainer의 m_val을 inElem으로 채우고
       * 		Container class의 참조자를 반환하여
       * 		act에 연결
       * 		(즉, act에 참조자 연결해서 해당 데이터에 접근할 수 있게 함)
       */
      accu[pe] = mac<SIMD>(accu[pe], wgt, act, r);
      /* hwkim commented
       * type
       * 	accu[pe] -> ThresholdsActivaiton class
       * 		layer 0 -> 24-bit fixed
       * 	wgt -> BinaryWeight class
       * 		layer 0 -> 1-bit
       * 	act -> ThresholdsActivaiton
       * 		layer 0 -> 8-bit fixed
       */
    }

    // keep track of which folded synapse/neuron we are processing
    ++tile;
    if(++sf == SF) {
    	/* hwkim commented
    	 * input channel 방향 끝까지 계산 다 했으면
    	 */
      // produce output and clear accumulators
      auto  outElem = TDstI().template operator()<TO>();
      /* hwkim commented
       * outElem은 TO-width의 ap_uint -> ap_uint<TO>
       * TO는 PE*TDstI::width
       * 즉, output은 PE-width의 stream
       * mvau 함수 끝나고, 밖에서 WidthAdjustedOutputStream의
       * 	소멸자 호출 시, PE-width -> OutStreamWidth(output channel 개수)로
       * 	변환함
       */
      for (unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
	    outElem[pe] = activation.activate(nf, pe, accu[pe]);
	    /* hwkim commented
	     * activation은 thresholds 값 갖고 있음
	     * ThresholdsActivation class의 activate 함수 호출
	     * accu 값과 저장되어 있는 threshold 값을 단순 비교하여
	     * 		threshold보다 크면 1, 작으면 0 반환?
	     * 		(1-bit 반환; NumTH*1-bit로 width가 계산됨)
	     */
      }

      out.write(outElem);

      // next folded neuron or image
      sf = 0;
      if(++nf == NF) {
    	  /* hwkim commented
    	   * 모든 kernel에 대해 연산 다 했으면
    	   */
	    nf   = 0;
	    tile = 0;
      }
    }
  }
}




template<
  unsigned MatrixW, unsigned MatrixH, unsigned SIMD, unsigned PE,
  // hwkim modified for padding
  unsigned OFMDim,

  typename TSrcI = Identity, typename TDstI = Identity, typename TWeightI = Identity,
  typename TI, typename TO, typename TW, typename TA, typename R
>
/* hwkim commented
 * 	MatrixW -> 3x3x3
 * 		ConvKernelDim * ConvKernelDim * IFMChannels
 *	MatrixH -> 64
 *		OFMChannels -> output neuron channels
 *	SIMD -> 3
 *	PE -> 16
 *	TSrcI -> 8-bit
 *	TDstI -> 1-bit
 *	TWeightI -> 1-bit
 *	TI -> 24-bit
 *		static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp)
 *	TO -> 16-bit
 *		static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut)
 *	TW -> BinaryWeights<L0_SIMD, L0_PE, L0_WMEM>
 *		해당 class type 임
 *	TA -> ThresholdsActivation<L0_TMEM, L0_PE, L0_API, ap_fixed<24, 16>, ap_uint<L0_API> >
 *		해당 class type 임
 *	R -> LUT
*/
void Matrix_Vector_Activate_Batch_Padding(hls::stream<TI> & in,
				  hls::stream<TO> &out,
				  // hwkim modified for debug
#ifdef ACTIVATION_LOG
				  hls::stream<TO> &out_log,
#endif
				  TW  const &weights,
				  TA  const &activation,
				  int const  reps,
				  R const &r) {
  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  unsigned const  NF = MatrixH / PE;
  // how many synapse groups each row is split into
  // alternatively: number of horizontal matrix chunks
  unsigned const  SF = MatrixW / SIMD;

  // hwkim modified for padding (fan-in scaling)
  //unsigned const FAN_IN = MatrixW;

  // hwkim modified for debug
#ifdef ACTIVATION_LOG
  string conv_out_file_name = "conv_" + to_string(weighted_layer_cnt+1) + "_out_file.txt";
  ofstream conv_out_log_file(conv_out_file_name);
  if(!conv_out_log_file.is_open()){
 	 cout << "conv_out_log_file open error" << endl;
  }

  extern string golden_file_dir;
  string golden_conv_out_file_name = golden_file_dir + "binConv" + to_string(weighted_layer_cnt+1) + "_minusBias.txt";
  ifstream golden_conv_out_file(golden_conv_out_file_name);
  if(!golden_conv_out_file.is_open()){
	  cout << "golden_conv_out_file open error" << endl;
  }

#endif

  // input vector buffers
  TI  inputBuf[SF];
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1

  decltype(activation.init(0,0))  accu[PE];
#pragma HLS ARRAY_PARTITION variable=accu complete dim=0

  unsigned  nf   = 0;
  unsigned  sf   = 0;
  unsigned  tile = 0; // invariant: tile = nf*SF + sf
  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelinening the way we want
  unsigned const TOTAL_FOLD = NF * SF;

  // hwkim modified for padding
  int xy, x, y, kx, ky;

  for(unsigned  i = 0; i < reps * TOTAL_FOLD; i++) {
#pragma HLS PIPELINE II=1
    TI  inElem;
    if(nf == 0) {
      // read input from stream
      inElem = in.read();
      // store in appropriate buffer for reuse
      inputBuf[sf] = inElem;
    }
    else {
      // reuse buffered input
      inElem = inputBuf[sf];
    }

    // Threshold Initialisation
    if(sf == 0) {
      for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
	    accu[pe] = activation.init(nf, pe);
      }
    }

    // compute matrix-vector product for each processing element
    auto const &w = weights.weights(tile);

    // hwkim modified for padding (fan-in scaling)
//    ap_uint<1> padding[PE];
//    unsigned int fan_in_loss[PE];
//    for(unsigned  pe = 0; pe < PE; pe++) {
//#pragma HLS UNROLL
//    	padding[pe] = 0;
//    	fan_in_loss[pe] = 0;
//    }
	// hwkim modified for padding
	xy = i/TOTAL_FOLD;
	y = xy/OFMDim;
	x = xy%OFMDim;
	//tile: sf -> kx -> ky -> nf
	ky = ((tile/(SF/9))%9)/3;
	kx = ((tile/(SF/9))%9)%3;
	// hwkim modified for debug
//	cout << "nf=" << nf << ", sf=" << sf << ", pe=" << pe << ",
//		y=" << y << ", x=" << x << ", ky=" << ky << ", kx=" << kx << ", tile=" << tile << endl;
//	if(y==7 && x==24){
//		cout << "here" << endl;
//	}


    for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
      auto const  wgt = TWeightI()(w[pe]);
      auto const  act = TSrcI()(inElem);

      // hwkim modified for padding
      //accu[pe] = mac<SIMD>(accu[pe], wgt, act, r);
      if((x==0 && kx==0)
    	  ||(y==0 && ky==0)
		  ||(x==(OFMDim-1) && kx==2)
		  ||(y==(OFMDim-1) && ky==2)){
    	  // for fan-in scaling
    	  //padding[pe] = 1;
    	  //fan_in_loss[pe] += SIMD;
    	  ;
       }
      else{
    	  // hwkim modified for debug
//    	  if(pe==0){
//    		  cout << "start =====================" << endl;
//    	  }

    	  accu[pe] = mac<SIMD>(accu[pe], wgt, act, r);

    	  // hwkim modified for debug
//    	  if(pe==0){
//    		  cout << "end =====================" << endl;
//    	  }
       }
    }

    // keep track of which folded synapse/neuron we are processing
    ++tile;
    if(++sf == SF) {
      // produce output and clear accumulators
      auto  outElem = TDstI().template operator()<TO>();
      for (unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
 	     // hwkim modified for debug
 #ifdef ACTIVATION_LOG
   		conv_out_log_file << fixed;
 		conv_out_log_file.setf(ios::showpoint);
 		conv_out_log_file.precision(8);
 		conv_out_log_file << accu[pe] << " | ";
 		//decltype(activation.init(0,0))	golden_buf;
// 		double golden_buf;
// 		golden_conv_out_file >> golden_buf;
// 		if(accu[pe]!=golden_buf){
// 			cout << fixed;
// 			cout.precision(7);
// 			cout.setf(ios_base::showpoint);
// 			cout << "differ @ (" << y << "," << x << ") gold: " << golden_buf << ", accu[" << nf << ", " << pe << "]: " << accu[pe] << endl;
// 		}

 #endif
    	  // hwkim modified for bias
    	//accu[pe] = accu[pe] + activation.m_thresholds[pe][nf][0];

    	outElem[pe] = activation.activate(nf, pe, accu[pe]);
    	// hwkim modified for padding (fan-in scaling)
//    	if((TI::width==1) && (padding[pe])){
//			accu[pe] = accu[pe]*(decltype(activation.init(0,0)))((float)FAN_IN/(FAN_IN - fan_in_loss[pe]));
//			outElem[pe] = activation.activate(nf, pe, accu[pe]);
//		}
//    	else{
//    		outElem[pe] = activation.activate(nf, pe, accu[pe]);
//    	}
	     // hwkim modified for padding (fan-in scaling)
	     //padding[pe] = 0;
	     //fan_in_loss[pe] = 0;
      }
      out.write(outElem);
      // hwkim modified for debug
#ifdef ACTIVATION_LOG
      out_log.write(outElem);
#endif

      // next folded neuron or image
      sf = 0;
      if(++nf == NF) {
    	  /* hwkim commented
    	   * 모든 kernel에 대해 연산 다 했으면
    	   */
	    nf   = 0;
	    tile = 0;
#ifdef ACTIVATION_LOG
	    conv_out_log_file << endl;
#endif
      }
    }
  }

#ifdef ACTIVATION_LOG
  conv_out_log_file.close();
#endif
}
#endif
