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
#include <stdio.h>
//extern int weighted_layer_cnt;
//#define DEBUG
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
      // hwkim modified for conv layer's activation comparison using +- accumulation
      accu[pe] = mac<SIMD>(accu[pe], wgt, act, r
#ifdef ACTIVATION_LOG
    		  , 0
#endif
    		  );
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


template<unsigned MatrixW, unsigned MatrixH, unsigned SIMD, unsigned PE, unsigned OFMDim,
  unsigned OFMHeight,	// hwkim added for segmentation
  unsigned Top, unsigned Bottom, unsigned Left, unsigned Right,	// hwkim modified for padding
#ifdef ACTIVATION_LOG
  unsigned int LayerCnt, unsigned int OutWidth,	// hwkim added for log
#endif
  typename TDstElem,	// hwkim added for batch norm scale
  typename TSrcI = Identity, typename TDstI = Identity, typename TWeightI = Identity,
  typename TI, typename TO, typename TW, typename TA,
  typename R>
void Matrix_Vector_Activate_Batch_Padding(
		hls::stream<TI> & in,
		hls::stream<TO> &out,
		TW  const &weights,
		TA  const &activation,
		int const  reps,
		R const &r)
{
  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  unsigned const  NF = MatrixH / PE;
  // how many synapse groups each row is split into
  // alternatively: number of horizontal matrix chunks
  unsigned const  SF = MatrixW / SIMD;
  /*
   * conv kernel dim * conv kernel dim * IFMChannels
   * -> SF = 3 * 3 * (IFMChannels / SIMD)
   */

  // hwkim modified for debug
#ifdef ACTIVATION_LOG
  extern string snapshot_dir;
  extern string golden_file_dir;
  int compare_skip = 0;
  ap_uint<OutWidth> act_buf_arr[NF];
  ap_uint<NF*OutWidth> act_buf=0;
  string conv_out_file_name = snapshot_dir + "conv_" + to_string(LayerCnt+1) + "_out_minusBias.txt";
  string act_file_name = snapshot_dir + "activation_" + to_string(LayerCnt+1) + "_log.txt";
  string golden_conv_out_file_name = golden_file_dir + "binConv" + to_string(LayerCnt+1) + "_minusBias.txt";
  string golden_act_file_name = golden_file_dir + "Sign" + to_string(LayerCnt+1) + ".txt";
  string conv_out_comp_file_name = "conv_" + to_string(LayerCnt+1) + "_out_minusBias_comp.txt";
  string act_comp_file_name = "act_" + to_string(LayerCnt+1) + "_comp.txt";

  ofstream conv_out_log_file(conv_out_file_name);
  ofstream activation_log_file(act_file_name);
  ifstream golden_conv_out_file(golden_conv_out_file_name);
  FILE * golden_file = fopen(golden_act_file_name.c_str(),"rt");
  ofstream conv_out_comp_file(conv_out_comp_file_name);
  ofstream act_comp_file(act_comp_file_name);
  ofstream last_layer_scaled_file("last_layer_scaled_log.log");

  if(!conv_out_log_file.is_open())	cout << "conv_out_log_file open error" << endl;
  if(!activation_log_file.is_open())	cout << act_file_name << " open error!!" << endl;
  if(!golden_conv_out_file.is_open())	cout << "golden_conv_out_file open error" << endl;
  if(golden_file==NULL){
	  cout << golden_act_file_name << " open error!!" << endl;
	  compare_skip = 1;
  }
  if(!conv_out_comp_file.is_open())	cout << "conv_out_comp_file open error" << endl;
  if(!act_comp_file.is_open())	cout << act_comp_file_name << " open error!" << endl;
  if(!last_layer_scaled_file.is_open())	cout << "last_layer_scaled_file open error" << endl;
#endif

  // input vector buffers
  TI	inputBuf[SF];
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1

  decltype(activation.init(0,0))  accu[PE];
#pragma HLS ARRAY_PARTITION variable=accu complete dim=0

// hwkim added for activation comparision using +- accumulation
#ifdef ACTIVATION_LOG
  decltype(activation.init(0,0))  accu_pm[PE];
#endif

  unsigned  nf   = 0;
  unsigned  sf   = 0;
  unsigned  tile = 0; // invariant: tile = nf*SF + sf
  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelinening the way we want
  unsigned const TOTAL_FOLD = NF * SF;

  // hwkim modified for padding (fan-in scaling)
  //unsigned int xy;
  unsigned int x=0, y=0, kx=0, ky=0;
  unsigned const fan_in_step = MatrixW/9;
  unsigned const sf_ch = SF/9;
  unsigned int fan_in=0;
  unsigned int sf_ch_cnt=0;

  // hwkim modified for dependency
  //for(unsigned  i = 0; i < reps * TOTAL_FOLD; i++) {
  for(unsigned  i = 0; i < reps * TOTAL_FOLD / SF; i++) {
	  for(unsigned sf=0; sf < SF; sf++){
#pragma HLS PIPELINE II=1 rewind
		  // hwkim commented for positive only accumulation - counter implementation? rather than / %
//		  xy = i*SF/TOTAL_FOLD;		//xy = i/TOTAL_FOLD;	// hwkim modified for dependency
//		  y = xy/OFMDim;
//		  x = xy%OFMDim;
//		  ky = ((tile/(SF/9))%9)/3;
//		  kx = ((tile/(SF/9))%9)%3;
		  //tile: sf -> kx -> ky -> nf

		  TI  inElem;
		  if(nf == 0) {
			  // hwkim modified for padding
//			  inElem = in.read();	// read input from stream
//			  inputBuf[sf] = inElem;	// store in appropriate buffer for reuse
			  if((x<Left && kx<Left)
				  ||(y<Top && ky<Top)
				  ||(x>(OFMDim-1-Right) && kx>(3-1-Right))
				  ||(y>(OFMHeight-1-Bottom) && ky>(3-1-Bottom))){
				  ;	// skip
			  }
			  else{
				  inElem = in.read();
				  inputBuf[sf] = inElem;
			  }
		  }
		  else {
			  inElem = inputBuf[sf];		// reuse buffered input
		  }

		  // Threshold Initialisation
		  if(sf == 0) {
			  for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
				  accu[pe] = activation.init(nf, pe);
#ifdef ACTIVATION_LOG
				  accu_pm[pe] = activation.init(nf, pe);
#endif
			  }
		  }

		  // compute matrix-vector product for each processing element
		  auto const &w = weights.weights(tile);

		  for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
			  auto const  wgt = TWeightI()(w[pe]);
			  auto const  act = TSrcI()(inElem);

			  // hwkim modified for padding
			  //accu[pe] = mac<SIMD>(accu[pe], wgt, act, r);
			  if((x<Left && kx<Left)
					  ||(y<Top && ky<Top)
					  ||(x>(OFMDim-1-Right) && kx>(3-1-Right))
					  ||(y>(OFMHeight-1-Bottom) && ky>(3-1-Bottom))){
				  ;	//skip pad from accumulation
			  }
			  else{
				  // hwkim modified for activation comparison using +- accumulation
				  accu[pe] = mac<SIMD>(accu[pe], wgt, act, r
#ifdef ACTIVATION_LOG
						  , 0
#endif
				  );
#ifdef ACTIVATION_LOG
				  accu_pm[pe] = mac<SIMD>(accu_pm[pe], wgt, act, r, 1);
#endif
			  }
		  }

		  // keep track of which folded synapse/neuron we are processing
		  ++tile;

		  // hwkim added for debug
//		  cout << de << "tile: " << tile << ", sf: " << sf << ", nf: " << nf;
//		  cout << ", ky, kx: " << ky << "," << kx << ", fan_in: " << fan_in;
//		  cout << ", y, x: " << y << "," << x << endl;

		  // hwkim added for positive only accumulation
		  if(++sf_ch_cnt==sf_ch){
			  sf_ch_cnt=0;
			  if((x<Left && kx<Left)
				  ||(y<Top && ky<Top)
				  ||(x>(OFMDim-1-Right) && kx>(3-1-Right))
				  ||(y>(OFMHeight-1-Bottom) && ky>(3-1-Bottom))){
				  ;
			  }
			  else{
				  fan_in += fan_in_step;
			  }
			  if(++kx==3){
				  kx=0;
				  if(++ky==3){
					  ky=0;
					  if(nf==(NF-1)){
						  if(++x==OFMDim){
							  x=0;
							  if(++y==OFMHeight){
								  y=0;
							  }
						  }
					  }
				  }
			  }
		  }

		  // hwkim modified for dependency - 191004 ------------------------------------------------
		  if(sf == (SF-1)) {
		  	// produce output and clear accumulators
		  	auto  outElem = TDstI().template operator()<TO>();
		  	for (unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL

// hwkim modified for debug
// hwkim commented for 0-1 accumulation - not +- accum
#ifdef ACTIVATION_LOG
		  		// write conv_out_minus_bias
		  		conv_out_log_file << setprecision(8) << accu_pm[pe] << endl;//" | ";
		  		 //compare golden with computed
		  		decltype(activation.init(0,0))	golden_buf;
		  		golden_conv_out_file >> golden_buf;
		  		if(accu_pm[pe]!=golden_buf){
		  			conv_out_comp_file << fixed;
		  			conv_out_comp_file.precision(8);
		  			conv_out_comp_file.setf(ios_base::showpoint);
		  			conv_out_comp_file << "differ @ (" << y << "," << x << ") gold: " << golden_buf << ", accu[" << nf << ", " << pe << "]: " << accu[pe] << endl;
		  		}
#endif
		  		// hwkim modified for positive only accumulation
		      	//accu[pe] = accu[pe] + activation.m_thresholds[pe][nf][0];
		      	//outElem[pe] = activation.activate(nf, pe, accu[pe]);
		  		// hwkim modified for batch norm scale
		  		//ap_fixed<24,16,AP_TRN,AP_SAT> fxdoutElem;
		  		TDstElem outElem_unit;
		  		//ap_uint<TDstI::width> uoutElem;
		  		outElem_unit = activation.activate(nf, pe, accu[pe], fan_in);
		  		outElem[pe] = *reinterpret_cast<ap_uint<TDstI::width> *>(&outElem_unit);
		  		//cout << " " << outElem[pe] << endl;	// for debug
		  	}
		  	out.write(outElem);
		  	// hwkim modified for positive only accumulation
	  		fan_in=0;

// hwkim added for debug
#ifdef ACTIVATION_LOG
		  	act_buf_arr[nf] = outElem;
		  //	cout << hex << (unsigned int)act_buf_arr[nf] << endl;
		  	if(nf==(NF-1)){
		  		// write activation log - OutWidth(PE*TDst::width) is under 32 for conv(tconv) layers
		  		act_buf = 0;
		  		for(int nf_cnt=NF-1; nf_cnt>=0; nf_cnt--){
		  			for(int word_cnt=0; word_cnt<OutWidth/4; word_cnt++){
		  				activation_log_file << uppercase << hex << ((unsigned int)(act_buf_arr[nf_cnt]>>(OutWidth-4*(word_cnt+1)))&0xF);
		  			}
		  			// filling act_buf
		  			act_buf = act_buf << OutWidth;	//OutWidth or PE
		  			act_buf = act_buf | act_buf_arr[nf_cnt];
		  		}
		  		activation_log_file  << endl;

		  		// compare activation with gold result
		  		ap_uint<NF*OutWidth> gold_buf = 0;
		  		char gold_buf_ch[(NF*OutWidth)/4+1];
		  		char gold_buf_ch64[17];
		  		gold_buf_ch64[16] = 0;
		  		unsigned long gold_buf_long;
		  		if(compare_skip==0){
		  			fscanf(golden_file, "%s", gold_buf_ch);
		  			// there's no layers with channel count smaller than 64
		  			for(int word_cnt=0; word_cnt<(NF*OutWidth)/64; word_cnt++){
		  				for(int i=0; i<64/4; i++){
		  					gold_buf_ch64[i] = gold_buf_ch[word_cnt*16+i];
		  				}
		  				gold_buf_long = strtoul(gold_buf_ch64, NULL, 16);
		  				gold_buf = gold_buf << 64;
		  				gold_buf = gold_buf | (*reinterpret_cast<ap_uint<64> *>(&gold_buf_long));
		  			}

		  			if(act_buf!=gold_buf){
		  				act_comp_file << dec << "@(" << setw(2) << y << "," << setw(2) << x << ")" <<
		  						hex << " golden: ";
		  				if((NF*OutWidth)>=64){
		  					for(int i=0; i<(NF*OutWidth)/64; i++){
		  						act_comp_file << (unsigned long long )(gold_buf >> 64*(OutWidth/64-i-1));
		  					}
		  					act_comp_file << "," << endl << setw(17) << hex << "act: ";
		  					for(int i=0; i<OutWidth/64; i++){
		  						act_comp_file << (unsigned long long )(act_buf >> 64*(OutWidth/64-i-1));
		  					}
		  					act_comp_file << endl;
		  				}
		  				else{
		  					act_comp_file << (unsigned long long )gold_buf << "," << endl
		  							<< setw(17) << hex << "act: "  << (unsigned long long )act_buf << endl;
		  				}
		  			}
		  		}
		  		if(x==0)
					cout << dec << y << "/" << OFMHeight << endl;
		  	}
#endif

		  	// next folded neuron or image
		  	// hwkim modified for dependency
		  	//sf = 0;

		  	if(++nf == NF) {
		  		/* hwkim commented
		  		 * 모든 kernel에 대해 연산 다 했으면
		  		 */
		  		nf   = 0;
		  		tile = 0;
		  	}
		}



		// hwkim modified for dependency - 191004 -----------------------------------------------------
	}
  }

#ifdef ACTIVATION_LOG
  conv_out_log_file.close();
  conv_out_comp_file.close();
  last_layer_scaled_file.close();
#endif
}


template<
  unsigned IFMChannels, unsigned MatrixH, unsigned SIMD, unsigned PE, unsigned OFMDim,
  // hwkim modified for segmentation
  unsigned OFMHeight,
  // hwkim modified for padding
  unsigned Top,
  unsigned Bottom,
  unsigned Left,
  unsigned Right,
#ifdef ACTIVATION_LOG
  unsigned int LayerCnt,
  unsigned int OutWidth,
#endif

  typename TSrcI = Identity, typename TDstI = Identity, typename TWeightI = Identity,
  typename TI, typename TO, typename TW, typename TA, typename R
>
void Matrix_Vector_Activate_Batch_Skipping(hls::stream<TI> & in,
				  hls::stream<TO> &out,
				  TW  const &weights,
				  TA  const &activation,
				  int const  reps,
				  R const &r) {
  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  unsigned const  NF = MatrixH / PE;
  // how many synapse groups each row is split into
  // alternatively: number of horizontal matrix chunks

  //unsigned const  SF = MatrixW / SIMD;

  // hwkim modified for debug
#ifdef ACTIVATION_LOG
  extern string snapshot_dir;
  extern string golden_file_dir;
  int compare_skip = 0;
  ap_uint<OutWidth> act_buf_arr[NF];
  ap_uint<NF*OutWidth> act_buf=0;

  string conv_out_file_name = snapshot_dir + "conv_" + to_string(LayerCnt+1) + "_out_minusBias.txt";
  ofstream conv_out_log_file(conv_out_file_name);
  if(!conv_out_log_file.is_open()){
 	 cout << "conv_out_log_file open error" << endl;
  }

  string act_file_name = snapshot_dir + "activation_" + to_string(LayerCnt+1) + "_log.txt";
  ofstream activation_log_file(act_file_name);
  if(!activation_log_file.is_open()){
	  cout << act_file_name << " open error!!" << endl;
  }

  string golden_conv_out_file_name = golden_file_dir + "binConv" + to_string(LayerCnt+1) + "_minusBias.txt";
  ifstream golden_conv_out_file(golden_conv_out_file_name);
  if(!golden_conv_out_file.is_open()){
	  cout << "golden_conv_out_file open error" << endl;
  }

  string golden_act_file_name = golden_file_dir + "Sign" + to_string(LayerCnt+1) + ".txt";
  FILE * golden_file = fopen(golden_act_file_name.c_str(),"rt");
  if(golden_file==NULL){
	  cout << golden_act_file_name << " open error!!" << endl;
	  compare_skip = 1;
  }

  string conv_out_comp_file_name = "conv_" + to_string(LayerCnt+1) + "_out_minusBias_comp.txt";
  ofstream conv_out_comp_file(conv_out_comp_file_name);
  if(!conv_out_comp_file.is_open()){
	  cout << "conv_out_comp_file open error" << endl;
  }

  string act_comp_file_name = "act_" + to_string(LayerCnt+1) + "_comp.txt";
  ofstream act_comp_file(act_comp_file_name);
  if(!act_comp_file.is_open()){
	  cout << act_comp_file_name << " open error!" << endl;
  }

#endif

  // input vector buffers
  //TI  inputBuf[SF];
  TI  inputBuf[4*IFMChannels/SIMD];	// hwkim: ky < 2, kx < 2
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1

  decltype(activation.init(0,0))  accu[PE];
#pragma HLS ARRAY_PARTITION variable=accu complete dim=0

  // hwkim modified for activation comparison using +- accumulation
#ifdef ACTIVATION_LOG
  decltype(activation.init(0,0))  accu_pm[PE];
#endif

  unsigned  nf   = 0;
  unsigned  sf   = 0;
  unsigned  tile = 0; // invariant: tile = nf*SF + sf
  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelinening the way we want
  unsigned const TOTAL_FOLD = (reps/4 + reps + reps) * NF * (IFMChannels/SIMD);	// for padding of 0, 1, 0, 1 only
  	  	  	  	  	  	  	  //(reps/4 + reps/4*2*2 + reps/4*4) * NF * (IFMChannels/SIMD);

  // hwkim modified for padding
  unsigned int x=0, y=0, kx=0, ky=0;
  unsigned int simd_cnt=0;
  unsigned int w_addr=0;
  unsigned int sf_max=0;

  // hwkim modified for padding (fan-in scaling)
	//unsigned const FAN_IN = IFMChannels*3*3;
  unsigned int fan_in = 0;
  unsigned const fan_in_step = IFMChannels;

  //for(unsigned  i = 0; i < reps * TOTAL_FOLD; i++) {
  for(unsigned  i = 0; i < TOTAL_FOLD; i++) {
#pragma HLS PIPELINE II=1

	//tile: sf -> kx -> ky -> nf
    TI  inElem;
    if(nf == 0) {
    	// hwkim modified for weight flip
    	if(((y==0)&&(ky==0))
			|| ((x==0)&&(kx==0))){
    		;	// skip
    	}
    	else{
			inElem = in.read();
			// store in appropriate buffer for reuse
			inputBuf[sf] = inElem;
    	}
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
// hwkim added for activation comparison using +- accumulation
#ifdef ACTIVATION_LOG
	    accu_pm[pe] = activation.init(nf, pe);
#endif
      }
    }

    // compute matrix-vector product for each processing element
//    auto const &w = weights.weights(tile);
    w_addr = nf*3*3*(IFMChannels/SIMD)
    		// hwkim modified for weight flip
    		//+ (((y&0x1) + ((!(y&0x1)-ky)<<1))*3 + (x&0x1) + ((!(x&0x1)-kx)<<1))*(IFMChannels/SIMD)
			+ (((y&0x1) + (ky<<1))*3 + (x&0x1) + (kx<<1))*(IFMChannels/SIMD)

			+ simd_cnt;
    auto const &w = weights.weights(w_addr);

    for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
      auto const  wgt = TWeightI()(w[pe]);
      auto const  act = TSrcI()(inElem);

      if(((y==0)&&((1-ky)==1))
    		  || ((x==0)&&((1-kx)==1))){
    	  ;	// skip
      }
      else{
    	  // hwkim modified for activation comparison using +- accumulation
    	  accu[pe] = mac<SIMD>(accu[pe], wgt, act, r
#ifdef ACTIVATION_LOG
    			  , 0
#endif
    			  );
#ifdef ACTIVATION_LOG
    	  accu_pm[pe] = mac<SIMD>(accu_pm[pe], wgt, act, r, 1);
#endif

    	  // hwkim modified for positive only accumulation
    	  if((pe==0) && (simd_cnt==0))
    		  fan_in += fan_in_step;
      }
    }

    // hwkim added for debug
//    if(((y==0)&&((1-ky)==1))
//		|| ((x==0)&&((1-kx)==1))){
//		cout << "(skipped) ";	// skip
//	}
//	cout << "y: " << y;
//	cout << ", x: " << x;
//	cout << ", ky: " << (!(y&0x1)-ky);
//	cout << ", kx: " << (!(x&0x1)-kx);
//	cout << ", simd_cnt: " << simd_cnt;
//	cout << ", w_addr: " << w_addr;
//	cout << ", sf: " << sf;
//	cout << ", nf: " << nf;
//	cout << ", fanin: " << fan_in << endl;

    // keep track of which folded synapse/neuron we are processing
    //++tile;
    //if(++sf == SF) {
    if(++sf==((1<<(!(y&0x1)+!(x&0x1)))*(IFMChannels/SIMD))){
    	// hwkim added for debug
//    	cout << "sf_max = 1<<(!(y&0x1)+!(x&0x1)): " << dec << (1<<(!(y&0x1)+!(x&0x1))) << endl;

      // produce output and clear accumulators
      auto  outElem = TDstI().template operator()<TO>();
      for (unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
// hwkim commented for 0-1 accumulation - not +- accum
#ifdef ACTIVATION_LOG
			conv_out_log_file << setprecision(8) << accu_pm[pe] << endl;//" | ";
			decltype(activation.init(0,0))	golden_buf;
			golden_conv_out_file >> golden_buf;
			if(accu_pm[pe]!=golden_buf){
				conv_out_comp_file << fixed;
				conv_out_comp_file.precision(8);
				conv_out_comp_file.setf(ios_base::showpoint);
				conv_out_comp_file << "differ @ (" << y << "," << x << ") gold: " << golden_buf << ", accu[" << nf << ", " << pe << "]: " << accu[pe] << endl;
			}
#endif
			// hwkim modified for positive only accum
			//outElem[pe] = activation.activate(nf, pe, accu[pe]);
			outElem[pe] = activation.activate(nf, pe, accu[pe], fan_in);
      }
     out.write(outElem);
      // hwkim modified for debug
#ifdef ACTIVATION_LOG
     // hwkim added for debug
     act_buf_arr[nf] = outElem;
     if(nf==(NF-1)){
		  // write activation log - PE is under 32 now
		  act_buf = 0;
		  for(int nf_cnt=NF-1; nf_cnt>=0; nf_cnt--){
			  //activation_log_file << uppercase << setfill('0') << setw(PE/4) << hex << (unsigned long long)act_buf_arr[nf_cnt];
			  for(int word_cnt=0; word_cnt<OutWidth/4; word_cnt++){
				  activation_log_file << uppercase << hex << ((unsigned int)(act_buf_arr[nf_cnt]>>(OutWidth-4*(word_cnt+1)))&0xF);
			  }
			  act_buf = act_buf << OutWidth;	//OutWidth or PE
			  act_buf = act_buf | act_buf_arr[nf_cnt];
		  }
		  activation_log_file  << endl;

		  // compare activation with gold result
		  ap_uint<NF*OutWidth> gold_buf = 0;
		  char gold_buf_ch[(NF*OutWidth)/4+1];
		  char gold_buf_ch64[17];
		  gold_buf_ch64[16] = 0;
		  unsigned long gold_buf_long;
		  if(compare_skip==0){
			  fscanf(golden_file, "%s", gold_buf_ch);
			  // there's no layers with channel count smaller than 64
			  for(int word_cnt=0; word_cnt<(NF*OutWidth)/64; word_cnt++){
				  for(int i=0; i<64/4; i++){
					  gold_buf_ch64[i] = gold_buf_ch[word_cnt*16+i];
				  }
				  gold_buf_long = strtoul(gold_buf_ch64, NULL, 16);
				  gold_buf = gold_buf << 64;
				  gold_buf = gold_buf | (*reinterpret_cast<ap_uint<64> *>(&gold_buf_long));
			  }

			  if(act_buf!=gold_buf){
				  act_comp_file << dec << "@(" << setw(2) << y << "," << setw(2) << x << ")" <<
						  hex << " golden: ";
				  if((NF*OutWidth)>=64){
					  for(int i=0; i<(NF*OutWidth)/64; i++){
						  act_comp_file << (unsigned long long )(gold_buf >> 64*(OutWidth/64-i-1));
					  }
					  act_comp_file << "," << endl << setw(17) << hex << "act: ";
					  for(int i=0; i<OutWidth/64; i++){
						  act_comp_file << (unsigned long long )(act_buf >> 64*(OutWidth/64-i-1));
					  }
					  act_comp_file << endl;
				  }
				  else{
					  act_comp_file << (unsigned long long )gold_buf << "," << endl
							  << setw(17) << hex << "act: "  << (unsigned long long )act_buf << endl;
				  }
			  }
		  }
      }
#endif

      // next folded neuron or image
      sf = 0;
//      if(++nf == NF) {
//	    nf   = 0;
//	    tile = 0;
//      }
    }


    if(++simd_cnt==IFMChannels/SIMD){
    	simd_cnt=0;
    	if(++kx==(!(x&0x1)+1)){
    		kx=0;
    		if(++ky==(!(y&0x1)+1)){
    			ky=0;
    			fan_in = 0;
    			if(++nf==NF){
    				nf=0;
//    				cout << "==================================" << endl;
    				if(++x==OFMDim){
    					x=0;
        				cout << dec << y << "/" << OFMHeight << endl;
    					if(++y==OFMHeight){
    						y=0;
    					}
    				}
    			}
    		}
    	}
    }


  }

#ifdef ACTIVATION_LOG
  conv_out_log_file.close();
  conv_out_comp_file.close();
#endif
}


template<unsigned MatrixW, unsigned MatrixH, unsigned SIMD, unsigned PE, unsigned OFMDim,
  unsigned OFMHeight,	// hwkim added for segmentation
  unsigned Top, unsigned Bottom, unsigned Left, unsigned Right,	// hwkim modified for padding
  unsigned WAY,
#ifdef ACTIVATION_LOG
  unsigned int LayerCnt, unsigned int OutWidth,	// hwkim added for log
#endif
  typename TDstElem,	// hwkim added for batch norm scale
  typename TSrcI = Identity, typename TDstI = Identity, typename TWeightI = Identity,
//  typename TI,
  typename TO,
  // hwkim added for ternary
//  typename TIM, // type mask
  typename TOM,
  //typename TW, typename TWM,
  typename PI, typename PW, typename PM,
  typename FI,

  typename TA,
  typename R>
void Matrix_Vector_Activate_Batch_SkipSeparately(
//		hls::stream<TI> & in,
//		hls::stream<TIM> & in_mask,
		hls::stream<TO> &out,
		hls::stream<TOM> &out_mask,

		// hwkim modified for ternary
		//TW  const &weights,
		hls::stream<PI> * packed_input,
		hls::stream<PW> * packed_weight,
		hls::stream<FI> * sf_num,
		hls::stream<PM> * packed_mask,

		TA  const &activation,
		int const  reps,
		R const &r)
{
  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  unsigned const  NF = MatrixH / PE;
  // how many synapse groups each row is split into
  // alternatively: number of horizontal matrix chunks
  unsigned const  SF = MatrixW / SIMD;
  /*
   * conv kernel dim * conv kernel dim * IFMChannels
   * -> SF = 3 * 3 * (IFMChannels / SIMD)
   */

  // hwkim added for debug
//  for(unsigned char pe=0; pe<PE; pe++){
//	  cout << packed_input[pe].size();
//  }


  // hwkim added for debug
#ifdef ACTIVATION_LOG
  extern string snapshot_dir;
  extern string golden_file_dir;
  int compare_skip = 0;
  ap_uint<OutWidth> act_buf_arr[NF];
  ap_uint<PE> act_mask_buf_arr[NF];
  ap_uint<NF*OutWidth> act_buf=0;
  ap_uint<NF*PE> act_mask_buf=0;

  string conv_out_file_name 		= snapshot_dir + "conv_" + to_string(LayerCnt+1) + "_out_minusBias.txt";
  string act_file_name 				= snapshot_dir + "activation_" + to_string(LayerCnt+1) + "_log.txt";
  string act_mask_file_name 		= snapshot_dir + "activation_mask_" + to_string(LayerCnt+1) + "_log.txt";
  string golden_conv_out_file_name = golden_file_dir + "binConv" + to_string(LayerCnt+1) + "_minusBias.txt";
  string golden_act_file_name 		= golden_file_dir + "Sign" + to_string(LayerCnt+1) + ".txt";
  string golden_mask_file_name 		= golden_file_dir + "Sign" + to_string(LayerCnt+1) + "_flag.txt";
  string conv_out_comp_file_name 	= "conv_" + to_string(LayerCnt+1) + "_out_minusBias_comp.txt";
  string act_comp_file_name 		= "act_" + to_string(LayerCnt+1) + "_comp.txt";
  string mask_comp_file_name 		= "mask_" + to_string(LayerCnt+1) + "_comp.txt";

  ofstream conv_out_log_file(conv_out_file_name);
  ofstream activation_log_file(act_file_name);
  ofstream activation_mask_log_file(act_mask_file_name);
  ifstream golden_conv_out_file(golden_conv_out_file_name);
  FILE * golden_file = fopen(golden_act_file_name.c_str(),"rt");
  FILE * golden_mask_file = fopen(golden_mask_file_name.c_str(),"rt");
  ofstream conv_out_comp_file(conv_out_comp_file_name);
  ofstream act_comp_file(act_comp_file_name);
  ofstream mask_comp_file(mask_comp_file_name);
  ofstream last_layer_scaled_file("last_layer_scaled_log.log");

  if(!conv_out_log_file.is_open())			cout << "conv_out_log_file open error" << endl;
  if(!activation_log_file.is_open())		cout << act_file_name << " open error!!" << endl;
  if(!activation_mask_log_file.is_open())	cout << act_mask_file_name << " open error!!" << endl;
  if(!golden_conv_out_file.is_open())		cout << "golden_conv_out_file open error" << endl;
  if(golden_file==NULL){
	  cout << golden_act_file_name << " open error!!" << endl;
	  compare_skip = 1;
  }
  if(golden_mask_file==NULL)				cout << golden_mask_file_name << " open error!!" << endl;
  if(!conv_out_comp_file.is_open())		cout << "conv_out_comp_file open error" << endl;
  if(!act_comp_file.is_open())				cout << act_comp_file_name << " open error!" << endl;
  if(!mask_comp_file.is_open())				cout << mask_comp_file_name << " open error!" << endl;
  if(!last_layer_scaled_file.is_open())	cout << "last_layer_scaled_file open error" << endl;
#endif

  // input vector buffers
//  TI	inputBuf[SF];
//#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1
  // hwkim added for ternary
//  TIM	imaskBuf[SF];
//#pragma HLS ARRAY_PARTITION variable=imaskBuf complete dim=1

  decltype(activation.init(0,0))  accu[PE];
#pragma HLS ARRAY_PARTITION variable=accu complete dim=0

// hwkim added for activation comparision using +- accumulation
#ifdef ACTIVATION_LOG
  decltype(activation.init(0,0))  accu_pm[PE];
#endif

  unsigned char nf   = 0;
  unsigned char sf   = 0;
  unsigned char tile = 0; // invariant: tile = nf*SF + sf

  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelinening the way we want
  unsigned const TOTAL_FOLD = NF * SF;

  unsigned short x=0, y=0;
  ap_uint<2> kx=0, ky=0;

  unsigned const fan_in_step = MatrixW/9;
  unsigned const sf_ch = SF/9;
  unsigned short fan_in=0;
//  unsigned char sf_pe[PE];
//#pragma HLS ARRAY_PARTITION variable=sf_pe complete dim=1

//  for(unsigned pe=0; pe<PE; pe++){
//#pragma HLS UNROLL
//	  sf_pe[pe] = 0;
//  }

  x = 0;
  y = 0;
  kx = 0;
  ky = 0;
  nf = 0;

  for(unsigned  i = 0; i < reps * TOTAL_FOLD / SF; i++) {
#pragma HLS LOOP_FLATTEN off

//	  TI  inElem;
	  // hwkim added for ternary
//	  TI  inElem_pe[PE];
	  PI  inElem_pe[PE];
#pragma HLS ARRAY_PARTITION variable=inElem_pe complete dim=1
	  // hwkim modified for way
//	  unsigned short sf_num_buf[PE];
//#pragma HLS ARRAY_PARTITION variable=sf_num_buf complete dim=1
	  unsigned short sf_num_buf[PE][SIMD/WAY];
#pragma HLS ARRAY_PARTITION variable=sf_num_buf complete dim=0

	  unsigned char sf_max = 0;

	  sf = 0;
	  // hwkim added for sf separation
//	  for(unsigned char pe=0; pe<PE; pe++){
//#pragma HLS UNROLL
//		  sf_pe[pe] = 0;
//	  }

	  // hwkim added for ternary
	  auto  outElem = TDstI().template operator()<TO>();
	  TOM outMaskElem;
	  ap_uint<WAY> mask;

	  // hwkim modified for sf separation
//	  ap_uint<PE> sf_sync_flag = 0;

	  // ** hwkim modified for OPTIMIZATION
//	  for(unsigned char pe=0; pe<PE; pe++){
//#pragma HLS UNROLL
//		  for(unsigned char way_cnt=0; way_cnt<(SIMD/WAY); way_cnt++){
	  unsigned char pe=0;
	  unsigned char way_cnt=0;
	  for(unsigned short pe_way_cnt=0; pe_way_cnt<PE*(SIMD/WAY); pe_way_cnt++){
#pragma HLS UNROLL

			  // ** hwkim modified for OPTIMIZATION
//			  sf_num_buf[pe][way_cnt] = sf_num[pe*(SIMD/WAY)+way_cnt].read();
			  sf_num_buf[pe][way_cnt] = sf_num[pe_way_cnt].read();

			  if(sf_max < sf_num_buf[pe][way_cnt])
				  sf_max = sf_num_buf[pe][way_cnt];
//		  }	// ** hwkim modified for OPTIMIZATION - pe/way loop integration
			  // ** hwkim added for OPTIMIZATION
			  if(++way_cnt==(SIMD/WAY)){
				  pe++;
				  way_cnt=0;
			  }
	  }

	  for(sf=0; sf<SF*SIMD; sf+=SIMD){
#pragma HLS PIPELINE II=1 rewind
//		  if((~sf_sync_flag)==0)
//		  if(sf_max < sf)
//			  break;

		  for(unsigned char pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
//			  if(sf_pe[pe] == 0) {
			  if(sf == 0) {

				  accu[pe] = activation.init(nf, pe);
#ifdef ACTIVATION_LOG
				  accu_pm[pe] = activation.init(nf, pe);
#endif
			  }
		  }

		  // hwkim modified for OPTIMIZATION
//		  for(unsigned char pe = 0; pe < PE; pe++) {
//#pragma HLS UNROLL
//			  for(unsigned char way_cnt=0; way_cnt<(SIMD/WAY); way_cnt++){	// hwkim added for way
		  pe = 0;
		  way_cnt = 0;
		  for(unsigned short pe_way_cnt=0; pe_way_cnt<PE*(SIMD/WAY); pe_way_cnt++){
#pragma HLS UNROLL

				  if(sf < sf_num_buf[pe][way_cnt]){

					  // hwkim modified for mask
//					  ap_uint<WAY> mask;
//					  if((sf+WAY) <= sf_num_buf[pe][way_cnt])
//						  mask = ~0x0;
//					  else
//						  mask = (~(ap_uint<WAY>)0x0) >> (WAY-(sf_num_buf[pe][way_cnt]%WAY));
					  // ** hwkim modified for OPTIMIZATION
//					  mask = ~packed_mask[pe*(SIMD/WAY)+way_cnt].read();
					  mask = ~packed_mask[pe_way_cnt].read();

				  	  // hwkim added and modified for ternary & debug (inElem_pe and dummy_w can be removed)
//					  auto const  act = TSrcI()(packed_input[pe].read());
//					  auto const  wgt = TWeightI()(packed_weight[pe].read());
					  // hwkim modified for way
//					  inElem_pe[pe] = packed_input[pe].read();
//					  auto const  act = TSrcI()(inElem_pe[pe]);
//					  ap_uint<SIMD> dummy_w = packed_weight[pe].read();
//					  auto const  wgt = TWeightI()(dummy_w);
					  // ** hwkim modified for OPTIMIZATION
//					  inElem_pe[pe] = packed_input[pe*(SIMD/WAY)+way_cnt].read();
//					  ap_uint<SIMD> dummy_w = packed_weight[pe*(SIMD/WAY)+way_cnt].read();
					  inElem_pe[pe] = packed_input[pe_way_cnt].read();
					  ap_uint<SIMD> dummy_w = packed_weight[pe_way_cnt].read();

					  auto const  act = TSrcI()(inElem_pe[pe]);
					  auto const  wgt = TWeightI()(dummy_w);


				  	  // hwkim modified for ternary
//					  accu[pe] = mac<SIMD>(accu[pe], wgt, act, r		// hwkim modified for activation comparison using +- accumulation
//#ifdef ACTIVATION_LOG
//							  	  	  	  	  , 0
//#endif
//					  	  	  	  	  	  	  );
					  // hwkim modified for way
//				  	  accu[pe] = mac_masked<SIMD, SIMD>(accu[pe], wgt, act, r, mask);
					  // hwkim modified for mask
					  accu[pe] = mac_masked<WAY, WAY>(accu[pe], wgt, act, r, mask);	// should be modified - N = WAY

#ifdef ACTIVATION_LOG
					  // hwkim modified for way
//				  	  accu_pm[pe] = mac<SIMD>(accu_pm[pe], wgt, act, r, 1);
					  if(TSrcI::width==1)	// hwkim: rest layers except for the first layer
						  accu_pm[pe] = mac_masked_pm<WAY, WAY>(accu[pe], wgt, act, r, mask);	// should be modified - N = WAY


				  	  // hwkim added for debug
//					  cout << fixed;
//					  cout.precision(8);
//					  cout << "nf " << (int)nf << ", accu[" << (int)pe << "]: " << accu[pe];
//					  cout << hex << "\tact: " << inElem_pe[pe] << "\twgt: " << dummy_w << endl;
#endif
			  	  }
//			  }	// ** hwkim commented for OPTIMIZATION
				  // ** hwkim added for OPTIMIZATION
				  if(++way_cnt==(SIMD/WAY)){
					  pe++;
					  way_cnt=0;
				  }
		  }
		  // ** hwkim modified for OPTIMIZATION
		  if(sf_max <= sf)
//		  if(~mask)
			  break;
	  }

	  for(unsigned char pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
//		  if(sf == (sf_max - 1)){

// hwkim modified for debug
// hwkim commented for 0-1 accumulation - not +- accum
#ifdef ACTIVATION_LOG
			  if(TSrcI::width!=1)	// hwkim: for the first layer
				  accu_pm[pe] = accu[pe];

			  // write conv_out_minus_bias
			  conv_out_log_file << fixed;
			  conv_out_log_file.precision(8);
			  conv_out_log_file.setf(ios_base::showpoint);
			  conv_out_log_file <<  accu_pm[pe] << endl;//" | ";

			  //compare golden with computed
			  decltype(activation.init(0,0))	golden_buf;
//				  float golden_buf;
			  golden_conv_out_file >> golden_buf;
//				  float accu_pm_tmp = accu[pe].to_float();
			  if(accu_pm[pe]!=golden_buf){
//				  if(accu_pm_tmp!=golden_buf){
				  conv_out_comp_file << fixed;
				  conv_out_comp_file.precision(8);
				  conv_out_comp_file.setf(ios_base::showpoint);
				  conv_out_comp_file << "differ @ (y " << y << ",x " << x << ")";
				  conv_out_comp_file << "gold: " << golden_buf << ", accu[nf " << (int)nf << ",pe " << (int)pe << "]: " << accu[pe] << endl;

//				  cout.fixed;
//				  cout.precision(8);
//				  cout << "differ @ (y " << y << ",x " << x << "), " << "nf " << (int)nf;
//				  cout << "accu[" << (int)pe << "]: " << accu[pe] << endl;
			  }
#endif
			  // hwkim modified for ternary
//			  TDstElem outElem_unit;
//			  outElem_unit = activation.activate(nf, pe, accu[pe], sf_num_buf[pe]);
			  ap_uint<TDstElem::width+1> outElem_ter_unit;	// include zero mask
			  // hwkim modified for way
//			  outElem_ter_unit = activation.activate(nf, pe, accu[pe], sf_num_buf[pe]);
			  outElem_ter_unit = activation.activate(nf, pe, accu[pe], sf_num_buf[pe][0]);

			  ap_uint<1> outMaskElem_unit;
			  outMaskElem_unit = (ap_uint<1>)outElem_ter_unit[TDstI::width];
			  TDstElem outElem_unit;
			  outElem_unit = (TDstElem)outElem_ter_unit[TDstI::width-1];

			  // hwkim modified for ternary
			  outElem[pe] = *reinterpret_cast<ap_uint<TDstI::width> *>(&outElem_unit);
			  outMaskElem[pe] = outMaskElem_unit;
//		  }
	  }

	  // sync all PE here - per every sliding window
	  out.write(outElem);
	  out_mask.write(outMaskElem);

// hwkim added for debug
#ifdef ACTIVATION_LOG
		  act_buf_arr[nf] = outElem;
		  act_mask_buf_arr[nf] = outMaskElem;
//			  cout << hex << (unsigned int)act_buf_arr[nf] << endl;
		  if(nf==(NF-1)){
	  		act_buf = 0;
	  		act_mask_buf = 0;
	  		for(int nf_cnt=NF-1; nf_cnt>=0; nf_cnt--){
	  			// hwkim: filling act_buf
	  			act_buf = act_buf << OutWidth;	//OutWidth or PE
	  			act_buf = act_buf | act_buf_arr[nf_cnt];

	  			act_mask_buf = act_mask_buf << PE;	//OutWidth or PE
				act_mask_buf = act_mask_buf | act_mask_buf_arr[nf_cnt];
	  		}

	  		// hwkim: compare activation with gold result
	  		ap_uint<NF*OutWidth> gold_buf = 0;
	  		char gold_buf_ch[(NF*OutWidth)/4+1];
	  		char gold_buf_ch64[17];
	  		gold_buf_ch64[16] = 0;
	  		unsigned long gold_buf_long;

	  		if(compare_skip==0){

	  			// hwkim: write activation log
	  			activation_log_file << uppercase << hex;
  				if((NF*OutWidth)>=64){
  					for(int i=0; i<(NF*OutWidth)/64; i++){
  						activation_log_file << (unsigned long long )(act_buf >> 64*(NF*OutWidth/64-i-1));
  					}
  					activation_log_file << endl;
  				}
  				else{
  					activation_log_file << (unsigned long long )act_buf << endl;
  				}

  				// hwkim: write activation mask log
	  			activation_mask_log_file << uppercase << hex;
  				if((NF*PE)>=64){
  					for(int i=0; i<(NF*PE)/64; i++){
  						activation_mask_log_file << (unsigned long long )(act_mask_buf >> 64*(NF*PE/64-i-1));
  					}
  					activation_mask_log_file << endl;
  				}
  				else{
  					activation_mask_log_file << (unsigned long long )act_mask_buf << endl;
  				}

	  			// read golden results (activation)
	  			fscanf(golden_file, "%s", gold_buf_ch);
	  			for(int word_cnt=0; word_cnt<(NF*OutWidth)/64; word_cnt++){	// there's no layers with channel count smaller than 64
	  				for(int i=0; i<64/4; i++){
	  					gold_buf_ch64[i] = gold_buf_ch[word_cnt*16+i];
	  				}
	  				gold_buf_long = strtoul(gold_buf_ch64, NULL, 16);
	  				gold_buf = gold_buf << 64;
	  				gold_buf = gold_buf | (*reinterpret_cast<ap_uint<64> *>(&gold_buf_long));
	  			}

	  			if(act_buf!=gold_buf){
	  				act_comp_file << dec << "@(" << setw(2) << y << "," << setw(2) << x << ")" <<
	  						hex << " golden: ";
	  				if((NF*OutWidth)>=64){
	  					for(int i=0; i<(NF*OutWidth)/64; i++){
	  						act_comp_file << (unsigned long long )(gold_buf >> 64*(NF*OutWidth/64-i-1));
	  					}
	  					act_comp_file << "," << endl << setw(17) << hex << "act: ";
	  					for(int i=0; i<(NF*OutWidth)/64; i++){
	  						act_comp_file << (unsigned long long )(act_buf >> 64*(NF*OutWidth/64-i-1));
	  					}
	  					act_comp_file << endl;
	  				}
	  				else{
	  					act_comp_file << (unsigned long long )gold_buf << "," << setw(17);
	  					act_comp_file << hex << "act: "  << (unsigned long long )act_buf << endl;
	  				}
	  			}

	  			// read golden results (zero mask)
	  			fscanf(golden_mask_file, "%s", gold_buf_ch);
	  			for(int word_cnt=0; word_cnt<(NF*OutWidth)/64; word_cnt++){	// there's no layers with channel count smaller than 64
	  				for(int i=0; i<64/4; i++){
	  					gold_buf_ch64[i] = gold_buf_ch[word_cnt*16+i];
	  				}
	  				gold_buf_long = strtoul(gold_buf_ch64, NULL, 16);
	  				gold_buf = gold_buf << 64;
	  				gold_buf = gold_buf | (*reinterpret_cast<ap_uint<64> *>(&gold_buf_long));
	  			}

	  			if(act_mask_buf!=gold_buf){
	  				mask_comp_file << dec << "@(" << setw(2) << y << "," << setw(2) << x << ")";
	  				mask_comp_file << hex << " golden: ";
	  				if((NF*PE)>=64){
	  					for(int i=0; i<(NF*PE)/64; i++){
	  						mask_comp_file << (unsigned long long )(gold_buf >> 64*(NF*PE/64-i-1));
	  					}
	  					mask_comp_file << "," << endl << setw(17) << hex << "act: ";
	  					for(int i=0; i<(NF*PE)/64; i++){
	  						mask_comp_file << (unsigned long long )(act_mask_buf >> 64*(NF*PE/64-i-1));
	  					}
	  					mask_comp_file << endl;
	  				}
	  				else{
	  					mask_comp_file << (unsigned long long )gold_buf << "," << setw(17);
	  					mask_comp_file << hex << "act: "  << (unsigned long long )act_mask_buf << endl;
	  				}
	  			}

	  		}
	  		if(x==0)
				cout << dec << y << "/" << OFMHeight << endl;
	  	}
#endif
		  if(++nf == NF) {
			  nf = 0;
			  if(++x==OFMDim){
				  x=0;
				  if(++y==OFMHeight){
					  y=0;
				  }
			  }
		  }
  }

#ifdef ACTIVATION_LOG
  conv_out_log_file.close();
  conv_out_comp_file.close();
  last_layer_scaled_file.close();
#endif
}

template<unsigned MatrixW, unsigned MatrixH,
	unsigned SIMD, unsigned PE,
	unsigned OFMDim, unsigned OFMHeight,
	unsigned Top, unsigned Bottom, unsigned Left, unsigned Right,
	unsigned SrcWidth,
	unsigned WAY,
#ifdef ACTIVATION_LOG
	unsigned LayerCnt,
#endif
	typename TI, typename TIM, typename TW, typename TM,
	typename PI, typename PW, typename FI, typename PM
  >
void nonzero_activation_weight_stream_gen(
		hls::stream<TI> & in,
		hls::stream<TIM> & in_mask,
		TW  const &weights,
		TM  const &wmasks,
		hls::stream<PI>* packed_input,
		hls::stream<PW>* packed_weight,
		hls::stream<FI>* sf_num,
		hls::stream<PM>* packed_mask,
		int const reps
)
{
#if defined (ACTIVATION_LOG) & defined (DEBUG)
	string nonz_dbg_file_name = "nonz_" + to_string(LayerCnt+1) + "debug.txt";
	ofstream nonz_dbg_file(nonz_dbg_file_name);
	if(!nonz_dbg_file.is_open())	cout << nonz_dbg_file_name << " open error!" << endl;
#endif
#ifdef ACTIVATION_LOG
	extern string snapshot_dir;

	ofstream nonz_i_log_file[PE][SIMD/WAY];
	ofstream nonz_w_log_file[PE][SIMD/WAY];
	ofstream nonz_f_log_file[PE][SIMD/WAY];

	for(unsigned char pe=0; pe<PE; pe++){
		for(unsigned char way_cnt=0; way_cnt<(SIMD/WAY); way_cnt++){
			string nonz_i_log_file_name = snapshot_dir + "nonzero_" + to_string(LayerCnt+1)
					+ "_" + to_string(pe) + "_" + to_string(way_cnt) + "_input_log.txt";
			string nonz_w_log_file_name = snapshot_dir + "nonzero_" + to_string(LayerCnt+1)
					+ "_" + to_string(pe) + "_" + to_string(way_cnt) + "_weight_log.txt";
			string nonz_f_log_file_name = snapshot_dir + "nonzero_" + to_string(LayerCnt+1)
					+ "_" + to_string(pe) + "_" + to_string(way_cnt) + "_fanin_log.txt";

			nonz_i_log_file[pe][way_cnt].open(nonz_i_log_file_name);
			nonz_w_log_file[pe][way_cnt].open(nonz_w_log_file_name);
			nonz_f_log_file[pe][way_cnt].open(nonz_f_log_file_name);

			if(!nonz_i_log_file[pe][way_cnt].is_open())	cout << nonz_i_log_file_name << " open error!" << endl;
			if(!nonz_w_log_file[pe][way_cnt].is_open())	cout << nonz_w_log_file_name << " open error!" << endl;
			if(!nonz_f_log_file[pe][way_cnt].is_open())	cout << nonz_f_log_file_name << " open error!" << endl;

			nonz_i_log_file[pe][way_cnt] << hex;
			nonz_w_log_file[pe][way_cnt] << hex;
			nonz_f_log_file[pe][way_cnt] << dec;
		}
	}
#endif

	unsigned char const  NF = MatrixH / PE;
	unsigned char const  SF = MatrixW / SIMD;
	unsigned short const TOTAL_FOLD = NF * SF;
	// ** hwkim added for pipeline expansion
	unsigned short const TOTAL_FOLD_NEW = NF * (SF+2);

	// ** hwkim added for OPTIMIZATION
	TI inElem;
	TIM imaskElem;
	ap_uint<SIMD> wmaskElem[PE];
	ap_uint<SIMD> wgtElem[PE];

	TI	inputBuf[SF];
	TIM	imaskBuf[SF];

	ap_uint<WAY>	mask_pack_ping[PE*SIMD/WAY];
	ap_uint<WAY>	mask_pack_pong[PE*SIMD/WAY];
	ap_uint<SrcWidth*WAY>	input_pack_ping[PE*SIMD/WAY];
	ap_uint<SrcWidth*WAY>	input_pack_pong[PE*SIMD/WAY];
	ap_uint<WAY> w_pack_ping[PE*SIMD/WAY];
	ap_uint<WAY> w_pack_pong[PE*SIMD/WAY];
	ap_uint<WAY>	mask_delay_buf[PE*SIMD/WAY];
	ap_uint<SrcWidth*WAY>	input_delay_buf[PE*SIMD/WAY];
	ap_uint<WAY> w_delay_buf[PE*SIMD/WAY];
	FI	sf_cnt[PE*SIMD/WAY];

	ap_uint<2> kx=0, ky=0;
	unsigned short x=0, y=0;
	unsigned char const sf_ch = SF/9;
	unsigned char nf   = 0;
	unsigned char sf_ch_cnt=0;
	unsigned char sf   = 0;
	unsigned char tile = 0;
	unsigned char pe=0;
	unsigned char way_cnt=0;

	unsigned char const way_num = SIMD/WAY;


	// hwkim modified for dependency
//	for(unsigned  i = 0; i < reps * TOTAL_FOLD; i++) {
	// ** hwkim modified for pipeline expansion
//	for(unsigned  i = 0; i < reps * TOTAL_FOLD / SF; i++) {
	for(unsigned  i = 0; i < reps * TOTAL_FOLD_NEW; i++) {
//#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II=1

		// ** hwkim modified for pipeline expansion
//		if(nf==0){
		if(nf==0 && sf==0){
			tile = 0;
		}

		for(unsigned short pe_way_cnt=0; pe_way_cnt<PE*(SIMD/WAY); pe_way_cnt++){
#pragma HLS UNROLL
			if(sf==0)	// ** hwkim added for pipeline expansion
				sf_cnt[pe_way_cnt] = 0;
		}

		// ** hwkim commented for pipeline expansion
//		for(sf=0; sf<SF+2; sf++){	// for 2 initialization cycles
//#pragma HLS PIPELINE II=1

			if((sf>=SF) ||
					((x<Left && kx<Left)
					||(y<Top && ky<Top)
					||(x>(OFMDim-1-Right) && kx>(3-1-Right))
					||(y>(OFMHeight-1-Bottom) && ky>(3-1-Bottom)))){
				inElem = TI(~0x0);
				imaskElem = ~0x0;
			}
			else if(nf==0){
				inElem = in.read();
				imaskElem = in_mask.read();
			}
			else{
				inElem = inputBuf[sf];
				imaskElem = imaskBuf[sf];
			}

			if((sf<SF) && (nf==0)){
				inputBuf[sf] = inElem;
				imaskBuf[sf] = imaskElem;
			}

			if(sf<SF){
				auto const &w = weights.weights(tile);
				auto const &wm = wmasks.masks(tile);
				for(unsigned char pe=0; pe<PE; pe++){
#pragma HLS UNROLL
					wgtElem[pe] = w[pe];
					wmaskElem[pe] = wm[pe];
				}
			}

#if defined(ACTIVATION_LOG) & defined(DEBUG)
			cout << "(" << y << "," << x << ") ";
			cout << "[" << ky << "," << kx << "] ";
			cout << "nf: " << (int)nf << " sf: " << (int)sf;
			cout << "-------------------------------------" << endl;
#endif

			pe = 0;
			way_cnt = 0;
			for(unsigned short pe_way_cnt=0; pe_way_cnt<PE*(SIMD/WAY); pe_way_cnt++){
#pragma HLS UNROLL
				ap_uint<WAY> new_mask = 0;

				unsigned char next_bit_start = 0;
				for(unsigned char prev_bit_cnt=0; prev_bit_cnt<WAY; prev_bit_cnt++){
					for(unsigned char next_bit_cnt=0; next_bit_cnt<WAY; next_bit_cnt++){
#pragma HLS UNROLL
						if(mask_pack_ping[pe_way_cnt][prev_bit_cnt]
							   && (mask_delay_buf[pe_way_cnt][next_bit_cnt]==0)
							   && (next_bit_cnt >= next_bit_start)){

							new_mask[next_bit_cnt] = 1;

							mask_pack_pong[pe_way_cnt][prev_bit_cnt] = 0;
							input_pack_pong[pe_way_cnt](prev_bit_cnt*SrcWidth+(SrcWidth-1), prev_bit_cnt*SrcWidth)
								= input_delay_buf[pe_way_cnt](next_bit_cnt*SrcWidth+(SrcWidth-1), next_bit_cnt*SrcWidth);
							w_pack_pong[pe_way_cnt][prev_bit_cnt] = w_delay_buf[pe_way_cnt][next_bit_cnt];

							next_bit_start = next_bit_cnt+1;
							break;
						}
						else{
							mask_pack_pong[pe_way_cnt][prev_bit_cnt] = mask_pack_ping[pe_way_cnt][prev_bit_cnt];	// current input
							input_pack_pong[pe_way_cnt](prev_bit_cnt*SrcWidth+(SrcWidth-1), prev_bit_cnt*SrcWidth)
								= input_pack_ping[pe_way_cnt](prev_bit_cnt*SrcWidth+(SrcWidth-1), prev_bit_cnt*SrcWidth);
							w_pack_pong[pe_way_cnt][prev_bit_cnt] = w_pack_ping[pe_way_cnt][prev_bit_cnt];
						}
					}
				}
#if defined(ACTIVATION_LOG) & defined(DEBUG)
				cout << "bit packing............." << endl;
#endif

				if(((mask_pack_ping[pe_way_cnt]==0) || ((sf==SF+1) && (~mask_pack_ping[pe_way_cnt]!=0)))
						&& (sf >= 2)){

					packed_input[pe_way_cnt].write(input_pack_ping[pe_way_cnt]);
					packed_weight[pe_way_cnt].write(w_pack_ping[pe_way_cnt]);
					packed_mask[pe_way_cnt].write(mask_pack_ping[pe_way_cnt]);
#ifdef ACTIVATION_LOG
					nonz_i_log_file[pe][way_cnt] << (unsigned long)input_pack_ping[pe_way_cnt] << endl;
					nonz_w_log_file[pe][way_cnt] << (unsigned long)w_pack_ping[pe_way_cnt] << endl;
#endif
#if defined(ACTIVATION_LOG) & defined(DEBUG)
					cout << "push ping................" << hex << input_pack_ping[pe_way_cnt] << endl;
#endif
				}
				else if((mask_pack_pong[pe_way_cnt]==0) && (sf >= 2)){
					packed_input[pe_way_cnt].write(input_pack_pong[pe_way_cnt]);
					packed_weight[pe_way_cnt].write(w_pack_pong[pe_way_cnt]);
					packed_mask[pe_way_cnt].write(mask_pack_pong[pe_way_cnt]);
#ifdef ACTIVATION_LOG
					nonz_i_log_file[pe][way_cnt] << (unsigned long)input_pack_pong[pe_way_cnt] << endl;
					nonz_w_log_file[pe][way_cnt] << (unsigned long)w_pack_pong[pe_way_cnt] << endl;
#endif
#if defined(ACTIVATION_LOG) & defined(DEBUG)
					cout << "push pong................" << hex << input_pack_pong[pe_way_cnt] << endl;
#endif
				}

				if((((mask_pack_ping[pe_way_cnt]==0) || ((sf==SF+1) && (~mask_pack_ping[pe_way_cnt]!=0)))
						|| (mask_pack_pong[pe_way_cnt]==0))
						&& (sf >= 2)){
					sf_cnt[pe_way_cnt]+=WAY;
				}

				if((sf < 2) || (mask_pack_ping[pe_way_cnt]==0)){
					mask_pack_ping[pe_way_cnt] = mask_delay_buf[pe_way_cnt];
				}
				else if(mask_pack_pong[pe_way_cnt]==0){
					mask_pack_ping[pe_way_cnt] = mask_delay_buf[pe_way_cnt] | new_mask;
				}
				else{
					mask_pack_ping[pe_way_cnt] = (mask_delay_buf[pe_way_cnt] | new_mask) & mask_pack_pong[pe_way_cnt];
				}

				if((sf < 2) || (mask_pack_ping[pe_way_cnt]==0)
						|| (mask_pack_pong[pe_way_cnt]==0)){
					input_pack_ping[pe_way_cnt] = input_delay_buf[pe_way_cnt];
					w_pack_ping[pe_way_cnt] = w_delay_buf[pe_way_cnt];
				}
				else{
					input_pack_ping[pe_way_cnt] = input_pack_pong[pe_way_cnt];
					w_pack_ping[pe_way_cnt] = w_pack_pong[pe_way_cnt];
				}

				mask_delay_buf[pe_way_cnt] = (wmaskElem[pe] | imaskElem) >> (way_cnt*WAY);
				input_delay_buf[pe_way_cnt] = inElem >> (way_cnt*WAY*SrcWidth);
				w_delay_buf[pe_way_cnt] = wgtElem[pe] >> (way_cnt*WAY);

#if defined(ACTIVATION_LOG) & defined(DEBUG)
				cout << dec << "***** pe: " << (int)pe << " way: " << (int)way_cnt << endl;
				cout << hex;
				cout << "m_buf: " << mask_delay_buf[pe_way_cnt];
				cout << " i_buf: " << input_delay_buf[pe_way_cnt];
				cout << " w_buf: " << w_delay_buf[pe_way_cnt] << endl;
				cout << "m_pac: " << mask_pack_ping[pe_way_cnt];
				cout << " i_pac: " << input_pack_ping[pe_way_cnt];
				cout << " w_pac: " << w_pack_ping[pe_way_cnt] << endl;
				cout << "m_pac: " << mask_pack_pong[pe_way_cnt];
				cout << " i_pac: " << input_pack_pong[pe_way_cnt];
				cout << " w_pac: " << w_pack_pong[pe_way_cnt] << endl;
				cout << dec;
				cout << "sf_cnt: " << sf_cnt[pe_way_cnt] << endl;
#endif

				if(sf==SF+1){
					sf_num[pe_way_cnt].write(sf_cnt[pe_way_cnt]);
#ifdef ACTIVATION_LOG
					nonz_f_log_file[pe][way_cnt] << (unsigned long)sf_cnt[pe_way_cnt] << endl;
#endif
				}

				if(++way_cnt==(SIMD/WAY)){
					way_cnt=0;
					pe++;
				}
			}

			if(sf<SF){
				tile++;
				if(++sf_ch_cnt==sf_ch){
					sf_ch_cnt=0;
					if(++kx==3){
						kx=0;
						if(++ky==3){
							ky=0;
							if(++nf==NF){
								nf=0;
								if(++x==OFMDim){
									x=0;
									if(++y==OFMHeight){
										y=0;
									}
#ifdef ACTIVATION_LOG
								cout << "nonz func: " << y << "/" << OFMHeight << endl;
#endif
								}
							}
						}
					}
				}
			}
//		}	// ** hwkim commented for pipeline expansion
		// ** hwkim added for pipeline expansion
		if(++sf==(SF+2)){
			sf = 0;
		}
	}

#if defined (ACTIVATION_LOG) & defined (DEBUG)
	nonz_dbg_file.close();
#endif

#ifdef ACTIVATION_LOG
	for(unsigned char pe=0; pe<PE; pe++){
		for(unsigned char way_cnt=0; way_cnt<(SIMD/WAY); way_cnt++){
			nonz_i_log_file[pe][way_cnt].close();
			nonz_w_log_file[pe][way_cnt].close();
			nonz_f_log_file[pe][way_cnt].close();
		}
	}
#endif

}

#endif
