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
/******************************************************************************
 *
 *
 * @file top.cpp
 *
 * HLS Description of the CNV BNN with axi-lite based parameter loading (DoMemInit) 
 * and  dataflow architecture of the image inference (DoCompute).
 * The network uses 1 bit weights and 1 bit activation.
 *
 *****************************************************************************/
#include "config.h"

#include "bnn-library.h"

#include "weights.hpp"
#include "activations.hpp"
#include "interpret.hpp"
#include "mvau.hpp"

static BinaryWeights<L0_SIMD, L0_PE, L0_WMEM>  weights0;
static BinaryWeights<L1_SIMD, L1_PE, L1_WMEM>  weights1;
static BinaryWeights<L2_SIMD, L2_PE, L2_WMEM>  weights2;
static BinaryWeights<L3_SIMD, L3_PE, L3_WMEM>  weights3;
static BinaryWeights<L4_SIMD, L4_PE, L4_WMEM>  weights4;
static BinaryWeights<L5_SIMD, L5_PE, L5_WMEM>  weights5;
static BinaryWeights<L6_SIMD, L6_PE, L6_WMEM>  weights6;
//static BinaryWeights<L7_SIMD, L7_PE, L7_WMEM>  weights7;
//static BinaryWeights<L8_SIMD, L8_PE, L8_WMEM>  weights8;

static ThresholdsActivation<L0_TMEM, L0_PE, L0_API, ap_fixed<24, 16>, ap_uint<L0_API> > threshs0;
static ThresholdsActivation<L1_TMEM, L1_PE, L1_API, ap_int<16>, ap_uint<L1_API>>  		threshs1;
static ThresholdsActivation<L2_TMEM, L2_PE, L2_API, ap_int<16>, ap_uint<L2_API>>  		threshs2;
static ThresholdsActivation<L3_TMEM, L3_PE, L3_API, ap_int<16>, ap_uint<L3_API>>  		threshs3;
static ThresholdsActivation<L4_TMEM, L4_PE, L4_API, ap_int<16>, ap_uint<L4_API>>  		threshs4;
static ThresholdsActivation<L5_TMEM, L5_PE, L5_API, ap_int<16>, ap_uint<L5_API>>  		threshs5;
//static ThresholdsActivation<L6_TMEM, L6_PE, L6_API, ap_int<16>, ap_uint<L6_API>>  		threshs6;
//static ThresholdsActivation<L7_TMEM, L7_PE, L7_API, ap_int<16>, ap_uint<L7_API>>  		threshs7;
/* hwkim commented
 * 8 layer는 없음
 * 마지막 layer라 thresholding(activation) 안 하고,
 * pass through activation
 */

unsigned int paddedSizeHW(unsigned int in, unsigned int padTo) {
  if(in % padTo == 0) {
    return in;
  } else {
    return in + padTo - (in % padTo);
  }
}

void DoMemInit(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd,
		unsigned int targetThresh, ap_uint<64> val) {
  switch (targetLayer) {
    case 0:
      weights0.m_weights[targetMem][targetInd] = val;
      /* hwkim commented
       *
       * weights0
       * 	-> static BinaryWeights<L0_SIMD, L0_PE, L0_WMEM>  weights0;
       * 	-> m_weights를 멤버로 갖는 class
       *
       * m_weights
       * 	-> ap_uint<SIMD>  m_weights[PE][TILES];
       * 	-> SIMD - input channel number가 아님
       * 		>> input channel 중 SIMD 개수만큼 한 번에 한다는 말
       * 	-> PE - targetMem - output channel number가 아님
       * 		>> PE 별 - layer 0은 16개 PE - out ch가 64니까 PE 당 4번 씩 함
       * 		>> PE가 16개라는 말은 한 번에 output channel 16개씩 계산한다는 말
       * 	-> TILES - target ind(ex?) - 64-bit 단위 index
       * 		>> kernel의 다음 pixel 위치에 해당하는 x-channel ap_uint<x>
       * 		>> PE 별 weight 들
       * 			TILES == PE 당 가지는 weight 개수
       * 			PE 당 가지는 weight kernel 수는 out_ch/PE
       * 				layer0의 경우 64/16=4.
       * 				즉, 16개의 PE가 16개 output channel 동시에 계산,
       * 				즉, 각 PE는 output channel 4개 씩 수행
       * 					(4개의 weight kernel에 대해 conv 수행)
       * 				따라서 TILES == 36 = 3x3x4
       *
       * layer 0의 경우, SIMD가 3이기 때문에, weights0.m_weights는 ap_uint<3>인데 64-bit짜리 val을 넣음
       * 	-> weight file에서 64-bit에 3-bit만 저장되어 있음
       */
      break;
    case 1:
      threshs0.m_thresholds[targetMem][targetInd][targetThresh] = *reinterpret_cast<ap_fixed<64, 56> *>(&val);
      /* hwkim comment
       *
       * threshs0
       * 	-> static ThresholdsActivation< , L0_PE, L0_API, ap_fixed<24, 16>, ap_uint<L0_API> > threshs0;
       * 	-> m_thresholds를 멤버로 갖는 class
       * m_thresholds
       * 	-> TA m_thresholds[PE][NF][NumTH]; <-L0_API
       *                  L0_PE^   ^L0_TMEM(4)
       * 	-> TA (Threshold Accuracy?)
       * 		layer 0의 경우 ap_fixed<24, 16> - 24-bit fixed point (with sign), 16-bit int/8-bit fractal
       * 		다른 layer의 경우 ap_int<16>
       * 	-> NF - targetInd
       * 		L0_TMEM(4) - 64-bit line(output channel?) 단위
       * 		layer 0의 L0_TMEM(NF)가 4인 이유
       * 			각 PE 당, output channel 4개 씩 계산함
       * 			output channel 당 batch normalization 수행하므로 threshold가 1개 존재
       * 	-> PE - 걍 PE number - targetMem
       * 	-> NumTH - output channel 당 여러 개의 threshold가 있을 수 있는 듯
       * 		하지만 여기서는 1개만 있음
       * targetMem - PE number
       * targetInd - 64-bit line(pixel) index
       * 	layer 0의 경우 0~3으로 4개
       * targetThresh - 0 고정
       */
      break;
    case 2:
      weights1.m_weights[targetMem][targetInd] = val;
      break;
    case 3:
      threshs1.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 4:
      weights2.m_weights[targetMem][targetInd] = val;
      break;
    case 5:
      threshs2.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 6:
      weights3.m_weights[targetMem][targetInd] = val;
      break;
    case 7:
      threshs3.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 8:
      weights4.m_weights[targetMem][targetInd] = val;
      break;
    case 9:
      threshs4.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 10:
      weights5.m_weights[targetMem][targetInd] = val;
      break;
    case 11:
      threshs5.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 12:
      weights6.m_weights[targetMem][targetInd] = val;
      break;
    case 13:
//      threshs6.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
//    case 14:
//      weights7.m_weights[targetMem][targetInd] = val;
//      break;
//    case 15:
//      threshs7.m_thresholds[targetMem][targetInd][targetThresh] = val;
//      break;
//    case 16:
//      weights8.m_weights[targetMem][targetInd] = val;
//      break;
//    case 17:
//      // do nothing, no thres mem for layer 8 as PassThrough activation is used
//      break;
  }
}

void DoCompute(ap_uint<64> *in, ap_uint<64>* out, const unsigned int numReps) {
#pragma HLS DATAFLOW
  stream<ap_uint<64>> inter0("DoCompute.inter0");
  stream<ap_uint<192>> inter0_1("DoCompute.inter0_1");
  stream<ap_uint<24>> inter0_2("DoCompute.inter0_2");
#pragma HLS STREAM variable=inter0_2 depth=128
  /* hwkim commented
   *  단순 FIFO size 지정
   */
  stream<ap_uint<64>> inter1("DoCompute.inter1");
#pragma HLS STREAM variable=inter1 depth=128
  stream<ap_uint<64>> inter2("DoCompute.inter2");
  stream<ap_uint<64>> inter3("DoCompute.inter3");
#pragma HLS STREAM variable=inter3 depth=128
  stream<ap_uint<128>> inter4("DoCompute.inter4");
#pragma HLS STREAM variable=inter4 depth=128
  stream<ap_uint<128>> inter5("DoCompute.inter5");
  stream<ap_uint<128>> inter6("DoCompute.inter6");
#pragma HLS STREAM variable=inter6 depth=81
  stream<ap_uint<256>> inter7("DoCompute.inter7");
#pragma HLS STREAM variable=inter7 depth=1
  stream<ap_uint<256>> inter8("DoCompute.inter8");
#pragma HLS STREAM variable=inter8 depth=1
  stream<ap_uint<64>> inter9("DoCompute.inter9");
#pragma HLS STREAM variable=inter9 depth=128
  stream<ap_uint<64>> inter10("DoCompute.inter10");
#pragma HLS STREAM variable=inter10 depth=3
  stream<ap_uint<64>> memOutStrm("DoCompute.memOutStrm");

  const unsigned int inBits = 32 * 32 * 3 * 8;
  // const unsigned int inBitsPadded = paddedSize(inBits, 64);
//  const unsigned int outBits = L8_MH*16;
  const unsigned int outBits = L6_MH*16;

  Mem2Stream_Batch<64, inBits / 8>(in, inter0, numReps);
  /* hwkim commented
   * dma.h에 선언
   *
   * 64 -> word bit 수 - stream해 올 interface(AXI)의 data width
   * inBits -> image 1장 당 총 bit 수
   * inBits/8 -> image 1장 당 총 byte 수
   *
   * 이 함수는 in의 주소가 가리키는 DRAM에서 inter0 stream으로
   * image를 numReps만큼 streaming해 오는 함수
   * (16 image 단위로 pipelining해서 streaming)
   * (single image일 경우, no pipelining)
   *
   * in stream의 ordering
   * 	-> c->x->y
   * 	-> 64-bit에 위의 ordering으로 8개 꽉 채움
   */
  StreamingDataWidthConverter_Batch<64, 192, (32 * 32 * 3 * 8) / 64>(inter0, inter0_1, numReps);
  /* hwkim commented
   * < InWidth, OutWidth, NumInWords >
   * InWidth를 OutWidth로 NumInWords(InWidth의 word 수)만큼 변환
   * inter0은 interleave(+quantized+packed)된 channel first order (c->x->y)
   * 192-bit에 3채널 64 pixel 들어감
   * 64-bit에 8개 들어가는데, channel이 3채널 씩이라 애매하게 끊어지므로,
   * 	192-bit으로 변환하고 아래에서 24-bit 단위(채널 묶음)으로 다시 변환
   */
  StreamingDataWidthConverter_Batch<192, 24, (32 * 32 * 3 * 8) / 192>(inter0_1, inter0_2, numReps);
  /* hwkim commented
   * 내부에 memory가 있는게 아니라 stream이므로,
   * 	width 변환해주는 hardware가 내부에 필요함
   * 24-bit에는 3채널 짜리 pixel 1개 들어감
   * inter0_1 stream의 ordering
   * 	-> x->y
   * 	-> c는 이미 24-bit 단위로 3개 묶여서 ordering되어 있음
   */
  // convolutional layers
  ConvLayer_Batch<L0_K, L0_IFM_CH, L0_IFM_DIM, L0_OFM_CH, L0_OFM_DIM, L0_SIMD, L0_PE,
  	  	  Slice<ap_fixed<8, 1, AP_TRN, AP_SAT>>, Identity, Recast<Binary>>
	  (inter0_2, inter1, weights0, threshs0, numReps, ap_resource_lut());
  /* hwkim commented
   * template
   * 	-> ConvKernelDim, IFMch, IFMdim, OFMch, OFMdim,
   * 		SIMD, PE,
   * 		TSrcI -> src width -> Slice<ap_fixed<~~>> -> ap_fixed의 width,
   * 		TDstI -> dst width -> Identity -> width는 1,
   * 		TWeightI -> weight width -> Recast<Binary> -> width는 1
   * arguments
   * 	-> in(stream), out(stream), weight(memory), activation(memory),
   * 		reps, R(resource;LUT/DSP)
   * 	-> 여기서는 thresholding을 activation function으로 보고
   * 		activation 값으로 여김
   */

  ConvLayer_Batch<L1_K, L1_IFM_CH, L1_IFM_DIM, L1_OFM_CH, L1_OFM_DIM, L1_SIMD, L1_PE,
  	  Recast<XnorMul>>
  	  (inter1, inter2, weights1, threshs1, numReps, ap_resource_lut());

  StreamingMaxPool_Batch<L1_OFM_DIM, 2, L1_OFM_CH>(inter2, inter3, numReps);

  ConvLayer_Batch<L2_K, L2_IFM_CH, L2_IFM_DIM, L2_OFM_CH, L2_OFM_DIM, L2_SIMD, L2_PE, Recast<XnorMul>>(inter3, inter4, weights2, threshs2, numReps, ap_resource_lut());
  ConvLayer_Batch<L3_K, L3_IFM_CH, L3_IFM_DIM, L3_OFM_CH, L3_OFM_DIM, L3_SIMD, L3_PE, Recast<XnorMul>>(inter4, inter5, weights3, threshs3, numReps, ap_resource_lut());

  StreamingMaxPool_Batch<L3_OFM_DIM, 2, L3_OFM_CH>(inter5, inter6, numReps);

  ConvLayer_Batch<L4_K, L4_IFM_CH, L4_IFM_DIM, L4_OFM_CH, L4_OFM_DIM, L4_SIMD, L4_PE, Recast<XnorMul>>(inter6, inter7, weights4, threshs4, numReps, ap_resource_lut());
  ConvLayer_Batch<L5_K, L5_IFM_CH, L5_IFM_DIM, L5_OFM_CH, L5_OFM_DIM, L5_SIMD, L5_PE,
  	  Recast<XnorMul>>(inter7, inter8, weights5, threshs5, numReps, ap_resource_lut());

  // fully connected layers
//  WidthAdjustedOutputStream<16 * L8_PE, 64, L8_MH / L8_PE>  wa_out(memOutStrm, numReps);
  WidthAdjustedOutputStream<16 * L6_PE, 64, L6_MH / L6_PE>  wa_out(memOutStrm, numReps);
  /* hwkim commented
   * 마지막 output을 받기 위한 stream (16-bit integer 값 score)
   * WidthAdjustedOutputStream은 argument로 들어온 stream을
   * 	자신의 class member m_target에 연결시켜 놓고,
   * 	소멸자가 호출될 때, class member m_buffer의 내용을
   * 	width adjust하여 연결했던 stream에 채워넣음
   * 	16*L8_PE=16*4=64-bit -> 64-bit
   */
//  StreamingFCLayer_Batch<L6_MW, L6_MH, L6_SIMD, L6_PE, Recast<XnorMul>>
//    (inter8, inter9,  weights6, threshs6, numReps, ap_resource_lut());
//  StreamingFCLayer_Batch<L7_MW, L7_MH, L7_SIMD, L7_PE, Recast<XnorMul>>
//    (inter9, inter10, weights7, threshs7, numReps, ap_resource_lut());
//  StreamingFCLayer_Batch<L8_MW, L8_MH, L8_SIMD, L8_PE,
//  	  Recast<XnorMul>, Slice<ap_uint<16> >>
//    (inter10, static_cast<hls::stream<ap_uint<16 * L8_PE>>&>(wa_out),weights8, PassThroughActivation<ap_uint<16>>(), numReps, ap_resource_lut());

  StreamingFCLayer_Batch<L6_MW, L6_MH, L6_SIMD, L6_PE, Recast<XnorMul>, Slice<ap_uint<16> >>
	  (inter8,
	  static_cast<hls::stream<ap_uint<16 * L6_PE>>&>(wa_out),
	  weights6,
	  PassThroughActivation<ap_uint<16>>(),
	  numReps,
	  ap_resource_lut());

  /* hwkim commented
   * score 값을 위해 activation function (binarize) 건너뜀
   * ThresholdsActivation class 대신 PassThroughActivation class를 사용
   * 	-> 차이점은 activate 멤버 함수 수행 시, pass through든 그냥 단순히
   * 		accu 값만 return, thresholding 수행하지 않음
   * 		threshold 저장하는 m_threshold 멤버 변수도 없음
   *
   * TDstI에 ap_uint<16> 사용 -> integer output
   *
   * L8_MH(output channel 수)가 왜 64??
   * 	마지막 layer면, output이 score인데, CIFAR는 10개 또는 100개 아님?
   * 	아마 정확도를 봤을 때, CIFAR-10일텐데..
   */

  Stream2Mem_Batch<64, outBits/8>(memOutStrm, out, numReps);
  /* hwkim commented
   * output(score)를 DRAM으로 전송
   * <DataWidth, numBytes> -> 64-bit(16-bit score * 4)을
   * 	-> numBytes는 output score가 byte로 몇 개인지
   * 	-> outBits가 왜 64*16? 10*16이 아니라?
   */
}

void BlackBoxJam(ap_uint<64> *in, ap_uint<64> *out, bool doInit,
		unsigned int targetLayer, unsigned int targetMem,
		unsigned int targetInd, unsigned int targetThresh, ap_uint<64> val, unsigned int numReps) {
	/* hwkim commented
	 * numReps - number of repetitions? (input image 장 수)
	 */
// pragmas for MLBP jam interface
// signals to be mapped to the AXI Lite slave port
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=doInit bundle=control
#pragma HLS INTERFACE s_axilite port=targetLayer bundle=control
#pragma HLS INTERFACE s_axilite port=targetMem bundle=control
#pragma HLS INTERFACE s_axilite port=targetInd bundle=control
#pragma HLS INTERFACE s_axilite port=targetThresh bundle=control
#pragma HLS INTERFACE s_axilite port=val bundle=control
#pragma HLS INTERFACE s_axilite port=numReps bundle=control
// signals to be mapped to the AXI master port (hostmem)
#pragma HLS INTERFACE m_axi offset=slave port=in bundle=hostmem depth=512
#pragma HLS INTERFACE s_axilite port=in bundle=control
#pragma HLS INTERFACE m_axi offset=slave port=out bundle=hostmem depth=16
#pragma HLS INTERFACE s_axilite port=out bundle=control

// partition PE arrays
#pragma HLS ARRAY_PARTITION variable=weights0.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs0.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs0.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights1.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs1.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs1.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights2.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs2.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs2.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights3.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs3.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs3.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights4.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs4.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs4.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights5.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs5.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs5.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights6.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs6.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs6.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights7.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs7.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs7.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights8.m_weights complete dim=1

  if (doInit) {
    DoMemInit(targetLayer, targetMem, targetInd, targetThresh, val);
  } else {
    DoCompute(in, out, numReps);
  }
}
