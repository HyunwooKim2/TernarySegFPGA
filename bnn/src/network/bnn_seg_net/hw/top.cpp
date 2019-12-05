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

// hwkim modified for debug
//#define ACTIVATION_LOG
#ifdef ACTIVATION_LOG
#include <fstream>
// hwkim commented for segmentation - there's only conv(tconv) layers
//int weighted_layer_cnt = 0;
//int pooling_layer_cnt = 0;
#endif

#ifdef FPGA_DEBUG
#define ACT_LOG_LAYER	0
#endif

static BinaryWeights< L0_SIMD,  L0_PE,  L0_WMEM>  weights0;
static BinaryWeights< L1_SIMD,  L1_PE,  L1_WMEM>  weights1;
static BinaryWeights< L2_SIMD,  L2_PE,  L2_WMEM>  weights2;
static BinaryWeights< L3_SIMD,  L3_PE,  L3_WMEM>  weights3;
static BinaryWeights< L4_SIMD,  L4_PE,  L4_WMEM>  weights4;
static BinaryWeights< L5_SIMD,  L5_PE,  L5_WMEM>  weights5;
static BinaryWeights< L6_SIMD,  L6_PE,  L6_WMEM>  weights6;
static BinaryWeights< L7_SIMD,  L7_PE,  L7_WMEM>  weights7;
static BinaryWeights< L8_SIMD,  L8_PE,  L8_WMEM>  weights8;
static BinaryWeights< L9_SIMD,  L9_PE,  L9_WMEM>  weights9;
static BinaryWeights<L10_SIMD, L10_PE, L10_WMEM>  weights10;

// hwkim modified for positive only accumulation
//static ThresholdsActivation< L0_TMEM,  L0_PE,  L0_API, ap_fixed<24, 16>, ap_uint<L0_API> > threshs0;
static InputLayerActivation< L0_TMEM,  L0_PE,  L0_API, ap_fixed<24, 16>, ap_uint<L0_API> > threshs0;

static ThresholdsActivation< L1_TMEM,  L1_PE,  L1_API, ap_int<16>, ap_uint<L1_API>>  		threshs1;
static ThresholdsActivation< L2_TMEM,  L2_PE,  L2_API, ap_int<16>, ap_uint<L2_API>>  		threshs2;
static ThresholdsActivation< L3_TMEM,  L3_PE,  L3_API, ap_int<16>, ap_uint<L3_API>>  		threshs3;
static ThresholdsActivation< L4_TMEM,  L4_PE,  L4_API, ap_int<16>, ap_uint<L4_API>>  		threshs4;
static ThresholdsActivation< L5_TMEM,  L5_PE,  L5_API, ap_int<16>, ap_uint<L5_API>>  		threshs5;
static ThresholdsActivation< L6_TMEM,  L6_PE,  L6_API, ap_int<16>, ap_uint<L6_API>>  		threshs6;
static ThresholdsActivation< L7_TMEM,  L7_PE,  L7_API, ap_int<16>, ap_uint<L7_API>>  		threshs7;
static ThresholdsActivation< L8_TMEM,  L8_PE,  L8_API, ap_int<16>, ap_uint<L8_API>>  		threshs8;
static ThresholdsActivation< L9_TMEM,  L9_PE,  L9_API, ap_int<16>, ap_uint<L9_API>>  		threshs9;
// hwkim modified for last fc layer & batch norm scale
//static PassThroughAndBatchNorm<L10_TMEM, L10_PE, L10_API, ap_int<16>, ap_int<16>>			threshs10;
static PassThroughAndBatchNorm<L10_TMEM, L10_PE, L10_API, ap_int<16>, ap_fixed<24,16,AP_TRN,AP_SAT>, ap_ufixed<8,0,AP_TRN,AP_SAT>>	threshs10;
//static PassThroughAndBatchNorm<L10_TMEM, L10_PE, L10_API, ap_int<16>, ap_int<16>, ap_ufixed<8,0,AP_TRN,AP_SAT>>	threshs10;
/* hwkim commented
 * 마지막 layer라 thresholding(activation) 안 하고,
 * pass through activation
 */

#ifdef ACTIVATION_LOG
string golden_file_dir = "/home/hwkim/work/params/guinness_params/camvid_params/1017/Activations/";
string snapshot_dir = "/home/hwkim/work/params/finn_params/camvid_params/1017/snapshots/";
#endif

// hwkim added to find inferred category
template <unsigned int InStreamW, unsigned int OutStreamW>
void infer_category(
		stream<ap_uint<InStreamW>>& in,
		stream<ap_uint<OutStreamW>>& out
		)
{
#ifdef ACTIVATION_LOG
		string golden_score_file_name = golden_file_dir + "OutputScaleLayer.txt";
		ifstream golden_score_file(golden_score_file_name);
		if(!golden_score_file.is_open()){
			cout << "golden_score_file open error" << endl;
		}
		ofstream computed_score_file("computed_score.txt");
		if(!computed_score_file.is_open()){
			cout << "computed_score_file open error" << endl;
		}
		ap_fixed<24,16,AP_TRN,AP_SAT> golden_score;
#endif

		ap_uint<L10_OFM_CH*24> score_buf;
		ap_uint<24> uscore;
		ap_fixed<24,16,AP_TRN,AP_SAT> cls_p;
		ap_fixed<24,16,AP_TRN,AP_SAT> cls_n;
		ap_uint<16> label=0;
		ap_uint<64> out_buf=0;

		for(int y=0; y<L10_OFM_HEIGHT; y++){
			for(int x=0; x<L10_OFM_DIM; x++){
#pragma HLS PIPELINE II=1 rewind
				label=0;
				score_buf=in.read();
				uscore = score_buf & 0xFFFFFF;
//				if(uscore&0x800000){
//					//uscore.VAL = 0xff000000 | uscore.VAL;
//					uscore = 0xff000000 | uscore;
//				}
				cls_p = *reinterpret_cast<ap_fixed<24,16,AP_TRN,AP_SAT> *>(&uscore);
#ifdef ACTIVATION_LOG
				//cout << setw(15) << setprecision(11) << cls_p << endl;
				computed_score_file << dec << cls_p << endl;
				golden_score_file >> golden_score;
				if((golden_score>>1)!=cls_p){
					cout << "golden: " << golden_score << ", computed: " << cls_p << endl;
				}
#endif
				for(int i=1; i<L10_OFM_CH; i++){
					score_buf = score_buf >> 24;
					uscore = score_buf & 0xFFFFFF;
//					if(uscore&0x800000){
//						//uscore.VAL = 0xff000000 | uscore.VAL;
//						uscore = 0xff000000 | uscore;
//					}
					cls_n = *reinterpret_cast<ap_fixed<24,16,AP_TRN,AP_SAT> *>(&uscore);
#ifdef ACTIVATION_LOG
					//cout << setw(15) << setprecision(11) << cls_n << endl;
					computed_score_file << dec << cls_n << endl;
					golden_score_file >> golden_score;
					if((golden_score>>1)!=cls_n){
						cout << "golden: " << golden_score << ", computed: " << cls_n << endl;
					}
#endif
					if(cls_p < cls_n){
						label=i;
						cls_p = cls_n;
					}
				}
				label++;
				//cout << dec << "\tlabel: " << label << endl;

				// packing output and write memOutStrm
				out_buf = out_buf >> 16;
				out_buf(63,48) = label;
				if(x%4==3){
					out.write(out_buf);
					out_buf=0;
				}
				//static_cast<hls::stream<ap_uint<16>>&>(wa_out).write(label);
			}
		}
#ifdef ACTIVATION_LOG
		golden_score_file.close();
		computed_score_file.close();
#endif
}

// hwkim modified for average pooling
template <int AVE_IFM_CH, int AVE_IFM_DIM>
void average_pooling(
		stream<ap_uint<AVE_IFM_CH>> &in,
		stream<ap_uint<AVE_IFM_CH>> &out,
		int ave_thres)
{
	  unsigned int ave_sum[AVE_IFM_CH];
	  ap_uint<AVE_IFM_CH> ave_buf;
	  for(int ch=0; ch<AVE_IFM_CH; ch++){
	#pragma HLS UNROLL
		  ave_sum[ch] = 0;
	  }
	  for(int y=0; y<AVE_IFM_DIM; y++){
		for(int x=0; x<AVE_IFM_DIM; x++){
			ave_buf = in.read();
			for(int ch=0; ch<AVE_IFM_CH; ch++){
				ave_sum[ch] = ave_sum[ch] + ave_buf[ch];
			}
		}
	  }
	  ave_buf = 0;
	  for(int ch=0; ch<AVE_IFM_CH; ch++){
		  if(ave_sum[ch]>=ave_thres)
			  ave_buf[ch] = 1;
	  }
	  out.write(ave_buf);

}



// hwkim modified for padding & segmentation
template <int IFMDim, int IFMHeight, int InWidth,
unsigned int Top, unsigned int Bottom, unsigned int Left, unsigned int Right>
void insert_pad(stream<ap_uint<InWidth>> & in_stream,
		stream<ap_uint<InWidth>>& out_stream)
{
  for(unsigned int y=0; y<IFMHeight; y++){
	  for(unsigned int x=0; x<IFMDim; x++){
		  // hwkim modified for individual padding
		  //if(x==0 || y==0 || x==(IFMDim-1) || y==(IFMHeight-1)){
		  if(x<Left || y<Top || x>(IFMDim-1-Right) || y>(IFMHeight-1-Bottom)){

			  out_stream.write(0);
			  //inter1_pad.write((ap_uint<64>)0xFFFFFFFFFFFFFFFF);
		  }
		  else{
			  out_stream.write(in_stream.read());
		  }
//		  cout << setw(20) << hex << inter1_pad.read() << "|";
	  }
//	  cout << endl;
  }
//  cout << "padded stream size = " << out_stream.size() << endl;
}


// hwkim modified for debug
#ifdef ACTIVATION_LOG
template <unsigned int OFMDim,
		unsigned int OFMHeight,
		unsigned int InWidth,
		// hwkim added for stride
		unsigned int Stride>

void activation_log(
		stream<ap_uint<InWidth>>& in_stream,
		int layer_cnt){

	// hwkim commented for segmentation - there's only conv(tconv) layers
//	unsigned char layer_type[11] = {1,2,2,2,2,2,2,2,2,2,2};

	// file open
	string act_file_name;
	string golden_file_name;
	string compare_result_file_name;

	ofstream activation_log_file;
	FILE * golden_file;
	ofstream compare_result_file;

	// for last layer - there's no compare(SignX.txt) file
	int compare_skip = 0;

	act_file_name = snapshot_dir + "activation_" + to_string(layer_cnt+1) + "_log.txt";
	//act_file_name = "activation_" + to_string(layer_cnt+1) + "_log.txt";
	activation_log_file.open(act_file_name);
	if(!activation_log_file.is_open()){
		cout << act_file_name << " open error!!" << endl;
	}
	// hwkim commented for segmentation - there's only conv(tconv) layers
//	switch(layer_type[layer_cnt]){
//	case 1:
//	case 2:
//		golden_file_name = golden_file_dir + "Sign" + to_string(weighted_layer_cnt+1) + ".txt";
//		break;
//	case 4:
//		golden_file_name = golden_file_dir + "MaxPool" + to_string(pooling_layer_cnt+1) + ".txt";
//		break;
//	case 8:
//	case 16:
//		break;
//	}
	golden_file_name = golden_file_dir + "Sign" + to_string(layer_cnt+1) + ".txt";
	golden_file = fopen(golden_file_name.c_str(),"rt");
	if(golden_file==NULL){
		cout << golden_file_name << " open error!!" << endl;
		compare_skip = 1;
	}

	compare_result_file_name = "compare_result_" + to_string(layer_cnt+1) + ".txt";
	compare_result_file.open(compare_result_file_name);
	if(!compare_result_file.is_open()){
	  cout << compare_result_file_name << " open error!" << endl;
	}

	// activation logging & compare result logging
	ap_uint<InWidth> act_buf;
	ap_uint<InWidth> gold_buf;
	char gold_buf_ch[InWidth/4+1];
	char gold_buf_ch64[17];
	gold_buf_ch64[16] = 0;
	unsigned long gold_buf_long;
	//unsigned long long gold_buf;
//	cout << "inter" << (layer_cnt+1) << " stream size = " << in_stream.size() << endl;

	if(InWidth%8!=0){
		cout << "Activation log error! InWidth is not a multiple of 8" << endl;
	}

	// hwkim modified for stride
	for(int y=0; y<OFMHeight; y++){
		  for(int x=0; x<OFMDim; x++){
//	for(int y=0; y<OFMHeight/Stride; y++){
//		  for(int x=0; x<OFMDim/Stride; x++){

			  act_buf = in_stream.read();
			  if(compare_skip==0){

				  gold_buf = 0;
				  fscanf(golden_file, "%s", gold_buf_ch);
				  for(int word_cnt=0; word_cnt<InWidth/64; word_cnt++){
					  for(int i=0; i<64/4; i++){
						  gold_buf_ch64[i] = gold_buf_ch[word_cnt*16+i];
					  }
					  gold_buf_long = strtoul(gold_buf_ch64, NULL, 16);
					  gold_buf = gold_buf << 64;
					  gold_buf = gold_buf | (*reinterpret_cast<ap_uint<64> *>(&gold_buf_long));
				  }

			  // hwkim added for stride
//				  for(int stride_cnt=0; stride_cnt<(Stride-1); stride_cnt++){
//					  fscanf(golden_file, "%s", gold_buf_ch);	//skip one activation read
//				  }

				  if(act_buf!=gold_buf){
					  compare_result_file << dec << "@(" << setw(2) << y << "," << setw(2) << x << ")" <<
							  hex << " golden: ";
					  if(InWidth>=64){
						  for(int i=0; i<InWidth/64; i++){
							  compare_result_file << (unsigned long long )(gold_buf >> 64*(InWidth/64-i-1));
						  }
						  compare_result_file << "," << endl << setw(17) << hex << "act: ";
						  for(int i=0; i<InWidth/64; i++){
							  compare_result_file << (unsigned long long )(act_buf >> 64*(InWidth/64-i-1));
						  }
						  compare_result_file << endl;
					  }
					  else{
						  compare_result_file << (unsigned long long )gold_buf << "," << endl
								  << setw(17) << hex << "act: "  << (unsigned long long )act_buf << endl;
					  }
				  }
			  }

			  if(InWidth>64){
//				  for(int i=0; i<InWidth/64; i++){
//					  activation_log_file << uppercase << setfill('0') << setw(16) << hex << (unsigned long long )(act_buf >> 64*(InWidth/64-i-1));
//				  }
				  ap_uint<InWidth> log_mask = ((ap_uint<InWidth>)0xF<<(InWidth-4));
				  for(int i=0; i<InWidth/4; i++){
					  activation_log_file << uppercase << hex << (unsigned int)((act_buf&log_mask)>>(InWidth-4));
					  act_buf = act_buf << 4;
				  }
				  activation_log_file << endl;	//" | ";				  cout << endl;
			  }
			  else{
				  activation_log_file << uppercase << setfill('0') << setw(16) << hex << (unsigned long long )act_buf << endl;	//" |\t";
			  }
		  }
		  //activation_log_file << endl;
	}

	activation_log_file.close();
	//golden_file.close();
	if(compare_skip==0)
		fclose(golden_file);
	compare_result_file.close();

}


// hwkim added for separated simulation
template <unsigned int OutWidth>
void read_activation_file(
		//string snapshot_file_name,
		stream<ap_uint<OutWidth>> & out_stream,
		int layer_cnt
		)
{
	string snapshot_file_name = snapshot_dir + "activation_" + to_string(layer_cnt+1) + "_log.txt";
	FILE * act_snapshot_file = fopen(snapshot_file_name.c_str(), "rt");
	//if(!inter4_snapshot_file.is_open()){
	if(act_snapshot_file==NULL){
		cout << "act_snapshot_file open error!" << endl;
	}

	//unsigned long inter_snap_buf;
	ap_uint<OutWidth> act_snap_buf;
	unsigned long act_snap_long;
	char act_snap_ch_arr[OutWidth/4];
	unsigned char act_snap_int;
//	char act_snap_ch64[17];
	int stream_cnt=0;
	while(1){
		act_snap_buf = 0;
//		act_snap_ch64[16] = 0;
		fscanf(act_snapshot_file, "%s", act_snap_ch_arr);
		if(feof(act_snapshot_file))
			break;
//		for(int word_cnt=0; word_cnt<OutWidth/64; word_cnt++){
//			for(int i=0; i<64/4; i++){
//				act_snap_ch64[i] = act_snap_ch[word_cnt*16+i];
//			}
//			act_snap_long = strtoul(act_snap_ch64, NULL, 16);
//			act_snap_buf = act_snap_buf << 64;
//			act_snap_buf = act_snap_buf | (*reinterpret_cast<ap_uint<64> *>(&act_snap_long));
//		}
		for(int word_cnt=0; word_cnt<OutWidth/4; word_cnt++){
			if(act_snap_ch_arr[word_cnt]>0x40)
				act_snap_int = act_snap_ch_arr[word_cnt]-55;
			else
				act_snap_int = act_snap_ch_arr[word_cnt]-0x30;
			act_snap_buf = act_snap_buf << 4;
			act_snap_buf = act_snap_buf | (act_snap_int&0xF);
		}
		out_stream.write(act_snap_buf);
		stream_cnt++;
//		cout << dec << stream_cnt << " : " << hex << out_stream.read() << endl;
	}
//	cout << out_stream.size() << endl;
	fclose(act_snapshot_file);
}

#endif

unsigned int paddedSizeHW(unsigned int in, unsigned int padTo) {
  if(in % padTo == 0) {
    return in;
  } else {
    return in + padTo - (in % padTo);
  }
}

// hwkim modified for batch norm scale
//void DoMemInit(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd,
void DoMemInit(int targetLayer,
		unsigned int targetMem, unsigned int targetInd, unsigned int targetThresh, ap_uint<64> val) {
  switch (targetLayer) {
    case 0:
      weights0.m_weights[targetMem][targetInd] = val;
      break;
    case 1:
      threshs0.m_thresholds[targetMem][targetInd][targetThresh] = *reinterpret_cast<ap_fixed<64, 56> *>(&val);
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
      threshs6.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 14:
      weights7.m_weights[targetMem][targetInd] = val;
      break;
    case 15:
      threshs7.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 16: weights8.m_weights[targetMem][targetInd] = val; break;
    case 17: threshs8.m_thresholds[targetMem][targetInd][targetThresh] = val; break;
    case 18: weights9.m_weights[targetMem][targetInd] = val; break;
    case 19: threshs9.m_thresholds[targetMem][targetInd][targetThresh] = val; break;
    case 20: weights10.m_weights[targetMem][targetInd] = val; break;
    case 21: threshs10.m_thresholds[targetMem][targetInd][targetThresh] = val; break;
    case -1: threshs10.m_scales[targetMem] = *reinterpret_cast<ap_ufixed<8, 0, AP_TRN, AP_SAT> *>(&val); break;
  }
}

void DoCompute(ap_uint<64> *in, ap_uint<64>* out, const unsigned int numReps) {
#pragma HLS DATAFLOW
  stream<ap_uint<64>> inter0("DoCompute.inter0");
#pragma HLS STREAM variable=inter0 //depth=256
  stream<ap_uint<192>> inter0_1("DoCompute.inter0_1");
#pragma HLS STREAM variable=inter0_1 //depth=256
  stream<ap_uint<24>> inter0_2("DoCompute.inter0_2");
#pragma HLS STREAM variable=inter0_2 //depth=256
  stream<ap_uint<64>> inter1("DoCompute.inter1");
#pragma HLS STREAM variable=inter1 //depth=256
  stream<ap_uint<64>> inter2("DoCompute.inter2");
#pragma HLS STREAM variable=inter2 //depth=256
  stream<ap_uint<128>> inter3("DoCompute.inter3");
#pragma HLS STREAM variable=inter3 //depth=256
  stream<ap_uint<128>> inter4("DoCompute.inter4");
#pragma HLS STREAM variable=inter4 //depth=256
  stream<ap_uint<256>> inter5("DoCompute.inter5");
#pragma HLS STREAM variable=inter5 //depth=256
  stream<ap_uint<256>> inter6("DoCompute.inter6");
#pragma HLS STREAM variable=inter6 //depth=256
  stream<ap_uint<128>> inter7("DoCompute.inter7");
#pragma HLS STREAM variable=inter7 //depth=256
  stream<ap_uint<128>> inter8("DoCompute.inter8");
#pragma HLS STREAM variable=inter8 //depth=256
  stream<ap_uint<64>> inter9("DoCompute.inter9");
#pragma HLS STREAM variable=inter9 //depth=256
  stream<ap_uint<64>> inter10("DoCompute.inter10");
#pragma HLS STREAM variable=inter10 //depth=256
  // hwkim modified for batch norm scale
  //stream<ap_uint<11*16>> inter11("DoCompute.inter11");
  stream<ap_uint<11*24>> inter11("DoCompute.inter11");
#pragma HLS STREAM variable=inter11 //depth=256

  stream<ap_uint<64>> memOutStrm("DoCompute.memOutStrm");
#ifdef FPGA_DEBUG
#pragma HLS STREAM variable=memOutStrm //depth=512

  stream<ap_uint<64>> inter_log("DoCompute.inter_log");
#pragma HLS STREAM variable=inter_log

#endif

  // hwkim modified for padding & segmentation
  //const unsigned int inBits = 32 * 32 * 3 * 8;
  const unsigned int inBits = L0_IFM_DIM * L0_IFM_HEIGHT * 3 * 8;

  // commented by author
  // const unsigned int inBitsPadded = paddedSize(inBits, 64);

  // hwkim modified for segmentation
  //const unsigned int outBits = L8_MH*16;
  const unsigned int outBits = L10_OFM_DIM*L10_OFM_HEIGHT*16;

#ifdef SEP_SIM
	int sep_sim_layer1_en = 0;
	int sep_sim_layer2_en = 0;
	int sep_sim_layer3_en = 0;
	int sep_sim_layer4_en = 0;
	int sep_sim_layer5_en = 0;
	int sep_sim_layer6_en = 0;
	int sep_sim_layer7_en = 0;
	int sep_sim_layer8_en = 0;
	int sep_sim_layer9_en = 0;
	int sep_sim_layer10_en = 0;
	int sep_sim_layer11_en = 0;
#endif

	Mem2Stream_Batch<64, inBits / 8>(in, inter0, numReps);

	// hwkim modified for padding & segmentation
	//StreamingDataWidthConverter_Batch<64, 192, (32 * 32 * 3 * 8) / 64>(inter0, inter0_1, numReps);
	//StreamingDataWidthConverter_Batch<192, 24, (32 * 32 * 3 * 8) / 192>(inter0_1, inter0_2, numReps);
	// hwkim modified for padding
	//StreamingDataWidthConverter_Batch<64, 192, (L0_IFM_DIM*L0_IFM_HEIGHT*3*8)/64+1>(inter0, inter0_1, numReps);
	//StreamingDataWidthConverter_Batch<192, 24, (L0_IFM_DIM*L0_IFM_HEIGHT*3*8)/192+1>(inter0_1, inter0_2, numReps);
	StreamingDataWidthConverter_Batch<64, 192, (L0_IFM_DIM*L0_IFM_HEIGHT*3*8)/64>(inter0, inter0_1, numReps);
	StreamingDataWidthConverter_Batch<192, 24, (L0_IFM_DIM*L0_IFM_HEIGHT*3*8)/192>(inter0_1, inter0_2, numReps);

#ifdef SEP_SIM
	if(sep_sim_layer1_en)
#endif
		//////////////////////////////////////////////////////////////////
		// Layer 1 - fixed point input, binary weight
		//////////////////////////////////////////////////////////////////
		ConvLayer_Batch<L0_K, L0_IFM_CH, L0_IFM_DIM, L0_OFM_CH, L0_OFM_DIM,
			L0_IFM_HEIGHT, L0_OFM_HEIGHT, 1, 1, 1, 1, 1,
#ifdef ACTIVATION_LOG
			0,
#endif
			L0_SIMD, L0_PE,
			ap_uint<1>,	// hwkim added for batch norm scale
			Slice<ap_fixed<8, 1, AP_TRN, AP_SAT>>, Identity, Recast<Binary>>
				(inter0_2, inter1, weights0, threshs0, numReps, ap_resource_lut());
#ifdef SEP_SIM
	else
		read_activation_file<L0_OFM_CH>(inter1, 0);
#endif

#ifdef FPGA_DEBUG
	#if ACT_LOG_LAYER == 0
		StreamingDataWidthConverter_Batch<L0_OFM_CH, 64, (L0_IFM_DIM*L0_IFM_HEIGHT*L0_OFM_CH/64)>(inter1, inter_log, numReps);
		Stream2Mem_Batch<64, (L0_IFM_DIM*L0_IFM_HEIGHT*64)/8>(inter_log, out, numReps);
	#endif
#endif

#ifdef FPGA_DEBUG
	#if ACT_LOG_LAYER >= 1
		#ifdef SEP_SIM
			if(sep_sim_layer2_en)
		#endif
				//////////////////////////////////////////////////////////////////
				// Layer 2 - binary convolution
				//////////////////////////////////////////////////////////////////
				// hwkim modified for padding
//				stream<ap_uint<64>> inter1_pad("DoCompute.inter1_pad");
//				insert_pad<L1_IFM_DIM, L1_IFM_HEIGHT, 64, 1, 1, 1, 1>(inter1, inter1_pad);
				ConvLayer_Batch<L1_K, L1_IFM_CH, L1_IFM_DIM, L1_OFM_CH, L1_OFM_DIM,
					L1_IFM_HEIGHT, L1_OFM_HEIGHT, 1, 1, 1, 1, 1,
		#ifdef ACTIVATION_LOG
					1,
		#endif
					L1_SIMD, L1_PE,
					ap_uint<1>,	// hwkim added for batch norm scale
					Recast<XnorMul>> (inter1, inter2, weights1, threshs1, numReps, ap_resource_lut());
		#ifdef SEP_SIM
			else
				read_activation_file<L1_OFM_CH>(inter2, 1);
		#endif
		#endif
#else
	#ifdef SEP_SIM
		if(sep_sim_layer2_en)
	#endif
			//////////////////////////////////////////////////////////////////
			// Layer 2 - binary convolution
			//////////////////////////////////////////////////////////////////
			// hwkim modified for padding
//			stream<ap_uint<64>> inter1_pad("DoCompute.inter1_pad");
//			insert_pad<L1_IFM_DIM, L1_IFM_HEIGHT, 64, 1, 1, 1, 1>(inter1, inter1_pad);
			ConvLayer_Batch<L1_K, L1_IFM_CH, L1_IFM_DIM, L1_OFM_CH, L1_OFM_DIM,
				L1_IFM_HEIGHT, L1_OFM_HEIGHT, 1, 1, 1, 1, 1,
	#ifdef ACTIVATION_LOG
				1,
	#endif
				L1_SIMD, L1_PE,
				ap_uint<1>,	// hwkim added for batch norm scale
				Recast<XnorMul>> (inter1, inter2, weights1, threshs1, numReps, ap_resource_lut());
	#ifdef SEP_SIM
		else
			read_activation_file<L1_OFM_CH>(inter2, 1);
	#endif
#endif

#ifdef FPGA_DEBUG
	#if ACT_LOG_LAYER == 1
		StreamingDataWidthConverter_Batch<L1_OFM_CH, 64, (L1_IFM_DIM*L1_IFM_HEIGHT*L1_OFM_CH/64)>(inter2, inter_log, numReps);
		Stream2Mem_Batch<64, (L1_IFM_DIM*L1_IFM_HEIGHT*64)/8>(inter_log, out, numReps);
	#endif
#endif

#ifdef FPGA_DEBUG
	#if ACT_LOG_LAYER >= 2
		// Layer 3 - binary convolution - stride 2, channel x2
		#ifdef SEP_SIM
			if(sep_sim_layer3_en)
		#endif
				// hwkim modified for padding
		//			stream<ap_uint<64>> inter2_pad("DoCompute.inter2_pad");
		//			insert_pad<L2_IFM_DIM, L2_IFM_HEIGHT, 64, 0, 1, 0, 1>(inter2, inter2_pad);
				ConvLayer_Batch<L2_K, L2_IFM_CH, L2_IFM_DIM, L2_OFM_CH, L2_OFM_DIM,
					L2_IFM_HEIGHT, L2_OFM_HEIGHT, 2, 0, 1, 0, 1,
		#ifdef ACTIVATION_LOG
					2,
		#endif
					L2_SIMD, L2_PE,
					ap_uint<1>,	// hwkim added for batch norm scale
					Recast<XnorMul>>
						(inter2, inter3, weights2, threshs2, numReps, ap_resource_lut());
		#ifdef SEP_SIM
			else
				read_activation_file<L2_OFM_CH>(inter3, 2);
		#endif
	#endif
#else
	// Layer 3 - binary convolution - stride 2, channel x2
	#ifdef SEP_SIM
		if(sep_sim_layer3_en)
	#endif
			// hwkim modified for padding
//			stream<ap_uint<64>> inter2_pad("DoCompute.inter2_pad");
//			insert_pad<L2_IFM_DIM, L2_IFM_HEIGHT, 64, 0, 1, 0, 1>(inter2, inter2_pad);
			ConvLayer_Batch<L2_K, L2_IFM_CH, L2_IFM_DIM, L2_OFM_CH, L2_OFM_DIM,
				L2_IFM_HEIGHT, L2_OFM_HEIGHT, 2, 0, 1, 0, 1,
	#ifdef ACTIVATION_LOG
				2,
	#endif
				L2_SIMD, L2_PE,
				ap_uint<1>,	// hwkim added for batch norm scale
				Recast<XnorMul>>
					(inter2, inter3, weights2, threshs2, numReps, ap_resource_lut());
	#ifdef SEP_SIM
		else
			read_activation_file<L2_OFM_CH>(inter3, 2);
	#endif
#endif

#ifdef FPGA_DEBUG
	#if ACT_LOG_LAYER == 2
		StreamingDataWidthConverter_Batch<L2_OFM_CH, 64, (L2_IFM_DIM*L2_IFM_HEIGHT*L2_OFM_CH/64)>(inter3, inter_log, numReps);
		Stream2Mem_Batch<64, (L2_IFM_DIM*L2_IFM_HEIGHT*64)/8>(inter_log, out, numReps);
	#endif
#endif

#ifdef FPGA_DEBUG
	#if ACT_LOG_LAYER >= 3
		// Layer 4 - binary convolution
		#ifdef SEP_SIM
			if(sep_sim_layer4_en)
		#endif
				// hwkim modified for padding
//				stream<ap_uint<128>> inter3_pad("DoCompute.inter3_pad");
//				insert_pad<L3_IFM_DIM, L3_IFM_HEIGHT, 128, 1, 1, 1, 1>(inter3, inter3_pad);
				ConvLayer_Batch<L3_K, L3_IFM_CH, L3_IFM_DIM, L3_OFM_CH, L3_OFM_DIM, L3_IFM_HEIGHT, L3_OFM_HEIGHT,
					1, 1, 1, 1, 1,
		#ifdef ACTIVATION_LOG
					3,
		#endif
					L3_SIMD, L3_PE,
					ap_uint<1>,	// hwkim added for batch norm scale
					Recast<XnorMul>>
						(inter3, inter4, weights3, threshs3, numReps, ap_resource_lut());
		#ifdef SEP_SIM
				else
					read_activation_file<L3_OFM_CH>(inter4, 3);
		#endif
	#endif
#else
		// Layer 4 - binary convolution
	#ifdef SEP_SIM
		if(sep_sim_layer4_en)
	#endif
			// hwkim modified for padding
//			stream<ap_uint<128>> inter3_pad("DoCompute.inter3_pad");
//			insert_pad<L3_IFM_DIM, L3_IFM_HEIGHT, 128, 1, 1, 1, 1>(inter3, inter3_pad);
			ConvLayer_Batch<L3_K, L3_IFM_CH, L3_IFM_DIM, L3_OFM_CH, L3_OFM_DIM, L3_IFM_HEIGHT, L3_OFM_HEIGHT,
				1, 1, 1, 1, 1,
	#ifdef ACTIVATION_LOG
				3,
	#endif
				L3_SIMD, L3_PE,
				ap_uint<1>,	// hwkim added for batch norm scale
				Recast<XnorMul>>
					(inter3, inter4, weights3, threshs3, numReps, ap_resource_lut());
	#ifdef SEP_SIM
			else
				read_activation_file<L3_OFM_CH>(inter4, 3);
	#endif
#endif

#ifdef FPGA_DEBUG
	#if ACT_LOG_LAYER == 3
		StreamingDataWidthConverter_Batch<L3_OFM_CH, 64, (L3_IFM_DIM*L3_IFM_HEIGHT*L3_OFM_CH/64)>(inter4, inter_log, numReps);
		Stream2Mem_Batch<64, (L3_IFM_DIM*L3_IFM_HEIGHT*64)/8>(inter_log, out, numReps);
	#endif
#endif

#ifdef FPGA_DEBUG
	#if ACT_LOG_LAYER >= 4
			// Layer 5 - binary convolution - stride 2, channel x2
		#ifdef SEP_SIM
			if(sep_sim_layer5_en)
		#endif
				// hwkim modified for padding
//				stream<ap_uint<128>> inter4_pad("DoCompute.inter4_pad");
//				insert_pad<L4_IFM_DIM, L4_IFM_HEIGHT, 128, 0, 1, 0, 1>(inter4, inter4_pad);
				ConvLayer_Batch<L4_K, L4_IFM_CH, L4_IFM_DIM, L4_OFM_CH, L4_OFM_DIM,
					L4_IFM_HEIGHT, L4_OFM_HEIGHT, 2, 0, 1, 0, 1,
		#ifdef ACTIVATION_LOG
					4,
		#endif
					L4_SIMD, L4_PE,
					ap_uint<1>,	// hwkim added for batch norm scale
					Recast<XnorMul>>
						(inter4, inter5, weights4, threshs4, numReps, ap_resource_lut());
		#ifdef SEP_SIM
				else
					read_activation_file<L4_OFM_CH>(inter5, 4);
		#endif
	#endif
#else
		// Layer 5 - binary convolution - stride 2, channel x2
	#ifdef SEP_SIM
		if(sep_sim_layer5_en)
	#endif
			// hwkim modified for padding
//			stream<ap_uint<128>> inter4_pad("DoCompute.inter4_pad");
//			insert_pad<L4_IFM_DIM, L4_IFM_HEIGHT, 128, 0, 1, 0, 1>(inter4, inter4_pad);
			ConvLayer_Batch<L4_K, L4_IFM_CH, L4_IFM_DIM, L4_OFM_CH, L4_OFM_DIM,
				L4_IFM_HEIGHT, L4_OFM_HEIGHT, 2, 0, 1, 0, 1,
	#ifdef ACTIVATION_LOG
				4,
	#endif
				L4_SIMD, L4_PE,
				ap_uint<1>,	// hwkim added for batch norm scale
				Recast<XnorMul>>
					(inter4, inter5, weights4, threshs4, numReps, ap_resource_lut());
	#ifdef SEP_SIM
			else
				read_activation_file<L4_OFM_CH>(inter5, 4);
	#endif
#endif

#ifdef FPGA_DEBUG
	#if ACT_LOG_LAYER == 4
		StreamingDataWidthConverter_Batch<L4_OFM_CH, 64, (L4_IFM_DIM*L4_IFM_HEIGHT*L4_OFM_CH/64)>(inter5, inter_log, numReps);
		Stream2Mem_Batch<64, (L4_IFM_DIM*L4_IFM_HEIGHT*64)/8>(inter_log, out, numReps);
	#endif
#endif

#ifdef FPGA_DEBUG
	#if ACT_LOG_LAYER >= 5
			// Layer 6 - binary convolution
		#ifdef SEP_SIM
			if(sep_sim_layer6_en)
		#endif
				// hwkim modified for padding
//				stream<ap_uint<256>> inter5_pad("DoCompute.inter5_pad");
//				insert_pad<L5_IFM_DIM, L5_IFM_HEIGHT, 256, 1, 1, 1, 1>(inter5, inter5_pad);
				ConvLayer_Batch<L5_K, L5_IFM_CH, L5_IFM_DIM, L5_OFM_CH, L5_OFM_DIM,
					L5_IFM_HEIGHT, L5_OFM_HEIGHT, 1, 1, 1, 1, 1,
		#ifdef ACTIVATION_LOG
					5,
		#endif
					L5_SIMD, L5_PE,
					ap_uint<1>,	// hwkim added for batch norm scale
					Recast<XnorMul>>
						(inter5, inter6, weights5, threshs5, numReps, ap_resource_lut());
		#ifdef SEP_SIM
				else
					read_activation_file<L5_OFM_CH>(inter6, 5);
		#endif
	#endif
#else
		// Layer 6 - binary convolution
	#ifdef SEP_SIM
		if(sep_sim_layer6_en)
	#endif
			// hwkim modified for padding
	//		stream<ap_uint<256>> inter5_pad("DoCompute.inter5_pad");
	//		insert_pad<L5_IFM_DIM, L5_IFM_HEIGHT, 256, 1, 1, 1, 1>(inter5, inter5_pad);
			ConvLayer_Batch<L5_K, L5_IFM_CH, L5_IFM_DIM, L5_OFM_CH, L5_OFM_DIM,
				L5_IFM_HEIGHT, L5_OFM_HEIGHT, 1, 1, 1, 1, 1,
	#ifdef ACTIVATION_LOG
				5,
	#endif
				L5_SIMD, L5_PE,
				ap_uint<1>,	// hwkim added for batch norm scale
				Recast<XnorMul>>
					(inter5, inter6, weights5, threshs5, numReps, ap_resource_lut());
	#ifdef SEP_SIM
			else
				read_activation_file<L5_OFM_CH>(inter6, 5);
	#endif
#endif

#ifdef FPGA_DEBUG
	#if ACT_LOG_LAYER == 5
		StreamingDataWidthConverter_Batch<L5_OFM_CH, 64, (L5_IFM_DIM*L5_IFM_HEIGHT*L5_OFM_CH/64)>(inter6, inter_log, numReps);
		Stream2Mem_Batch<64, (L5_IFM_DIM*L5_IFM_HEIGHT*64)/8>(inter_log, out, numReps);
	#endif
#endif

#ifdef FPGA_DEBUG
	#if ACT_LOG_LAYER >= 6
			// Layer 7 - binary transposed convolution - stride 2
		#ifdef SEP_SIM
			if(sep_sim_layer7_en)
		#endif
//				stream<ap_uint<256>> inter6_pad("DoCompute.inter6_pad");
//				insert_pad<L6_IFM_DIM, L6_IFM_HEIGHT, 256, 1, 1, 1, 1>(inter6, inter6_pad);
				UpConvLayer_Batch<L6_K, L6_IFM_CH, L6_IFM_DIM, L6_OFM_CH, L6_OFM_DIM,
					L6_IFM_HEIGHT, L6_OFM_HEIGHT, 1, 0, 1, 0, 1,
		#ifdef ACTIVATION_LOG
					6,
		#endif
					L6_SIMD, L6_PE, Recast<XnorMul>>
						(inter6, inter7, weights6, threshs6, numReps, ap_resource_lut());
		#ifdef SEP_SIM
				else
					read_activation_file<L6_OFM_CH>(inter7, 6);
		#endif
	#endif
#else
		// Layer 7 - binary transposed convolution - stride 2
	#ifdef SEP_SIM
		if(sep_sim_layer7_en)
	#endif
//			stream<ap_uint<256>> inter6_pad("DoCompute.inter6_pad");
//			insert_pad<L6_IFM_DIM, L6_IFM_HEIGHT, 256, 1, 1, 1, 1>(inter6, inter6_pad);
			UpConvLayer_Batch<L6_K, L6_IFM_CH, L6_IFM_DIM, L6_OFM_CH, L6_OFM_DIM,
				L6_IFM_HEIGHT, L6_OFM_HEIGHT, 1, 0, 1, 0, 1,
	#ifdef ACTIVATION_LOG
				6,
	#endif
				L6_SIMD, L6_PE, Recast<XnorMul>>
					(inter6, inter7, weights6, threshs6, numReps, ap_resource_lut());
	#ifdef SEP_SIM
			else
				read_activation_file<L6_OFM_CH>(inter7, 6);
	#endif
#endif

#ifdef FPGA_DEBUG
	#if ACT_LOG_LAYER == 6
		StreamingDataWidthConverter_Batch<L6_OFM_CH, 64, (L6_IFM_DIM*L6_IFM_HEIGHT*L6_OFM_CH/64)>(inter7, inter_log, numReps);
		Stream2Mem_Batch<64, (L6_IFM_DIM*L6_IFM_HEIGHT*64)/8>(inter_log, out, numReps);
	#endif
#endif

#ifdef FPGA_DEBUG
	#if ACT_LOG_LAYER >= 7
			// Layer 8 - binary transposed convolution - stride 1
		#ifdef SEP_SIM
			if(sep_sim_layer8_en)
		#endif
//				stream<ap_uint<128>> inter7_pad("DoCompute.inter3_pad");
//				insert_pad<L7_IFM_DIM, L7_IFM_HEIGHT, 128, 1, 1, 1, 1>(inter7, inter7_pad);
				ConvLayer_Batch<L7_K, L7_IFM_CH, L7_IFM_DIM, L7_OFM_CH, L7_OFM_DIM,
					L7_IFM_HEIGHT, L7_OFM_HEIGHT, 1, 1, 1, 1, 1,
		#ifdef ACTIVATION_LOG
					7,
		#endif
					L7_SIMD, L7_PE,
					ap_uint<1>,	// hwkim added for batch norm scale
					Recast<XnorMul>>
						(inter7, inter8, weights7, threshs7, numReps, ap_resource_lut());
		#ifdef SEP_SIM
				else
					read_activation_file<L7_OFM_CH>(inter8, 7);
		#endif
	#endif
#else
		// Layer 8 - binary transposed convolution - stride 1
	#ifdef SEP_SIM
		if(sep_sim_layer8_en)
	#endif
//			stream<ap_uint<128>> inter7_pad("DoCompute.inter3_pad");
//			insert_pad<L7_IFM_DIM, L7_IFM_HEIGHT, 128, 1, 1, 1, 1>(inter7, inter7_pad);
			ConvLayer_Batch<L7_K, L7_IFM_CH, L7_IFM_DIM, L7_OFM_CH, L7_OFM_DIM,
				L7_IFM_HEIGHT, L7_OFM_HEIGHT, 1, 1, 1, 1, 1,
	#ifdef ACTIVATION_LOG
				7,
	#endif
				L7_SIMD, L7_PE,
				ap_uint<1>,	// hwkim added for batch norm scale
				Recast<XnorMul>>
					(inter7, inter8, weights7, threshs7, numReps, ap_resource_lut());
	#ifdef SEP_SIM
			else
				read_activation_file<L7_OFM_CH>(inter8, 7);
	#endif
#endif

#ifdef FPGA_DEBUG
	#if ACT_LOG_LAYER == 7
		StreamingDataWidthConverter_Batch<L7_OFM_CH, 64, (L7_IFM_DIM*L7_IFM_HEIGHT*L7_OFM_CH/64)>(inter8, inter_log, numReps);
		Stream2Mem_Batch<64, (L7_IFM_DIM*L7_IFM_HEIGHT*64)/8>(inter_log, out, numReps);
	#endif
#endif

#ifdef FPGA_DEBUG
	#if ACT_LOG_LAYER >= 8
			// Layer 9 - binary transposed convolution - stride 2
		#ifdef SEP_SIM
				if(sep_sim_layer9_en)
		#endif
//					stream<ap_uint<128>> inter8_pad("DoCompute.inter8_pad");
//					insert_pad<L8_IFM_DIM, L8_IFM_HEIGHT, 128, 0, 1, 0, 1>(inter8, inter8_pad);
					UpConvLayer_Batch<L8_K, L8_IFM_CH, L8_IFM_DIM, L8_OFM_CH, L8_OFM_DIM,
						L8_IFM_HEIGHT, L8_OFM_HEIGHT, 1, 0, 1, 0, 1,
		#ifdef ACTIVATION_LOG
						8,
		#endif
						L8_SIMD, L8_PE, Recast<XnorMul>>
							(inter8, inter9, weights8, threshs8, numReps, ap_resource_lut());
		#ifdef SEP_SIM
				else
					read_activation_file<L8_OFM_CH>(inter9, 8);
		#endif
	#endif
#else
		// Layer 9 - binary transposed convolution - stride 2
	#ifdef SEP_SIM
		if(sep_sim_layer9_en)
	#endif
//			stream<ap_uint<128>> inter8_pad("DoCompute.inter8_pad");
//			insert_pad<L8_IFM_DIM, L8_IFM_HEIGHT, 128, 0, 1, 0, 1>(inter8, inter8_pad);
			UpConvLayer_Batch<L8_K, L8_IFM_CH, L8_IFM_DIM, L8_OFM_CH, L8_OFM_DIM,
				L8_IFM_HEIGHT, L8_OFM_HEIGHT, 1, 0, 1, 0, 1,
	#ifdef ACTIVATION_LOG
				8,
	#endif
				L8_SIMD, L8_PE, Recast<XnorMul>>
					(inter8, inter9, weights8, threshs8, numReps, ap_resource_lut());
	#ifdef SEP_SIM
		else
			read_activation_file<L8_OFM_CH>(inter9, 8);
	#endif
#endif

#ifdef FPGA_DEBUG
	#if ACT_LOG_LAYER == 8
		StreamingDataWidthConverter_Batch<L8_OFM_CH, 64, (L8_IFM_DIM*L8_IFM_HEIGHT*L8_OFM_CH/64)>(inter9, inter_log, numReps);
		Stream2Mem_Batch<64, (L8_IFM_DIM*L8_IFM_HEIGHT*64)/8>(inter_log, out, numReps);
	#endif
#endif

#ifdef FPGA_DEBUG
	#if ACT_LOG_LAYER >= 9
			// Layer 10 - binary convolution
		#ifdef SEP_SIM
			if(sep_sim_layer10_en)
		#endif
				ConvLayer_Batch<L9_K, L9_IFM_CH, L9_IFM_DIM, L9_OFM_CH, L9_OFM_DIM,
					L9_IFM_HEIGHT, L9_OFM_HEIGHT, 1, 1, 1, 1, 1,
		#ifdef ACTIVATION_LOG
					9,
		#endif
					L9_SIMD, L9_PE,
					ap_uint<1>,	// hwkim added for batch norm scale
					Recast<XnorMul>>
						(inter9, inter10, weights9, threshs9, numReps, ap_resource_lut());
		#ifdef SEP_SIM
			else
				read_activation_file<L9_OFM_CH>(inter10, 9);
		#endif
	#endif
#else
		// Layer 10 - binary convolution
	#ifdef SEP_SIM
		if(sep_sim_layer10_en)
	#endif
			ConvLayer_Batch<L9_K, L9_IFM_CH, L9_IFM_DIM, L9_OFM_CH, L9_OFM_DIM,
				L9_IFM_HEIGHT, L9_OFM_HEIGHT, 1, 1, 1, 1, 1,
	#ifdef ACTIVATION_LOG
				9,
	#endif
				L9_SIMD, L9_PE,
				ap_uint<1>,	// hwkim added for batch norm scale
				Recast<XnorMul>>
					(inter9, inter10, weights9, threshs9, numReps, ap_resource_lut());
	#ifdef SEP_SIM
		else
			read_activation_file<L9_OFM_CH>(inter10, 9);
	#endif
#endif

#ifdef FPGA_DEBUG
	#if ACT_LOG_LAYER == 9
		StreamingDataWidthConverter_Batch<L9_OFM_CH, 64, (L9_IFM_DIM*L9_IFM_HEIGHT*L9_OFM_CH/64)>(inter10, inter_log, numReps);
		Stream2Mem_Batch<64, (L9_IFM_DIM*L9_IFM_HEIGHT*64)/8>(inter_log, out, numReps);
	#endif
#endif

#ifdef FPGA_DEBUG
	#if ACT_LOG_LAYER >= 10
			// Layer 11 - binary convolution
		#ifdef SEP_SIM
			if(sep_sim_layer11_en)
		#endif
				ConvLayer_Batch<L10_K, L10_IFM_CH, L10_IFM_DIM, L10_OFM_CH, L10_OFM_DIM,
					L10_IFM_HEIGHT, L10_OFM_HEIGHT, 1, 1, 1, 1, 1,
		#ifdef ACTIVATION_LOG
					10,
		#endif
					L10_SIMD, L10_PE,
					ap_fixed<24,16,AP_TRN,AP_SAT>,	// hwkim added for batch norm scale
					Recast<XnorMul>,
					Slice<ap_fixed<24,16,AP_TRN,AP_SAT> >>	//Slice<ap_int<16> >>	// hwkim modified for batch norm scale
					(inter10, inter11, weights10, threshs10, numReps, ap_resource_lut());
		#ifdef SEP_SIM
			else
				// hwkim modified for batch norm scale
				//read_activation_file<L10_OFM_CH*16>(inter11, 10);
				read_activation_file<L10_OFM_CH*24>(inter11, 10);
		#endif
	#endif
#else
		// Layer 11 - binary convolution
	#ifdef SEP_SIM
		if(sep_sim_layer11_en)
	#endif
			ConvLayer_Batch<L10_K, L10_IFM_CH, L10_IFM_DIM, L10_OFM_CH, L10_OFM_DIM,
				L10_IFM_HEIGHT, L10_OFM_HEIGHT, 1, 1, 1, 1, 1,
	#ifdef ACTIVATION_LOG
				10,
	#endif
				L10_SIMD, L10_PE,
				ap_fixed<24,16,AP_TRN,AP_SAT>,	// hwkim added for batch norm scale
				Recast<XnorMul>,
				Slice<ap_fixed<24,16,AP_TRN,AP_SAT> >>	//Slice<ap_int<16> >>	// hwkim modified for batch norm scale
				(inter10, inter11, weights10, threshs10, numReps, ap_resource_lut());
	#ifdef SEP_SIM
		else
			// hwkim modified for batch norm scale
			//read_activation_file<L10_OFM_CH*16>(inter11, 10);
			read_activation_file<L10_OFM_CH*24>(inter11, 10);
	#endif
#endif

#ifdef FPGA_DEBUG
	#if ACT_LOG_LAYER == 10
		StreamingDataWidthConverter_Batch<L10_OFM_CH*24, 64, (L10_IFM_DIM*L10_IFM_HEIGHT)>(inter11, inter_log, numReps);
		Stream2Mem_Batch<64, (L10_IFM_DIM*L10_IFM_HEIGHT*(L10_OFM_CH*24+64-(L10_OFM_CH*24%64))/64)*8>(inter_log, out, numReps);
	#endif
#endif

  // hwkim modified for average pool
//#define AVE_IFM_CH 256
//#define AVE_OFM_CH 256
//#define AVE_IFM_DIM 4
//#define AVE_THRES (4*4/2)
//  average_pooling<AVE_IFM_CH, AVE_IFM_DIM>(inter9, inter10, AVE_THRES);

//	{
//		WidthAdjustedOutputStream<16, 64, L10_OFM_DIM*L10_OFM_HEIGHT>  wa_out(memOutStrm, numReps);
//	}	// region for calling destructor of wa_out

#ifdef FPGA_DEBUG
	#if ACT_LOG_LAYER == 11
		infer_category<(11*24), 64>(inter11, memOutStrm);
		Stream2Mem_Batch<64, outBits/8>(memOutStrm, out, numReps);
	#endif
#else
	infer_category<(11*24), 64>(inter11, memOutStrm);
	Stream2Mem_Batch<64, outBits/8>(memOutStrm, out, numReps);
#endif

//#ifdef FPGA_DEBUG
//Streaming:
//	switch(targetLayer){
//	case 0:
//		if(targetLayer==0){
//			StreamingDataWidthConverter_Batch<64, 64, (L0_IFM_DIM*L0_IFM_HEIGHT)>(inter1, inter1_64, numReps);
//			Stream2Mem_Batch<64, (L0_IFM_DIM*L0_IFM_HEIGHT*64)/8>(inter1_64, out, numReps);
//			break;
//	case 1:
//			StreamingDataWidthConverter_Batch<64, 64, (L1_IFM_DIM*L1_IFM_HEIGHT)>(inter2, inter2_64, numReps);
//			Stream2Mem_Batch<64, (L1_IFM_DIM*L1_IFM_HEIGHT*64)/8>(inter2_64, out, numReps);
//			break;
//	case 2:
//		}
//		else{
//			Stream2Mem_Batch<64, outBits/8>(memOutStrm, out, numReps);
//		}
//			break;
//	}
//#endif

}

void BlackBoxJam(ap_uint<64> *in, ap_uint<64> *out, bool doInit,
		// hwkim modified for batch norm scale
		int targetLayer,	//unsigned int targetLayer,

		unsigned int targetMem,
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
	// hwkim modified for cosim
//#pragma HLS INTERFACE m_axi offset=slave port=in bundle=hostmem depth=512
#pragma HLS INTERFACE m_axi depth=1024 port=in offset=slave bundle=hostmem

#pragma HLS INTERFACE s_axilite port=in bundle=control
	// hwkim modified for cosim
//#pragma HLS INTERFACE m_axi offset=slave port=out bundle=hostmem depth=16
#pragma HLS INTERFACE m_axi depth=1024 port=out offset=slave bundle=hostmem

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
#pragma HLS ARRAY_PARTITION variable=threshs8.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs8.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights9.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs9.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs9.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights10.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs10.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs10.m_thresholds complete dim=3
	// hwkim added for batch norm scale
#pragma HLS ARRAY_PARTITION variable=threshs10.m_scales complete dim=1

  if (doInit) {
    DoMemInit(targetLayer, targetMem, targetInd, targetThresh, val);
  } else {
    //DoCompute(in, out, numReps);
	  DoCompute(in, out, 1);
  }
}
