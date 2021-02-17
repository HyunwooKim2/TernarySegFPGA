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
		unsigned char WAY,	// hwkim added for ternary
		unsigned char FanInCntWidth,	// hwkim added for ternary, log2(fan-in/WAY)
		typename TDstElem,	// hwkim added for batch norm scale
		typename TSrcI = Identity,      // redefine I/O interpretation as needed for input activations
		typename TDstI = Identity,		// redefine I/O interpretation as needed for output activations
		typename TWeightI = Identity,	// redefine I/O interpretation as needed for weigths
		int InStreamW, int OutStreamW,  // safely deducible (stream width must be int though!)
		int InMaskStreamW, int OutMaskStreamW,	// hwkim added for ternary
		typename TW,
		typename TM,	// hwkim added for ternary
		typename TA,
		typename R>
void ConvLayer_Batch(
				hls::stream<ap_uint<InStreamW>> &in,
				hls::stream<ap_uint<InMaskStreamW>> &in_mask,	// hwkim added for ternary
				hls::stream<ap_uint<OutStreamW>> &out,
				hls::stream<ap_uint<OutMaskStreamW>> &out_mask,
				TW const &weights,
				TM const &wmasks,	// hwkim added for ternary
				TA const &activation,
#ifdef SEP_SIM
				unsigned nonzero_en,
#endif
				unsigned const   reps,
				R const &r
) {
#pragma HLS INLINE

  unsigned const MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
  unsigned const MatrixH = OFMChannels;
  unsigned const InpPerImage = (float)(IFMDim*IFMHeight*IFMChannels)/InStreamW * TSrcI::width;

  WidthAdjustedInputStream <InStreamW, SIMD*TSrcI::width, InpPerImage>  wa_in (in,  reps);
  hls::stream<ap_uint<SIMD*TSrcI::width> > convInp("StreamingConvLayer_Batch.convInp");
  // ** hwkim modified for PE interleaving
//  WidthAdjustedOutputStream <PE*TDstI::width, OutStreamW, OFMDim * OFMHeight * (OFMChannels / PE)> mvOut (out,  reps);
  WidthAdjustedOutputStream <2*PE*TDstI::width, OutStreamW, OFMDim * OFMHeight * (OFMChannels / PE) / 2> mvOut (out,  reps);

  // hwkim added for ternary
  WidthAdjustedInputStream <InMaskStreamW, SIMD, InpPerImage>  wa_in_mask (in_mask,  reps);
  hls::stream<ap_uint<SIMD>> convInp_mask("StreamingConvLayer_Batch.convInp_mask");
  // ** hwkim modified for PE interleaving
//  WidthAdjustedOutputStream <PE, OutMaskStreamW, OFMDim * OFMHeight * (OFMChannels / PE)> mvOutMask (out_mask,  reps);
  WidthAdjustedOutputStream <2*PE, OutMaskStreamW, OFMDim * OFMHeight * (OFMChannels / PE) / 2> mvOutMask (out_mask,  reps);


  ConvolutionInputGenerator<ConvKernelDim, IFMChannels, TSrcI::width, IFMDim, OFMDim,
	  IFMHeight, OFMHeight, Top, Bottom, Left, Right,	// hwkim added for segmentation
  	  SIMD,
	  Stride>
  	  	  (wa_in,
		  wa_in_mask,	// hwkim added for ternary
		  convInp,
		  convInp_mask,	// hwkim added for ternary
		  reps);

  // hwkim added for ternary
  // ** hwkim modified for PE interleaving
//	hls::stream<ap_uint<WAY*TSrcI::width>> packed_input[PE*(SIMD/WAY)];
//	hls::stream<ap_uint<WAY>> packed_weight[PE*(SIMD/WAY)];
//	hls::stream<ap_uint<FanInCntWidth>> sf_num[PE*(SIMD/WAY)];
//	hls::stream<ap_uint<WAY>> packed_mask[PE*(SIMD/WAY)];
	hls::stream<ap_uint<WAY*TSrcI::width>> packed_input[2*PE*(SIMD/WAY)];
	hls::stream<ap_uint<WAY>> packed_weight[2*PE*(SIMD/WAY)];
	hls::stream<ap_uint<FanInCntWidth>> sf_num[2*PE*(SIMD/WAY)];
	hls::stream<ap_uint<WAY>> packed_mask[2*PE*(SIMD/WAY)];

//#pragma HLS ARRAY_PARTITION variable=packed_input complete dim=1	// ** hwkim added for OPTIMIZATION
#pragma HLS STREAM variable=packed_input
#pragma HLS STREAM variable=packed_weight
#pragma HLS STREAM variable=sf_num
#pragma HLS STREAM variable=packed_mask

#ifdef SEP_SIM
	if(nonzero_en==1)
#endif
		nonzero_activation_weight_stream_gen
		// ** hwkim modified for PE interleaving
//		<MatrixW, MatrixH, SIMD, PE, OFMDim, OFMHeight, Top, Bottom, Left, Right, TSrcI::width, WAY
		<MatrixW, MatrixH, SIMD, 2*PE, OFMDim, OFMHeight, Top, Bottom, Left, Right, TSrcI::width, WAY

#ifdef ACTIVATION_LOG
			, LayerCnt
#endif
			>
			(static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
					convInp_mask,
					weights,
					wmasks,
					static_cast<hls::stream<ap_uint<WAY*TSrcI::width>>*>(packed_input),
					static_cast<hls::stream<ap_uint<WAY>>*>(packed_weight),
					static_cast<hls::stream<ap_uint<FanInCntWidth>>*>(sf_num),
					static_cast<hls::stream<ap_uint<WAY>>*>(packed_mask),
					reps * OFMDim * OFMHeight);
#ifdef SEP_SIM
	else{
		extern string snapshot_dir;
		// ** hwkim modified for PE interleaving
//		ifstream nonz_i_log_file[PE][SIMD/WAY];
//		ifstream nonz_w_log_file[PE][SIMD/WAY];
//		ifstream nonz_m_log_file[PE][SIMD/WAY];
//		ifstream nonz_f_log_file[PE][SIMD/WAY];
		ifstream nonz_i_log_file[2*PE][SIMD/WAY];
		ifstream nonz_w_log_file[2*PE][SIMD/WAY];
		ifstream nonz_m_log_file[2*PE][SIMD/WAY];
		ifstream nonz_f_log_file[2*PE][SIMD/WAY];

		ap_uint<WAY*TSrcI::width> nonz_i_buf;
		ap_uint<WAY> nonz_w_buf;
		ap_uint<WAY> nonz_m_buf;
		ap_uint<FanInCntWidth> nonz_f_buf;

		while(!convInp.empty())
			convInp.read();
		while(!convInp_mask.empty())
			convInp_mask.read();

		// ** hwkim modified for PE interleaving
//		for(unsigned char pe=0; pe<PE; pe++){
		for(unsigned char pe=0; pe<2*PE; pe++){

			for(unsigned char way_cnt=0; way_cnt<(SIMD/WAY); way_cnt++){
				string nonz_i_log_file_name = snapshot_dir + "nonzero_" + to_string(LayerCnt+1)
						+ "_" + to_string(pe) + "_" + to_string(way_cnt) + "_input_log.txt";
				string nonz_w_log_file_name = snapshot_dir + "nonzero_" + to_string(LayerCnt+1)
						+ "_" + to_string(pe) + "_" + to_string(way_cnt) + "_weight_log.txt";
				string nonz_m_log_file_name = snapshot_dir + "nonzero_" + to_string(LayerCnt+1)
						+ "_" + to_string(pe) + "_" + to_string(way_cnt) + "_mask_log.txt";
				string nonz_f_log_file_name = snapshot_dir + "nonzero_" + to_string(LayerCnt+1)
						+ "_" + to_string(pe) + "_" + to_string(way_cnt) + "_fanin_log.txt";

				nonz_i_log_file[pe][way_cnt].open(nonz_i_log_file_name);
				nonz_w_log_file[pe][way_cnt].open(nonz_w_log_file_name);
				nonz_m_log_file[pe][way_cnt].open(nonz_m_log_file_name);
				nonz_f_log_file[pe][way_cnt].open(nonz_f_log_file_name);

				if(!nonz_i_log_file[pe][way_cnt].is_open())	cout << nonz_i_log_file_name << " open error!" << endl;
				else	cout << "Reading " << nonz_i_log_file_name << " to skip nonzero_activation_gen..." << endl;
				if(!nonz_w_log_file[pe][way_cnt].is_open())	cout << nonz_w_log_file_name << " open error!" << endl;
				else	cout << "Reading " << nonz_w_log_file_name << " to skip nonzero_activation_gen..." << endl;
				if(!nonz_m_log_file[pe][way_cnt].is_open())	cout << nonz_m_log_file_name << " open error!" << endl;
				else	cout << "Reading " << nonz_m_log_file_name << " to skip nonzero_activation_gen..." << endl;
				if(!nonz_f_log_file[pe][way_cnt].is_open())	cout << nonz_f_log_file_name << " open error!" << endl;
				else	cout << "Reading " << nonz_f_log_file_name << " to skip nonzero_activation_gen..." << endl;

//				while(!nonz_i_log_file[pe][way_cnt].eof()){
				while(1){
					nonz_i_log_file[pe][way_cnt] >> hex >> nonz_i_buf;
					if(nonz_i_log_file[pe][way_cnt].eof())
						break;
					packed_input[pe*(SIMD/WAY)+way_cnt].write(nonz_i_buf);
				}

//				while(!nonz_w_log_file[pe][way_cnt].eof()){
				while(1){
					nonz_w_log_file[pe][way_cnt] >> hex >> nonz_w_buf;
					if(nonz_w_log_file[pe][way_cnt].eof()){
						break;
					}
					packed_weight[pe*(SIMD/WAY)+way_cnt].write(nonz_w_buf);
				}

//				while(!nonz_m_log_file[pe][way_cnt].eof()){
				while(1){
					nonz_m_log_file[pe][way_cnt] >> hex >> nonz_m_buf;
					if(nonz_m_log_file[pe][way_cnt].eof()){
						break;
					}
					packed_mask[pe*(SIMD/WAY)+way_cnt].write(nonz_m_buf);
				}

//				while(!nonz_f_log_file[pe][way_cnt].eof()){
				while(1){
					nonz_f_log_file[pe][way_cnt] >> nonz_f_buf;
					if(nonz_f_log_file[pe][way_cnt].eof()){
						break;
					}
					sf_num[pe*(SIMD/WAY)+way_cnt].write(nonz_f_buf);
				}
			}
		}
	}
#endif

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
		WAY,
#ifdef ACTIVATION_LOG
		LayerCnt, (PE*TDstI::width),
#endif
		TDstElem,	// hwkim added for batch norm scale
		TSrcI, TDstI, TWeightI>
			(
			// ** hwkim modified for PE interleaving
//			static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>(mvOut),
//			static_cast<hls::stream<ap_uint<PE>>&>(mvOutMask),
			static_cast<hls::stream<ap_uint<2*PE*TDstI::width>>&>(mvOut),
			static_cast<hls::stream<ap_uint<2*PE>>&>(mvOutMask),

			// hwkim added for ternary
			static_cast<hls::stream<ap_uint<WAY*TSrcI::width>>*>(packed_input),
			static_cast<hls::stream<ap_uint<WAY>>*>(packed_weight),
			static_cast<hls::stream<ap_uint<FanInCntWidth>>*>(sf_num),
			static_cast<hls::stream<ap_uint<WAY>>*>(packed_mask),

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
