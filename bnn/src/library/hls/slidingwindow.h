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
 *******************************************************************************/

 /******************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *           Thomas B. Preusser <thomas.preusser@utexas.edu>
 *             Marie-Curie Fellow, Xilinx Ireland, Grant Agreement No. 751339
 *           Christoph Doehring <cdoehrin@xilinx.com>
 *
 *  @file slidingwindow.h
 *
 *  Library of templated HLS functions for BNN deployment. 
 *  This file lists a set of convenience funtions used to implement  
 *  Sliding window generator for convolutions
 *
 *****************************************************************************/

#ifndef SLIDINGWINDOW_H
#define SLIDINGWINDOW_H
 
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) > (y)) ? (y) : (x))

// hwkim modified for debug
//#define ACTIVATION_LOG
#include <fstream>

// sliding window unit that produces several vectors simultaneously for feeding
// a matrix multiple vectors unit
template<unsigned int ConvKernelDim, 
		 unsigned int IFMChannels,
		 unsigned int Input_precision,		// Number of bits for each pixel
		 unsigned int IFMDim, 
		 unsigned int OFMDim,
		 // hwkim modified for segmentation
		 unsigned int IFMHeight,
		 unsigned int OFMHeight,
		 unsigned int Top,
		 unsigned int Bottom,
		 unsigned int Left,
		 unsigned int Right,

		 unsigned int SIMD,
		 unsigned int Stride = 1>
void ConvolutionInputGenerator(
		stream<ap_uint<SIMD*Input_precision> > & in,
		stream<ap_uint<SIMD*Input_precision> > & out,
		// hwkim modified for debug
#ifdef ACTIVATION_LOG
		stream<ap_uint<SIMD*Input_precision> > & out_log,
#endif
		const unsigned int numReps = 1) {
  if(IFMChannels % SIMD != 0) {
    cout << "Error: IFM channels has to be a multiple of SIMD" << endl;
  }
  if(ConvKernelDim % Stride != 0) {
    cout << "Error: Kernel size has to be multiple of Stride" << endl;
  }
  // hwkim added for stride - constraining padding size up to kernel size
  if((Top > ConvKernelDim) ||
	  (Bottom > ConvKernelDim) ||
	  (Left > ConvKernelDim) ||
	  (Right > ConvKernelDim)){
	  cout << "Error: padding size is bigger than kernel size" << endl;
  }

  const unsigned int multiplying_factor = IFMChannels/SIMD;
  // hwkim modified for stride
  //const unsigned int number_blocks = ConvKernelDim/Stride + 1 ;
  //ap_uint<SIMD*Input_precision> inputBuf[number_blocks][Stride * IFMDim * multiplying_factor];
  const unsigned int number_blocks = ConvKernelDim + Stride;
  ap_uint<SIMD*Input_precision> inputBuf[number_blocks][IFMDim * multiplying_factor];

#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1
#pragma HLS RESOURCE variable inputBuf core=RAM_2P

  const unsigned int cycles_write_block = (OFMDim * ConvKernelDim * ConvKernelDim * multiplying_factor);
  const unsigned int cycles_read_block = Stride * IFMDim * multiplying_factor;
  const unsigned int max_cycles = MAX(cycles_write_block,cycles_read_block);
  // hwkim modified for padding
  //const unsigned int baseIter = IFMDim * ConvKernelDim * multiplying_factor		// Initial buffer
  const unsigned int baseIter = IFMDim * (ConvKernelDim-Top) * multiplying_factor		// Initial buffer

		  	  	  	  	  	  // hwkim modified for segmentation - support for rectangle
			                //+ OFMDim * MAX(cycles_write_block,cycles_read_block);
		  	  	  	  	  	  // hwkim modified for stride
		  	  	  	  	  	  + OFMHeight * MAX(cycles_write_block,cycles_read_block);

  unsigned int counter_internal_block = 0;
  unsigned int current_block_write = 0;
  // hwkim modified for stride
  unsigned int current_block_read = 0;

  unsigned int next_block_write = 0;	
  unsigned int current_line = 0;
  unsigned int read_block = 0; 
  // hwkim modified for padding
  //unsigned int inp = 0, ofm_y = 0, ofm_x = 0, k_y = 0, k_x = 0, count_simd =0;
  unsigned int inp = 0, k_y = 0, k_x = 0, count_simd =0;
  int ofm_y = -Top, ofm_x = -Left;

#pragma HLS reset variable=inp
  for (unsigned int count_image = 0; count_image < numReps; count_image++) {
    for (unsigned int i = 0; i < baseIter; i++) {
#pragma HLS PIPELINE II=1

    	// hwkim modified for padding
      //if (inp < IFMDim * ConvKernelDim * multiplying_factor) {// Initial buffer of ConvKernelDim lines
	  if (inp < IFMDim * (ConvKernelDim - Top) * multiplying_factor) {// Initial buffer of ConvKernelDim lines

    	  ap_uint<SIMD*Input_precision> inElem;

    	  inElem = in.read();
    	  inputBuf[current_block_write][current_line] = inElem;

    	  current_line++;
    	  inp++;

    	  // hwkim modified for stride
    	  //if (current_line == Stride * IFMDim * multiplying_factor ) {
		  if (current_line == IFMDim * multiplying_factor ) {

    		  current_line = 0;
    		  current_block_write++;

    		  if (current_block_write == number_blocks) {
    			  current_block_write=0;
    		  }
    		  read_block++;
    		  counter_internal_block = 0;
    	  }
      }
      else {
    	  if (counter_internal_block < cycles_write_block-1) { // We are writing output, MMV IFMChan per cycle
    		  // hwkim modified for stride
    		  //unsigned int current_block_read = (current_block_write + 1 + k_y / Stride);
    		  //unsigned int current_block_read = (current_block_write + Stride + k_y);
			  unsigned int current_line_in_block
			  	  // hwkim modified for stride
			  	  //= ((k_y%Stride) * IFMDim + ofm_x*Stride + k_x)*multiplying_factor + count_simd;
			  	  = (ofm_x*Stride + k_x)*multiplying_factor + count_simd;

			  // hwkim modified for stride
			  //ap_uint<SIMD*Input_precision> outElem = inputBuf[current_block_read][(current_line_in_block)];
			  unsigned int current_block_read_kernel = current_block_read + k_y;
			  if (current_block_read_kernel >= number_blocks) {
				  current_block_read_kernel-= number_blocks;
			  }
			  // hwkim modified for padding
			  //ap_uint<SIMD*Input_precision> outElem = inputBuf[current_block_read_kernel][(current_line_in_block)];
			  ap_uint<SIMD*Input_precision> outElem;
			  if(((ofm_y < 0) && (k_y < -ofm_y)) ||
				  ((ofm_x < 0) && (k_x < -ofm_x)) ||
				  ((ofm_y >= (OFMHeight-1-Bottom-1)) && (k_y > OFMHeight-Bottom-Bottom+1-ofm_y)) ||
				  ((ofm_x >= (OFMDim-1-Right-1)) && (k_x > OFMDim-Right-Right+1-ofm_x))){
				  cout << setw(10) << "(Skipped) ";	//skip
			  }
			  else{
				  outElem = inputBuf[current_block_read_kernel][(current_line_in_block)];
		    	  out.write(outElem);
				#ifdef ACTIVATION_LOG
		    	  out_log.write(outElem);
				#endif
			  }
    		  // hwkim added for debug
    		  cout << "ofm (" << ofm_y << "," << ofm_x << "), ";
    		  cout << "ky kx (" << k_y << "," << k_x << "), ";
    		  cout << "simd_cnt: " << count_simd;
    		  cout << ", current_block_write: " << current_block_write;
    		  cout << ", current_block_read: " << current_block_read << endl;

			  // hwkim modified for removing pad from conv in
//			  out.write(outElem);	  // hwkim modified for debug
//#ifdef ACTIVATION_LOG
//			  out_log.write(outElem);
//#endif
			  count_simd++;
			  if (count_simd == multiplying_factor) {
				  count_simd=0;
				  k_x++;
				  if (k_x == ConvKernelDim) {
					  k_x = 0;
					  k_y++;
					  if (k_y == ConvKernelDim) {
						  k_y = 0;
						  // hwkim added for debug
						  cout << "=================================================" << endl;
						  ofm_x ++;
						  // hwkim modified for padding
						  //if (ofm_x == OFMDim) {
							  //ofm_x = 0;
						  if (ofm_x == OFMDim-Left) {
							  ofm_x = -Left;

							  ofm_y++;
							  // hwkim added for stride
							  current_block_read = current_block_read + Stride;
							  if (current_block_read >= number_blocks) {
								  current_block_read-= number_blocks;
							  }

							  // hwkim modified for segmentation
							  //if (ofm_y == OFMDim) {
							  // hwkim modified for padding
							  //if (ofm_y == OFMHeight) {
								  //ofm_y = 0;
							  if (ofm_y == OFMHeight-Top) {
								  ofm_y = -Top;

								  inp = 0;
							  }}}}}
    	  }

    	  // hwkim modified for segmentation debug
          //if ((counter_internal_block < cycles_read_block-1) && (read_block<IFMDim/Stride)) {
    	  //if ((counter_internal_block < cycles_read_block-1) && (read_block<(OFMHeight*Stride))) {	//(360+2)/Stride)) {
    	  if ((counter_internal_block < cycles_read_block-1) && (read_block<(IFMHeight))) {	//(360+2)/Stride)) {

        	  // In parallel we write in the buffer, in the current block write if we still need to
        	  ap_uint<SIMD*Input_precision> inElem;

        	  // hwkim modified for debug
			  if(in.empty())
				  printf("ConvInpGen stream read empty!!\n");

        	  inElem = in.read();
        	  inputBuf[current_block_write][current_line] = inElem;
#pragma AP dependence variable=inputBuf intra false
#pragma AP dependence variable=inputBuf inter false
        	  current_line++;
        	  // hwkim modified for stride
        	  //if (current_line == Stride * IFMDim * multiplying_factor) {// We read the whole block, we change the next block in which we want to we
			  if (current_line == IFMDim * multiplying_factor) {

				// We filled up a block, let's not read until
        		  current_line = 0;
        		  read_block++;
        		  current_block_write++;
        		  if (current_block_write == number_blocks) {
        			  current_block_write=0;
        		  }
#pragma AP dependence variable=current_block_write intra false	
        	  }
          }

          counter_internal_block++; // = (counter_internal_block +1) % max_cycles;

          if (counter_internal_block == (max_cycles-1)) {
        	  counter_internal_block = 0;
          }
      }
      /* hwkim commented
       * end of else
       */

    } // End base_iter
	read_block = 0;
  } // End count_image

} // End generator



template<unsigned int ConvKernelDim,
		 unsigned int IFMChannels,
		 unsigned int Input_precision,		// Number of bits for each pixel
		 unsigned int IFMDim,
		 unsigned int OFMDim,
		 unsigned int IFMHeight,
		 unsigned int OFMHeight,
//		 unsigned int Top,	// 내가 구현하는 TConv에서는 padding, cropping 고려 안 해도 됨
//		 unsigned int Bottom,
//		 unsigned int Left,
//		 unsigned int Right,
		 unsigned int SIMD>
void TConvolutionInputGenerator(
		stream<ap_uint<SIMD*Input_precision> > & in,
		stream<ap_uint<SIMD*Input_precision> > & out,
		// hwkim modified for debug
#ifdef ACTIVATION_LOG
		stream<ap_uint<SIMD*Input_precision> > & out_log,
#endif
		const unsigned int numReps = 1) {
  if(IFMChannels % SIMD != 0) {
    cout << "Error: IFM channels has to be a multiple of SIMD" << endl;
  }

  const unsigned int multiplying_factor = IFMChannels/SIMD;
  const unsigned int number_blocks = 2 + 1;
  	  // hwkim's comment - output 1줄 만드는데 2줄이 필요한 경우가 있음. 따라서 2줄은 덮어쓰기되면 안되므로 여유분 1줄 필요

  ap_uint<SIMD*Input_precision> inputBuf[number_blocks][IFMDim * multiplying_factor];
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1
#pragma HLS RESOURCE variable inputBuf core=RAM_2P

  const unsigned int cycles_write_block = (((OFMDim/2*2) + (OFMDim/2*4)	// hwkim's comment - first line
		  + (OFMDim/2*1) + (OFMDim/2*2))	// hwkim's comment - second line
		  * multiplying_factor);
  	  // hwkim's comment - output row 중 가장 많은 write가 발생하는 cycle. 이것보다 적은 경우는 skip
  const unsigned int cycles_read_block = IFMDim * multiplying_factor;	// hwkim's comment - input 1줄 읽어오는 cycle
  const unsigned int max_cycles = MAX(cycles_write_block,cycles_read_block);

  const unsigned int baseIter = IFMDim * multiplying_factor		// Initial buffer
		  	  	  	  	  	    + IFMHeight * MAX(cycles_write_block,cycles_read_block);

  unsigned int counter_internal_block = 0;
  unsigned int current_block_write = 0;
  unsigned int current_block_read = 0;
  unsigned int next_block_write = 0;
  unsigned int current_line = 0;
  unsigned int read_block = 0;
  unsigned int inp = 0, ofm_y = 0, ofm_x = 0, k_y = 0, k_x = 0, count_simd =0;

#pragma HLS reset variable=inp
  for (unsigned int count_image = 0; count_image < numReps; count_image++) {
    for (unsigned int i = 0; i < baseIter; i++) {
#pragma HLS PIPELINE II=1

      if (inp < IFMDim * multiplying_factor) {	// Initial buffer of ConvKernelDim lines
    	  ap_uint<SIMD*Input_precision> inElem;
    	  inElem = in.read();
    	  inputBuf[current_block_write][current_line] = inElem;

    	  current_line++;
    	  inp++;

    	  if (current_line == IFMDim * multiplying_factor ) {
    		  current_line = 0;
    		  current_block_write++;
    		  if (current_block_write == number_blocks) {
    			  current_block_write=0;
    		  }
    		  read_block++;
    		  counter_internal_block = 0;
    	  }
      }
      else {
    	  // hwkim modified for Tconv
    	  //if (counter_internal_block < cycles_write_block-1) { // We are writing output, MMV IFMChan per cycle
    	  if (counter_internal_block <= cycles_write_block-1) { // We are writing output, MMV IFMChan per cycle

    		  unsigned int current_line_in_block;
    		  unsigned int current_block_read_kernel;
			  // hwkim added for Tconv - (:,0) 예외 처리
    		  if(ofm_x == 0){
    			  current_line_in_block = k_x * multiplying_factor + count_simd;
    		  }
    		  else{
    			  current_line_in_block = ((int)(ofm_x+1)/2 - 1 + k_x) * multiplying_factor + count_simd;
    		  }
			  // hwkim added for Tconv - (0,:),(:,0) 예외 처리
    		  if(ofm_y == 0){
    			  current_block_read_kernel = k_y;
    		  }
    		  else{
    			  current_block_read_kernel = current_block_read + k_y;
    		  }
			  if (current_block_read_kernel >= number_blocks) {
				  current_block_read_kernel-= number_blocks;
			  }
			  ap_uint<SIMD*Input_precision> outElem = inputBuf[current_block_read_kernel][(current_line_in_block)];

			  // hwkim modified for Tconv
			  if((ofm_y==0 && k_y==1)
				  || (ofm_x==0 & k_x==1)){
				  // hwkim added for debug
//				  if (count_simd == multiplying_factor-1)
//					  cout << "skipped" << endl;	//skip
		      }
		     else{
		    	 // hwkim added for debug
//		    	 if (count_simd == multiplying_factor-1) {
//					  cout << "OFM: (" << ofm_y << "+" << k_y << ", " << ofm_x << "+" << k_x << "), ";
//					  cout << "IFM: (" << current_block_read_kernel << ", " << (int)current_line_in_block/multiplying_factor << "), ";
//					  cout << "written block: " << current_block_write << ", ";
//					  cout << "counter_internal_block: " << counter_internal_block << endl;
//				  }

			  	  out.write(outElem);
#ifdef ACTIVATION_LOG
		    	  out_log.write(outElem);
#endif
		      }
			  count_simd++;

			  if (count_simd == multiplying_factor) {
				  count_simd=0;
				  k_x++;
				  //if (k_x == ((ofm_x-1)%2 + 1)) {
				  if (k_x == (!(ofm_x&0x1) + 1)) {

					  k_x = 0;
					  k_y++;
					  //if (k_y == ((ofm_y-1)%2 + 1)) {
					  if (k_y == (!(ofm_y&0x1) + 1)) {

						  k_y = 0;
						  ofm_x ++;
						  if (ofm_x == OFMDim) {
							  ofm_x = 0;
							  ofm_y++;
							  // hwkim added for stride
							  //if((ofm_y%2==1) && (ofm_y!=1))
							  if((ofm_y&0x1==1) && (ofm_y!=1))

								  current_block_read++;
							  if (current_block_read >= number_blocks) {
								  current_block_read-= number_blocks;
							  }

							  if (ofm_y == OFMHeight) {
								  ofm_y = 0;
								  inp = 0;
							  }}}}}
    	  }

    	  // hwkim modified for Tconv
    	  //if ((counter_internal_block < cycles_read_block-1) && (read_block<(IFMHeight))) {
    	  if ((counter_internal_block <= cycles_read_block-1) && (read_block<(IFMHeight))) {

        	  // In parallel we write in the buffer, in the current block write if we still need to
        	  ap_uint<SIMD*Input_precision> inElem;

        	  inElem = in.read();
        	  inputBuf[current_block_write][current_line] = inElem;
#pragma AP dependence variable=inputBuf intra false
#pragma AP dependence variable=inputBuf inter false

        	  current_line++;
			  if (current_line == IFMDim * multiplying_factor) {
				// We filled up a block, let's not read until
        		  current_line = 0;
        		  read_block++;
        		  current_block_write++;
        		  if (current_block_write == number_blocks) {
        			  current_block_write=0;
        		  }
#pragma AP dependence variable=current_block_write intra false
        	  }
          }

         counter_internal_block++; // = (counter_internal_block +1) % max_cycles;

          // hwkim modified for Tconv
         //if (counter_internal_block == (max_cycles-1)) {
         if (counter_internal_block == max_cycles) {

        	  counter_internal_block = 0;
          }
      }

    } // End base_iter
	read_block = 0;
  } // End count_image

} // End generator



//template<unsigned int ConvKernelDim,
//		 unsigned int IFMChannels,
//		 unsigned int Input_precision,		// Number of bits for each pixel
//		 unsigned int IFMDim,
//		 unsigned int OFMDim,
//		 unsigned int IFMHeight,
//		 unsigned int OFMHeight,
//		 unsigned int Top,
//		 unsigned int Bottom,
//		 unsigned int Left,
//		 unsigned int Right,
//		 unsigned int SIMD>
//void UpConvolutionInputGenerator(
//		stream<ap_uint<SIMD*Input_precision> > & in,
//		stream<ap_uint<SIMD*Input_precision> > & out,
//		// hwkim modified for debug
//#ifdef ACTIVATION_LOG
//		stream<ap_uint<SIMD*Input_precision> > & out_log,
//#endif
//		const unsigned int numReps = 1) {
//  if(IFMChannels % SIMD != 0) {
//    cout << "Error: IFM channels has to be a multiple of SIMD" << endl;
//  }
//  const unsigned int multiplying_factor = IFMChannels/SIMD;
//  const unsigned int number_blocks = 2 + 1;	// output 1 block(2줄)을 만들기 위해서는 input 2줄이 필요, 그리고 미리 읽어올 1줄
//  ap_uint<SIMD*Input_precision> inputBuf[number_blocks][IFMDim * multiplying_factor];
//#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1
//#pragma HLS RESOURCE variable inputBuf core=RAM_2P
//  const unsigned int cycles_write_block = (9 * IFMDim * multiplying_factor);	// output 2줄을 만들기 위해 write되는 activation 개수
//  const unsigned int cycles_read_block = IFMDim * multiplying_factor;	// input 1줄 읽어오는 cycle
//  const unsigned int max_cycles = MAX(cycles_write_block,cycles_read_block);
//  const unsigned int baseIter = IFMDim * 2 * multiplying_factor		// Initial buffer
//		  	  	  	  	  	    + IFMHeight * MAX(cycles_write_block,cycles_read_block);
//  unsigned int counter_internal_block = 0;
//  unsigned int current_block_write = 0;
//  unsigned int current_block_read = 0;
//  unsigned int next_block_write = 0;
//  unsigned int current_line = 0;
//  unsigned int read_block = 0;
//  unsigned int inp = 0, ofm_y = 0, ofm_x = 0, k_y = 0, k_x = 0, count_simd =0;
//
//#pragma HLS reset variable=inp
//  for (unsigned int count_image = 0; count_image < numReps; count_image++) {
//    for (unsigned int i = 0; i < baseIter; i++) {
//#pragma HLS PIPELINE II=1
//
//      if (inp < IFMDim * 2 * multiplying_factor) {// Initial buffer of ConvKernelDim lines
//    	  ap_uint<SIMD*Input_precision> inElem;
//    	  inElem = in.read();
//    	  inputBuf[current_block_write][current_line] = inElem;
//
//    	  current_line++;
//    	  inp++;
//
//    	  if (current_line == IFMDim * multiplying_factor ) {
//    		  current_line = 0;
//    		  current_block_write++;
//
//    		  if (current_block_write == number_blocks) {
//    			  current_block_write=0;
//    		  }
//    		  read_block++;
//    		  counter_internal_block = 0;
//    	  }
//      }
//      else {
//    	  if (counter_internal_block < cycles_write_block-1) { // We are writing output, MMV IFMChan per cycle
//			  unsigned int current_line_in_block = ((int)(ofm_x/2) + k_x)*multiplying_factor + count_simd;
//			  unsigned int current_block_read_kernel = (int)(current_block_read/2) + k_y;
//			  if (current_block_read_kernel >= number_blocks) {
//				  current_block_read_kernel-= number_blocks;
//			  }
//			  ap_uint<SIMD*Input_precision> outElem = inputBuf[current_block_read_kernel][(current_line_in_block)];
//
//			  // hwkim modified for upconv
//		     if((ofm_x<Left && k_x<Left)
//				  ||(ofm_y<Top && k_y<Top)
//				  ||(ofm_x>(OFMDim-1-Right) && k_x>(2-1-Right))
//				  ||(ofm_y>(OFMHeight-1-Bottom) && k_y>(2-1-Bottom))){
//		    	  ;	//skip
//		      }
//		     else{
//		    	  out.write(outElem);
//#ifdef ACTIVATION_LOG
//		    	  out_log.write(outElem);
//#endif
//		      }
//			  count_simd++;
//
//			  if (count_simd == multiplying_factor) {
//				  count_simd=0;
//				  k_x++;
//				  if (k_x == (ofm_x%2+1)) {
//					  k_x = 0;
//					  k_y++;
//					  if (k_y == (ofm_y%2+1)) {
//						  k_y = 0;
//						  ofm_x ++;
//						  if (ofm_x == OFMDim) {
//							  ofm_x = 0;
//							  ofm_y++;
//							  // hwkim added for stride
//							  current_block_read++;
//							  if (current_block_read >= number_blocks) {
//								  current_block_read-= number_blocks;
//							  }
//
//							  if (ofm_y == OFMHeight) {
//								  ofm_y = 0;
//								  inp = 0;
//							  }}}}}
//    	  }
//
//    	  if ((counter_internal_block < cycles_read_block-1) && (read_block<(IFMHeight))) {
//        	  // In parallel we write in the buffer, in the current block write if we still need to
//        	  ap_uint<SIMD*Input_precision> inElem;
//
//        	  inElem = in.read();
//        	  inputBuf[current_block_write][current_line] = inElem;
//#pragma AP dependence variable=inputBuf intra false
//#pragma AP dependence variable=inputBuf inter false
//
//        	  current_line++;
//			  if (current_line == IFMDim * multiplying_factor) {
//				// We filled up a block, let's not read until
//        		  current_line = 0;
//        		  read_block++;
//        		  current_block_write++;
//        		  if (current_block_write == number_blocks) {
//        			  current_block_write=0;
//        		  }
//#pragma AP dependence variable=current_block_write intra false
//        	  }
//          }
//
//          counter_internal_block++; // = (counter_internal_block +1) % max_cycles;
//
//          if (counter_internal_block == (max_cycles-1)) {
//        	  counter_internal_block = 0;
//          }
//      }
//
//    } // End base_iter
//	read_block = 0;
//  } // End count_image
//
//} // End generator


	#endif
