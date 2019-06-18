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

// sliding window unit that produces several vectors simultaneously for feeding
// a matrix multiple vectors unit
template<unsigned int ConvKernelDim, 
		 unsigned int IFMChannels,
		 unsigned int Input_precision,		// Number of bits for each pixel
		 unsigned int IFMDim, 
		 unsigned int OFMDim,
		 unsigned int SIMD,
		 unsigned int Stride = 1>  			
/* hwkim commented
 * for layer 0
 * 		ConvKernelDim -> 3
 * 		IFMChannels -> 3
 * 		Input_precision -> TSrcI::width -> 8
 * 		IFMDim -> 32
 * 		OFMDim -> 30
 * 		SIMD -> 3
 */
void ConvolutionInputGenerator(
		stream<ap_uint<SIMD*Input_precision> > & in,
		stream<ap_uint<SIMD*Input_precision> > & out,
		// hwkim modified for debug
#ifdef ACTIVATION_LOG
		stream<ap_uint<SIMD*Input_precision> > & out_log,
#endif
		const unsigned int numReps = 1) {
/* hwkim commented
 * in의 ordering
 * 	->x->y
 * 	->c는 이미 24-bit에 ordering되어 있음
 */
  if(IFMChannels % SIMD != 0) {
    cout << "Error: IFM channels has to be a multiple of SIMD" << endl;
  }
  if(ConvKernelDim % Stride != 0) {
    cout << "Error: Kernel size has to be multiple of Stride" << endl;
  }
  const unsigned int multiplying_factor = IFMChannels/SIMD;
  /* hwkim commented
   * input channel에 대해 몇 번 연산해야 할 지
   */
  const unsigned int number_blocks = ConvKernelDim/Stride + 1 ;

  ap_uint<SIMD*Input_precision> inputBuf[number_blocks][Stride * IFMDim * multiplying_factor];
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1
#pragma HLS RESOURCE variable inputBuf core=RAM_2P
  /* hwkim commented
   * inputBuf -> input buffer array (stream 아니고 memory!!!)
   * SIMD 개수만큼의 input channel을 한 element로 담음
   * 	1-D index는 x,y 좌표
   * 	2-D index는 block 번호 (block = SIMD개의 channel x input 가로 한 줄)
   * RAM_2P -> A dual-port RAM
   * 	read operations on one port
   * 	both read and write operations on the other port
   */

  const unsigned int cycles_write_block = (OFMDim * ConvKernelDim * ConvKernelDim * multiplying_factor);
  /* hwkim commented
   * output 가로 한 줄에 대한 data 다 쓰기 위한 cycle?
   * output write은 stride와 무관계
   * output pixel 한 개 만들기 위해서는, conv x conv(3x3)개의 input read가 필요
   * 	-> 이 함수는 kernel 단위 sliding하면서 input 읽을 때,
   * 	   단순 순차적으로 읽기만 하면 되도록 out stream에 중복 저장하는 함수
   * 	-> 즉, cycles_write_block은
   * 		(X) 가로 3줄에 대해 out에 중복 저장을 완료할 때 걸리는 사이클 수
   * 		(O) output 가로 1줄 계산하기 위해 필요한 input을 모두 buffer에
   * 			write하는 데 걸리는	사이클 수
   */
  const unsigned int cycles_read_block = Stride * IFMDim * multiplying_factor;
  /* hwkim commented
   * input data 가로 한 줄 SIMD 단위로 읽어올 때 cycle
   * 	stride가 있으면 건너뛰면서 읽으므로, index가 x stride만큼 더 많이 가 있음???
   */
  const unsigned int max_cycles = MAX(cycles_write_block,cycles_read_block);
  /* hwkim commented
   * read/write 중 긴 것에 맞춤
   */
  const unsigned int baseIter = IFMDim * ConvKernelDim * multiplying_factor// Initial buffer
			                  + OFMDim * MAX(cycles_write_block,cycles_read_block);
  /* hwkim commented
   * 위의 것은 초기에 input 가로 한 줄(모든 채널)을 read해 오기 위한 cycle
   * 아래의 것은 output 가로 한 줄을 계산하기 위해 필요한 input을 buffer에
   * 	write하는 데 걸리는 cycle
   * 즉, 초기에 일단 input 한 줄(모든 채널)을 읽어오고(한 번만 수행)
   * 그 이후에는 output 한 줄 씩을 계산하는 데 필요한 input을 read 및 write
   */
  unsigned int counter_internal_block = 0;
  unsigned int current_block_write = 0;
  unsigned int next_block_write = 0;	
  unsigned int current_line = 0;
  unsigned int read_block = 0; 
  unsigned int inp = 0, ofm_y = 0, ofm_x = 0, k_y = 0, k_x = 0, count_simd =0;

#pragma HLS reset variable=inp
  /* hwkim commented
   * inp 변수에 reset logic 생성
   */
  for (unsigned int count_image = 0; count_image < numReps; count_image++) {
    for (unsigned int i = 0; i < baseIter; i++) {
#pragma HLS PIPELINE II=1

      /* hwkim commented
       * Block Read!
       */
      if (inp < IFMDim * ConvKernelDim * multiplying_factor) {// Initial buffer of ConvKernelDim lines
    	  /* hwkim commented
    	   * ConvKernelDim+1 lines만큼 3줄의 input을 read해서
    	   * 	기존 order(c->x->y)대로 inputBuf에 write하는 부분
    	   * 초기 1번만 수행
    	   *     ┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┐
    	   *   ┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┐┤
    	   * ┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┐┤┤
    	   * ├┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┤┤┘
    	   * ├┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┤┘
    	   * └┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┘
    	   */
    	  ap_uint<SIMD*Input_precision> inElem;

    	  inElem = in.read();
    	  inputBuf[current_block_write][current_line] = inElem;
    	  /* hwkim commented
    	   * input read!
    	   * in은 wa_in -> input image(activation)로 채워진 stream
    	   * 	(c->)x->y ordering
    	   * input을 SIMD channel 만큼 읽어서 inputBuf(memory)에 저장
    	   * 	layer 0의 경우 24-bit
    	   */

    	  current_line++;
    	  inp++;
    	  /* hwkim commented
    	   * current_line/inp -> SIMD 단위 index, SIMD->x->y 순
    	   * 두 변수가 초기화하는 순간이 다름
    	   * inp는 초기 buf read할 때만 증가(현재 if문)
    	   */
    	  if (current_line == Stride * IFMDim * multiplying_factor ) {
    		  /* hwkim commented
    		   * 가로 한 줄 읽을 때마다
    		   * 	-> current_line 초기화
    		   * 	  (current_line은 한 줄마다 증가하는 것이 아니라 매 SIMD read마다 증가)
    		   * 	-> current_block_write 증가
    		   * 	  (input을 read해서 inputBuf에 한 줄 write했다는 의미)
    		   *     ┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┐
    		   *   ┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┐┘
    		   * ┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┐┘
    		   * └┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┘
    		   */
    		  current_line = 0;
    		  current_block_write++;
    		  /* hwkim commented
    		   * current_block_write는 가로 한 줄 read해서 inputBuf에 쓸 때마다 1씩 증가
    		   * 	-> 즉, inputBuffer[current_block_write][]의 2-D index는 가로 한 줄 단위
    		   */
    		  if (current_block_write == number_blocks) {
    			  /* hwkim commented
    			   * number_block는 kernel 단위 block(가로 3줄)+1
    			   * 	-> +1은 속도 차이 완충을 위해?
    			   * 	-> 맨 위의 if 문에 의해 3줄만 되도 빠져나갈 텐데?
    			   * current_block_write는 kernel 세로 길이 만큼(4줄) 읽어올 때마다 초기화
    			   */
    			  current_block_write=0;
    		  }
    		  read_block++;
    		  /* hwkim commented
    		   * 한 줄 읽을 때마다 read_block도 1씩 증가
    		   * 	-> current_block_write와 동일하게 증가
    		   * 	-> 아래 inputBuf read할 때 어디까지 write했는 지를 확인할 수 있게?
    		   */
    		  counter_internal_block = 0;
    		  /* hwkim commented
    		   * current_line과 동일하게 초기화
    		   * 가로 1줄 다 읽으면 초기화
    		   */
    	  }
      }

      /* hwkim commented
       * Block Write!
       */
      else {
    	  /* hwkim commented
    	   * 위 if문의 저자 comment에 의하면
    	   * 	-> 초기 buffer read 완료 후 실행하는 부분인 듯
    	   * 	-> (첫 convkernel line수에 해당하는 read 완료 후)
    	   * 32 * 3 * (input channel / SIMD)
    	   * 아래 그림과 같이 input 가로 세 줄만큼 input을 초기에 다 읽은 경우
    	   *     ┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┐
    	   *   ┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┐┤
    	   * ┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┐┤┤
    	   * ├┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┤┤┘
    	   * ├┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┤┘
    	   * └┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┘
    	   * 한 block(가로 한 줄) 씩 parallel하게
    	   * 	in stream에서 read해서 inputBuf에 채우고
    	   * 	inputBuf로부터 out stream에 write
    	   */
    	  if (counter_internal_block < cycles_write_block-1) { // We are writing output, MMV IFMChan per cycle
    		  /*
    		   * (OFMDim * ConvKernelDim * ConvKernelDim * multiplying_factor);
    		   */
    		  /* hwkim commented
    		   * 여기는 out stream에 write하는 부분(block write)
    		   * counter_internal_block은 여기 상위 else문을 돌면서 계속 1씩 증가
    		   * counter_internal_block이 write에 필요한 cycle보다 적은 동안
    		   * 	현재 if 구문인 write 수행
    		   * 	즉, write cycle 동안 write 실행한다는 의미
    		   * counter_interanl_block은 몇 cycle 실행했는지를 count하는 counter
    		   *
    		   * output 가로 한 줄 계산에 필요한 input을 out stream에 write 수행
    		   * 	-> for문 계속 돌면서 output 가로 한 줄에 해당하는 input
    		   * 		다 write할 때까지 수행
    		   * 	-> write 끝났으면(counter_internal_block이 write cycle만큼 됐으면)
    		   * 		아래 in stream에서 input read해서
    		   * 		inputBuf 채울 때까지(block read) skip
    		   */
    		  unsigned int current_block_read = (current_block_write + 1 + k_y / Stride);
    		  /* hwkim commented
    		   * current_block_write 역시, 위의 초기 buf read(상단 if 문) 이후에는
    		   * 	여기 else 문에서 관리
    		   * 	아래 in stream read 및 inputBuf write하는 구문에서
    		   * 		한 줄 읽을 때마다 ++
    		   * current_block_read가 current_block_write + 1인 것은
    		   * 	항상 current_block_write+1의 block만 읽기 위해
    		   * 	뒤의 k_y는 다음 줄과 그 다음 줄(convolution 시 접근이
    		   * 	kernel 접근 순서대로 이루어지므로)
    		   */
    		  if (current_block_read >= number_blocks) {
    			  current_block_read-= number_blocks;
    			  /* hwkim commented
    			   * current_block_read가 inputBuf의 2-D 배열 인덱스 넘어갈 때
    			   * 0으로 초기화(circular)
    			   */
    		  }
			  unsigned int current_line_in_block
			  	  = ((k_y%Stride) * IFMDim + ofm_x*Stride + k_x)*multiplying_factor + count_simd;
			  /* hwkim commented
			   * 단순 (c->)x->y로 ordering되어 저장되어있는 inputBuf에서
			   * kernel 단위 중복되도록 out stream에 변환하여 저장하는 부분
			   * 	-> 즉, kernel 단위 sliding하면서 input 읽을 때,
			   * 	   그냥 out stream에서 순서대로 읽으면 되도록 중복 저장
			   * current_line_in_block은 inputBuf에서 current block 내 read 순서
			   * 	-> element 하나가 SIMD 개의 input channel 담고 있음
			   * 	-> count_simd는 모든 input channel 내에서 몇 번째 SIMD인가
			   * 	   (ex. input channel 9개이고 simd 3이면, count_simd는 0~2)
			   * ofm_y는 current_block_read에 의해 결정되는 듯?
			   * k_y, k_x는 kernel 단위 index
			   * ofm_y, ofm_x는 output 단위 index
			   */
			  ap_uint<SIMD*Input_precision> outElem = inputBuf[current_block_read][(current_line_in_block)];
			  out.write(outElem);
			  // hwkim modified for debug
#ifdef ACTIVATION_LOG
			  out_log.write(outElem);
#endif
			  count_simd++;

			  if (count_simd == multiplying_factor) {
				  count_simd=0;
				  k_x++;
				  if (k_x == ConvKernelDim) {
					  k_x = 0;
					  k_y++;
					  if (k_y == ConvKernelDim) {
						  k_y = 0;
						  ofm_x ++;
						  if (ofm_x == OFMDim) {
							  ofm_x = 0;
							  ofm_y++;
							  if (ofm_y == OFMDim) {
								  ofm_y = 0;
								  inp = 0;
								  /* hwkim commented
								   * ofm_y==OFMDim이라는 것은 image 한 장 다 읽었다는 의미
								   * inp는 out 전체를 계산할 수 있는 input(image) 전부를 읽어야 초기화 됨
								   * 	-> 즉, 위의 초기 buffer read하는 if문은 image 다 읽을 동안 처음 한 번만 수행됨
								   */
							  }
						  }
					  }
				  }
			  }

    	  }

          if ((counter_internal_block < cycles_read_block-1) && (read_block<IFMDim/Stride)) {
        	  // In parallel we write in the buffer, in the current block write if we still need to
        	  /* hwkim commented
        	   * 한 줄 read 아직 다 안 했으면 read
        	   * 위의 write와 read가 한 줄 단위로 parallel로 수행되는 듯
        	   * 	-> 즉, cycles_read_block이 짧다면, for문 계속 도는 동안 write는 if문 만족해서 계속 하고 read는 skip
        	   * 	-> cycles_write_block이 짧다면, for문 계속 도는 동안 read는 if문 만족해서 계속 하고 write는 skip
        	   * read_block은 input 가로 한 줄 다 읽을 때마다 증가
        	   * 	-> 위의 if문에서 (read_block<IFMDim/Stride) 조건은
        	   * 		input image의 세로 방향으로 모두 읽었는지
        	   * 		즉, input image 전체를 다 읽었는지 판단
        	   */
        	  ap_uint<SIMD*Input_precision> inElem;
        	  inElem = in.read();
        	  inputBuf[current_block_write][current_line] = inElem;
#pragma AP dependence variable=inputBuf intra false
#pragma AP dependence variable=inputBuf inter false
        	  current_line++;
        	  /* hwkim commented
        	   * current_line은 매 for문마다 증가
        	   * in stream read(inputBuf write)일 때만 증가
        	   * line이라기보다 pixel 단위
        	   */
        	  if (current_line == Stride * IFMDim * multiplying_factor) {// We read the whole block, we change the next block in which we want to we
				// We filled up a block, let's not read until
        		  /* hwkim commented
        		   * in stream에서 가로 한 줄 다 read했으면
        		   * 	-> current_line 초기화
        		   * 	-> read_block 증가
        		   * 	-> current_block_write 증가
        		   * 	   (inputBuf read/out stream write하는 루틴에서 참조)
        		   */
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
        	  /* hwkim commented
        	   * counter_internal_block은 for문 매 사이클마다 증가
        	   * read/write 중 cycle이 긴 것이 끝날 때까지 증가하고,
        	   * 	끝나면 초기화
        	   */
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

#endif
