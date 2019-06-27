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
 ******************************************************************************/

/******************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *           Thomas B. Preusser <thomas.preusser@utexas.edu>
 *             Marie-Curie Fellow, Xilinx Ireland, Grant Agreement No. 751339
 *           Christoph Doehring <cdoehrin@xilinx.com>
 *
 *  @file dma.h
 *
 *  Library of templated HLS functions for BNN deployment. 
 *  This file lists a set of functions to access memory mapped values into 
 *  streams. 
 *
 *****************************************************************************/

#include <ap_int.h>
#include <hls_stream.h>

// hwkim modified for padding
unsigned int paddedSizeHW(unsigned int in, unsigned int padTo);

// essentially small DMA generators, moving data between mem-mapped arrays and streams
template<unsigned int DataWidth, unsigned int numBytes>
/* hwkim commented
 * DataWidth == 64
 * numBytes == 32x32x3
 */
void Mem2Stream(ap_uint<DataWidth> * in, hls::stream<ap_uint<DataWidth> > & out) {
  CASSERT_DATAFLOW(DataWidth % 8 == 0);
  // hwkim modified for padding
  //const unsigned int numWords = numBytes / (DataWidth / 8);
  const unsigned int numWords = paddedSizeHW(numBytes, (DataWidth / 8)) / (DataWidth / 8);

  /* hwkim commented
   * numWords == image 1장의 word 개수
   */
  CASSERT_DATAFLOW(numWords != 0);
  for (unsigned int i = 0; i < numWords; i++) {
#pragma HLS PIPELINE II=1
    ap_uint<DataWidth> e = in[i];
    /* hwkim commented
     * DataWidth -> 64-bit
     */
    out.write(e);
  }
}

template<unsigned int DataWidth, unsigned int numBytes>
void Stream2Mem(hls::stream<ap_uint<DataWidth> > & in, ap_uint<DataWidth> * out) {
  CASSERT_DATAFLOW(DataWidth % 8 == 0);
  // hwkim modified for numWords with remainder
  //const unsigned int numWords = numBytes / (DataWidth / 8);
  const unsigned int numWords = paddedSizeHW(numBytes, (DataWidth / 8)) / (DataWidth / 8);

  CASSERT_DATAFLOW(numWords != 0);
  for (unsigned int i = 0; i < numWords; i++) {
#pragma HLS PIPELINE II=1
    ap_uint<DataWidth> e = in.read();
	out[i] = e;
  }
}

// call different statically-sized variants of Mem2Stream and Stream2Mem to
// generate larger bursts when possible. otherwise, reading single images all
// the time limits the memory throughput.
// the 16 here can be any power of two (has to be power of two, otherwise
// checking the modulo takes a lot more resources)
template<unsigned int DataWidth, unsigned int numBytes>
void Mem2Stream_Batch(ap_uint<DataWidth> * in, hls::stream<ap_uint<DataWidth> > & out, const unsigned int numReps) {
	/* hwkim commented
	 * input array의 주소만 전달 받음 -> DRAM 상의 주소?
	 * out은 stream으로, DRAM에서 (out) stream으로 읽어오는 함수
	 */
	// hwkim modifiedf for padding
  //const unsigned int indsPerRep = numBytes / (DataWidth / 8);
  const unsigned int indsPerRep = paddedSizeHW(numBytes, (DataWidth / 8)) / (DataWidth / 8);

  /* hwkim commented
   * numBytes - image 1장 당 byte 수
   * indsPerRep - image 1장 당 word(64-bit) 수
   */
  unsigned int rep = 0;
  // make sure Mem2Stream does not get inlined here
  // we lose burst inference otherwise
  while (rep != numReps) {
    unsigned int repsLeft = numReps - rep;
    if ((repsLeft & 0xF) == 0) {
      // repsLeft divisable by 16, read 16 images
      Mem2Stream<DataWidth, numBytes * 16>(&in[rep * indsPerRep], out);
      /* hwkim commented
       * DataWidth -> 64-bit (word size)
       * in[] -> 64-bit으로 packed image
       *
       * input image가 여러 장일 때만 여기의 batch로 수행
       * numBytes*16 -> 16장 image 단위로 pipelining
       */
      rep += 16;
    } else {
      // fallback, read single image
    	/* hwkim commented
    	 * image가 1장일 경우
    	 */
      Mem2Stream<DataWidth, numBytes>(&in[rep * indsPerRep], out);
      /* hwkim commented
       * rep는 0, single image이므로
       * 따라서 in[0]의 주소가 전달 됨(맨 처음 주소)
       */
      rep += 1;
    }
  }
}
template<unsigned int DataWidth, unsigned int numBytes>
void Stream2Mem_Batch(hls::stream<ap_uint<DataWidth> > & in, ap_uint<DataWidth> * out, const unsigned int numReps) {
  // hwkim modified for indsPerRep with remainder
	//const unsigned int indsPerRep = numBytes / (DataWidth / 8);
	const unsigned int indsPerRep = paddedSizeHW(numBytes, (DataWidth / 8)) / (DataWidth / 8);

  unsigned int rep = 0;
  // make sure Stream2Mem does not get inlined here
  // we lose burst inference otherwise
  while (rep != numReps) {
    unsigned int repsLeft = numReps - rep;
    if ((repsLeft & 0xF) == 0) {
      // repsLeft divisable by 16, write 16 images
      Stream2Mem<DataWidth, numBytes * 16>(in, &out[rep * indsPerRep]);
      rep += 16;
    } else {
      // fallback, write single image
      Stream2Mem<DataWidth, numBytes>(in, &out[rep * indsPerRep]);
      rep += 1;
    }
  }
}
