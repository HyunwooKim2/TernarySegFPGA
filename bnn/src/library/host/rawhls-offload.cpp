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
 * @file rawhls-offload.h
 *
 * Library of functions for compatible execution of HLS source code (SW execution)
 * 
 *
 *****************************************************************************/
 
#if defined(RAWHLS) && defined(OFFLOAD)

#include "foldedmv-offload.h"
#include <string.h>
#include <vector>
#include <iostream>

using namespace std;
using namespace tiny_cnn;

ExtMemWord * bufIn, * bufOut;
// hwkim added for ternary
ExtMemWord * bufInMask;

void FoldedMVInit(const char * attachName) {
  if (!bufIn) {
    bufIn = new ExtMemWord[INPUT_BUF_ENTRIES];
    if (!bufIn) {
      throw "Failed to allocate host buffer";
    }
  }
  if (!bufOut) {
    bufOut = new ExtMemWord[OUTPUT_BUF_ENTRIES];
    if (!bufOut) {
      throw "Failed to allocate host buffer";
    }
  }
  // hwkim added for ternary
  if (!bufInMask) {
	  bufInMask = new ExtMemWord[INPUT_BUF_ENTRIES];
    if (!bufInMask) {
      throw "Failed to allocate host buffer";
    }
  }
}

void FoldedMVDeinit() {
  delete bufIn;
  delete bufOut;
  bufIn = 0;
  bufOut = 0;
  // hwkim added for ternary
  delete bufInMask;
  bufInMask = 0;
}

void FoldedMVMemSet(
		// hwkim modified for batch norm scale
		int targetLayer,	//unsigned int targetLayer,

		unsigned int targetMem,
		unsigned int targetInd, unsigned int targetThresh, ExtMemWord val) {
  // call the accelerator in weight init mode
	/* hwkim commented
	 * FoldedMVMemSet(layerNo * 2, pe, line, 0, e);
	 * targetLayer - layerNo*2
	 * targetMem - pe
	 * targetInd - line (64-bit 단위)
	 * 		-> targetMem에서 64-bit 단위 index
	 * val - 64-bit word
	 */
  BlackBoxJam((ap_uint<64> *)bufIn, (ap_uint<64> *)bufOut, true, targetLayer, targetMem, targetInd, targetThresh, val, 0);
  /* hwkim commented
   * bufIn/bufOut - testbench buffer의 주소 -> DRAM의 주소
   * true -> memory init 수행한다는 의미
   */
}

// TODO implement batch execution version
void FoldedMVOffloadBinarized(const ExtMemWord * in, ExtMemWord * out, const unsigned int inBufWords, const unsigned int outBufWords, const unsigned int numImages) {
  // call the accelerator in compute mode
  BlackBoxJam((ap_uint<64> *)in, (ap_uint<64> *)out, false, 0, 0, 0, 0, 0, numImages);
}

#endif
