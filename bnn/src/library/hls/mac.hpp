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

/*****************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *           Thomas B. Preusser <thomas.preusser@utexas.edu>
 *             Marie-Curie Fellow, Xilinx Ireland, Grant Agreement No. 751339
 *           Christoph Doehring <cdoehrin@xilinx.com>
 *
 *  @file mac.hpp
 *
 *  Library of templated HLS functions for BNN deployment.
 *  This file lists a set of convenience funtions used to implement
 *  multipliers with selectable implementation resource
 *
 *  This project has received funding from the European Union's Framework
 *  Programme for Research and Innovation Horizon 2020 (2014-2020) under
 *  the Marie Skłodowska-Curie Grant Agreement No. 751339.
 *
 *****************************************************************************/
 
/*****************************************************************************
 * MAC operation template:
 *
 *   mac<N, T, TC, TD>(T a, TC c[N], TD d[N])
 *      = a + SUM_{i=0}^{N-1} c(i)*d(i)
 *
 * All template arguments but N can typically be inferred.
 *
 *   mac<ap_uint<14>>(0, c, d)
 *****************************************************************************/
 
#ifndef MAC_HPP
#define MAC_HPP

#include "utils.hpp"

// hwkim modified for debug
#include <iostream>

//- Multipliers with selectable implementation resource

//- Default: Let HLS choose
template<typename TC, typename TD>
auto mul(TC const &c, TD const &d, ap_resource_dflt const&) -> decltype(c*d) {
#pragma HLS inline
  auto  r = c*d;
  return  r;
}

//- Request LUT
template<typename TC, typename TD>
auto mul(TC const &c, TD const &d, ap_resource_lut const&) -> decltype(c*d) {
#pragma HLS inline
	/* hwkim commented
	 * caller
	 * 		res += mul(c[i], d[i], r);
	 * c -> wgt -> weights
	 * d -> act -> input
	 * r -> resource
	 * i -> SIMD #
	 */
  decltype(c*d) const  res = c*d;
#pragma HLS RESOURCE variable=res core=Mul_LUT
  return  res;
}

//- Request DSP48
template<typename TC, typename TD>
auto mul(TC const &c, TD const &d, ap_resource_dsp const&) -> decltype(c*d) {
#pragma HLS inline
  decltype(c*d) const  res = c*d;
#pragma HLS RESOURCE variable=res core=DSP48
  return  res;
}

//- MAC with selectable implementation resource
template<unsigned N, typename T, typename TC, typename TD, typename R>
T mac(T const &a, TC const &c, TD const &d, R const &r) {
	/* hwkim commented
	 * N -> SIMD
	 * a -> accu[pe] -> accumulation
	 * 		ThresholdsActivation class
	 * c -> wgt -> weights
	 * 		weights의 m_weights[pe][tile]에 연결된 참조자
	 * 		즉, pe 및 tile에 해당하는 weight를 가리키고 있음
	 * d -> act -> input
	 * 		TSrcI()(inElem)
	 * 			layer 0 -> ap_fixed<8,1>값 3개(SIMD)를 가진 24-bit
	 * r -> resource -> LUT
	 */
#pragma HLS inline
  T  res = a;
  for(unsigned  i = 0; i < N; i++) {
	  /* hwkim commented
	   * N == SIMD
	   * SIMD개 병렬 연산
	   * i -> SIMD 번호
	   */
#pragma HLS unroll
    res += mul(c[i], d[i], r);
	  /* hwkim commented
	   * r에 따라 다른 resource를 사용하는 mul이 호출
	   * 	-> LUT 또는 DSP48
	   * c[i], d[i]
	   * 	Slice class의 operator []가 호출
	   *	weight의 SIMD bits 중 해당 index에 해당하는 bits가 반환
	   * mul은 단순 곱하기
	   * 	-> operator*(곱하기)가 c[i]의 type에 따라 여러 개로
	   * 		overloading되어 있으며, layer 0을 제외한
	   * 		다른 layer는 XNOR로 연산하게 되어 있는 듯
	   * 	-> overloading 시, argument는 *연산자 우측, 즉 d[i]
	   * 	-> layer 0을 제외한 layer는 XNorMul class로 Recast되어 있음
	   * 		XNorMul class의 operator*는 아래와 같이 xnor로 구현
	   * 		m_val == b? 1 : 0;
	   * 		m_val과 b가 같으면 1, 다르면 0
	   */
    // hwkim modified for debug
    //cout << "res=c[" << i << "]*d[" << i << "]" << "=" << res << endl;
  }
  return  res;
}
template<unsigned N, typename T, typename TC, typename TD>
inline T mac(T const &a, TC const &c, TD const &d) {
#pragma HLS inline
  return  mac<N>(a, c, d, ap_resource_dflt());
}

#endif
