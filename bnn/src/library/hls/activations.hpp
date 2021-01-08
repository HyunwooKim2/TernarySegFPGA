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
 *  @file activations.hpp
 *
 *  Library of templated HLS classes for BNN deployment. 
 *  This file lists a set of classes used to implement  
 *  threshold memory in neural network. 
 *
 *  This project has received funding from the European Union's Framework
 *  Programme for Research and Innovation Horizon 2020 (2014-2020) under
 *  the Marie Skłodowska-Curie Grant Agreement No. 751339.
 *
 *******************************************************************************/

// hwkim added for debug
#include <cstdio>

#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP
/**
 * General contract for activation functions.
 *
 * This class itself has no formal significance for the implementation
 * of the MVAU. Implementations of activation functions are encouraged
 * to implement it nonetheless to guarantee appropriate function
 * signatures.
 */
template<typename TA, typename TO>
class Activation {
public:
  TA init(unsigned const  nf, unsigned const  pe) const {
#pragma HLS inline
    return  TA(0);
  }

  /**
   * Compute the activation of the passed accumulator value accu in row idx.
   */
  TO activate(unsigned const  nf, unsigned const  pe, TA const &accu) const;
};

/**
 * A no-op activation that simply outputs the computed accumulator
 * output as the final result.
 */
template<typename T>
class PassThroughActivation : public Activation<T, T> {
public:
  T activate(unsigned const  nf, unsigned const  pe, T const &accu) const {
#pragma HLS inline
    return  accu;
  }
};

/**
 * Use a simple global threshold comparison as activation function.
 *
 * The constant threshold is initialized at construction.
 * The default comparison returns true if the threshold value is
 * smaller than the passed accumulator value.
 */
template<typename TA, typename Compare = std::less<TA>>
class ThresholdActivation : public Activation<TA, bool> {
  TA const  m_threshold;
public:
  ThresholdActivation(TA const &threshold) : m_threshold(threshold) {
#pragma HLS inline
  }

public:
  bool activate(unsigned const  nf, unsigned const  pe, TA const &accu) const {
#pragma HLS inline
    return  Compare()(m_threshold, accu);
  }
};

/**
 * Use a simple per-row threshold comparison as activation function.
 *
 * The thresholds are taken from an array indexed by output row.
 * It is currently public to allow direct initialization and
 * to make its name accessible for top-level HLS pragmas.
 *
 * The default comparison returns true if the threshold value defined for
 * the indexed row is smaller than the passed accumulator value.
 */
template<unsigned NF, unsigned PE, unsigned NumTH, 
	 typename TA, typename TR, int ActVal = 0, typename Compare = std::less<TA>>
class ThresholdsActivation {
public:
  //TA m_thresholds[PE][NF][NumTH];
  /* hwkim commented
   * TA
   * 	layer 0 -> ap_fixed<24,16>
   * 	other layers -> ap_int<16>
   * NumTH -> threshold 개수 -> API가 2이면 2-bit(ternary) 의미 -> threshold가 2개 필요
   */
  // hwkim modified for ternary
  TA m_thresholds[PE][NF][NumTH];
  
public:
  TA init(unsigned const  nf, unsigned const  pe) const {
#pragma HLS inline
    return  TA(0);
  }

public:
//  TR activate(unsigned const  nf, unsigned const  pe,  TA const &accu) const {
    TR activate(unsigned const  nf, unsigned const  pe,  TA const &accu, TA const fan_in) const {
#pragma HLS inline
		TR result=ActVal;
		// original code
		/*
		for(unsigned int i=0; i< NumTH; i++){
#pragma HLS unroll
		  result+=Compare()(m_thresholds[pe][nf][i], accu);
		}
		*/
		// hwkim modified for ternary
		TA act = accu + (m_thresholds[pe][nf][0]>>1);	// sign of thres is inverted by trainer
		if(act >= (fan_in>>1))
			result = (TR)1;
		else
			result = (TR)0;
//		TA thres_p_new =  (-m_thresholds[pe][nf][0])>>1 + fan_in>>1;	//threshold scaling and shift
//		TA thres_n_new =  (-m_thresholds[pe][nf][1])>>1 + fan_in>>1;
//		if(accu >= thres_p_new)
//			result = (TR)1;
//		else if(accu < thres_n_new)
//			result = (TR)(-1);
//		else
//			result = (TR)0;
		return result;
    }
};

// hwkim added for last fc layer
template<unsigned NF, unsigned PE, unsigned NumTH,
	 typename TA, typename TR,
	 typename TS,	// hwkim added for batch norm scale
	 int ActVal = 0, typename Compare = std::less<TA>
>
class PassThroughAndBatchNorm {
public:
  TA m_thresholds[PE][NF][NumTH];	// hwkim commented: NumTH index - 0 for P(ositive), 1 for N(egative) threshold
  // hwkim added for batch norm scale
  TS m_scales[PE];

public:
  TA init(unsigned const  nf, unsigned const  pe) const {
#pragma HLS inline
    return  TA(0);
  }

public:
  // hwkim modified for positive only accumulation
  //TR activate(unsigned const  nf, unsigned const  pe,  TA const &accu) const {
  TR activate(unsigned const  nf, unsigned const  pe,  TA const &accu, unsigned const fan_in) const {
#pragma HLS inline
    TR result=ActVal;
	// hwkim modified for batch norm scale
    TR half_thresholds = m_thresholds[pe][nf][0];
    half_thresholds = half_thresholds/2;
    result = ((accu + half_thresholds - (fan_in>>1)) * m_scales[pe]);	// << 1;	//full scale
	// hwkim added for debug
//	cout << dec;
//	cout << "(";
//	cout << accu << "+";
//	cout << half_thresholds << "-";
//	cout << (fan_in>>1) << ")*";
//	cout << m_scales[pe] << "=";
//	cout << result << endl;
    return result;
  }
};

template<unsigned NF, unsigned PE, unsigned NumTH,
	 typename TA, typename TR, int ActVal = 0, typename Compare = std::less<TA>>
class InputLayerActivation {
public:
  TA m_thresholds[PE][NF][NumTH];
  // hwkim: NumTH 0 -> positive, 1 -> negative threshold

public:
  TA init(unsigned const  nf, unsigned const  pe) const {
#pragma HLS inline
    return  TA(0);
  }

public:
//  TR activate(unsigned const  nf, unsigned const  pe,  TA const &accu) const {
    TR activate(unsigned const  nf, unsigned const  pe,  TA const &accu, TA const fan_in) const {
#pragma HLS inline
    TR result=ActVal;

    // hwkim modified for unsigned popcount
//    TA act = accu + m_thresholds[pe][nf][0];
    //cout << setprecision(9) << act << " "; //<< "accu" << pe << "=" << act << endl;

    // hwkim modified for unsigned popcount
//    if(act >= (TA)0)
//    	result = (TR)1;
//    else
//    	result = (TR)0;

    // hwkim modified for ternary
    result[0] = Compare()(m_thresholds[pe][nf][0], accu);
    result[1] = Compare()(m_thresholds[pe][nf][1], accu) & (~result[0]);	// zero mask

    // hwkim added for debug
//    cout << "pe: " << (int)pe;
//    cout << fixed;
//    cout.precision(8);
//    cout << ", th[0]: " << m_thresholds[pe][nf][0];
//    cout << ", th[1]: " << m_thresholds[pe][nf][1];
//    cout << ", accu: " << accu;
//    cout << hex << ", result: " << result << endl;

    // hwkim: original code
//	for(unsigned int i=0; i< NumTH; i++){
//#pragma HLS unroll
//      result+=Compare()(m_thresholds[pe][nf][i], accu);	// inequality sign "<="
//    }

    return result;
  }
};



#endif
