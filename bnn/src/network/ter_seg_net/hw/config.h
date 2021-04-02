/**
 * Finnthesizer Config-File Generation
 *
 **/

#ifndef __LAYER_CONFIG_H_
#define __LAYER_CONFIG_H_

#define L0_K 3
#define L1_K 3
#define L2_K 3
#define L3_K 3
#define L4_K 3
#define L5_K 3
#define L6_K 3
#define L7_K 3
#define L8_K 3
#define L9_K 3
#define L10_K 3

#define L0_IFM_CH 	3
#define L1_IFM_CH 	64
#define L2_IFM_CH 	64
#define L3_IFM_CH 	128
#define L4_IFM_CH 	128
#define L5_IFM_CH 	256
#define L6_IFM_CH 	256
#define L7_IFM_CH 	128
#define L8_IFM_CH 	128
#define L9_IFM_CH 	64
#define L10_IFM_CH 	64

// hwkim modified for padding
//#define L0_IFM_DIM 	(480+2)
//#define L1_IFM_DIM 	(480+2)
//#define L2_IFM_DIM 	(480+1)
//#define L3_IFM_DIM 	(240+2)
//#define L4_IFM_DIM 	(240+1)
//#define L5_IFM_DIM 	(120+2)
//#define L6_IFM_DIM 	(120+1)
//#define L7_IFM_DIM 	(240+2)
//#define L8_IFM_DIM 	(240+1)
//#define L9_IFM_DIM 	(480+2)
//#define L10_IFM_DIM 	(480+2)

#define L0_IFM_DIM 	(480)
#define L1_IFM_DIM 	(480)
#define L2_IFM_DIM 	(480)
#define L3_IFM_DIM 	(240)
#define L4_IFM_DIM 	(240)
#define L5_IFM_DIM 	(120)
#define L6_IFM_DIM 	(120)
#define L7_IFM_DIM 	(240)
#define L8_IFM_DIM 	(240)
#define L9_IFM_DIM 	(480)
#define L10_IFM_DIM 	(480)

#define L0_OFM_CH 	64
#define L1_OFM_CH 	64
#define L2_OFM_CH 	128
#define L3_OFM_CH 	128
#define L4_OFM_CH 	256
#define L5_OFM_CH 	256
#define L6_OFM_CH 	128
#define L7_OFM_CH 	128
#define L8_OFM_CH 	64
#define L9_OFM_CH 	64
#define L10_OFM_CH 	11

#define L0_OFM_DIM 	480
#define L1_OFM_DIM 	480
#define L2_OFM_DIM 	240
#define L3_OFM_DIM 	240
#define L4_OFM_DIM 	120
#define L5_OFM_DIM 	120
#define L6_OFM_DIM 	240
#define L7_OFM_DIM 	240
#define L8_OFM_DIM 	480
#define L9_OFM_DIM 	480
#define L10_OFM_DIM 	480

// for base & base new perf
//#define L0_SIMD 	3
//#define L1_SIMD 	32
//#define L2_SIMD 	32
//#define L3_SIMD 	32
//#define L4_SIMD 	32
//#define L5_SIMD 	32
//#define L6_SIMD 	32
//#define L7_SIMD 	32
//#define L8_SIMD 	32
//#define L9_SIMD 	32
//#define L10_SIMD 	16
// for 2x perf
//#define L0_SIMD 	3
//#define L1_SIMD 	32
//#define L2_SIMD 	32
//#define L3_SIMD 	32
//#define L4_SIMD 	32
//#define L5_SIMD 	32
//#define L6_SIMD 	32
//#define L7_SIMD 	32
//#define L8_SIMD 	32
//#define L9_SIMD 	32
//#define L10_SIMD 	32
// for 4x perf
//#define L0_SIMD 	3
//#define L1_SIMD 	64
//#define L2_SIMD 	32
//#define L3_SIMD 	64
//#define L4_SIMD 	32
//#define L5_SIMD 	64
//#define L6_SIMD 	32
//#define L7_SIMD 	64
//#define L8_SIMD 	32
//#define L9_SIMD 	64
//#define L10_SIMD 	64
// hwkim modified for fast synthesis
//#define L0_SIMD 	3
//#define L1_SIMD 	8
//#define L2_SIMD 	8
//#define L3_SIMD 	8
//#define L4_SIMD 	8
//#define L5_SIMD 	8
//#define L6_SIMD 	8
//#define L7_SIMD 	8
//#define L8_SIMD 	8
//#define L9_SIMD 	8
//#define L10_SIMD	8
// for ternary zero skip
#define L0_SIMD 	3
#define L1_SIMD 	16	//32
#define L2_SIMD 	16	//32
#define L3_SIMD 	16	//32
#define L4_SIMD 	16	//32
#define L5_SIMD 	16	//32
#define L6_SIMD 	16	//32
#define L7_SIMD 	16	//32
#define L8_SIMD 	16	//32
#define L9_SIMD 	16	//32
#define L10_SIMD 	16

// hwkim modified for balancing pipeline stage latency
//#define L0_PE 	16
//#define L1_PE 	32
//#define L2_PE 	16
//#define L3_PE 	32
//#define L4_PE 	16
//#define L5_PE 	32
//#define L6_PE 	16
//#define L7_PE 	32
//#define L8_PE 	16
//#define L9_PE 	32
//#define L10_PE 	11
// for 2x perf
//#define L0_PE 	32
//#define L1_PE 	64
//#define L2_PE 	32
//#define L3_PE 	64
//#define L4_PE 	32
//#define L5_PE 	64
//#define L6_PE 	32
//#define L7_PE 	64
//#define L8_PE 	32
//#define L9_PE 	64
//#define L10_PE 	11
// for 4x performance
//#define L0_PE 	64
//#define L1_PE 	64
//#define L2_PE 	64
//#define L3_PE 	64
//#define L4_PE 	64
//#define L5_PE 	64
//#define L6_PE 	64
//#define L7_PE 	64
//#define L8_PE 	64
//#define L9_PE 	64
//#define L10_PE 	11
// for fast synthesis
//#define L0_PE 	8
//#define L1_PE 	8
//#define L2_PE 	8
//#define L3_PE 	8
//#define L4_PE 	8
//#define L5_PE 	8
//#define L6_PE 	8
//#define L7_PE 	8
//#define L8_PE 	8
//#define L9_PE 	8
//#define L10_PE 	11
// for ternary zero skip
#define L0_PE 	8	//16
#define L1_PE 	16	//32
#define L2_PE 	8	//16
#define L3_PE 	16	//32
#define L4_PE 	8	//16
#define L5_PE 	16	//32
#define L6_PE 	8	//16
#define L7_PE 	16	//32
#define L8_PE 	8	//16
#define L9_PE 	16	//32
#define L10_PE 	11

// hwkim modified for balancing pipeline stage latency
//#define L0_WMEM 	36
//#define L1_WMEM 	36
//#define L2_WMEM 	144
//#define L3_WMEM 	144
//#define L4_WMEM 	576
//#define L5_WMEM 	576
//#define L6_WMEM 	576
//#define L7_WMEM 	144
//#define L8_WMEM 	144
//#define L9_WMEM 	36
//#define L10_WMEM 	36

#define L0_WMEM		(3*3*(L0_IFM_CH/L0_SIMD)*(L0_OFM_CH/L0_PE))
#define L1_WMEM		(3*3*(L1_IFM_CH/L1_SIMD)*(L1_OFM_CH/L1_PE))
#define L2_WMEM		(3*3*(L2_IFM_CH/L2_SIMD)*(L2_OFM_CH/L2_PE))
#define L3_WMEM		(3*3*(L3_IFM_CH/L3_SIMD)*(L3_OFM_CH/L3_PE))
#define L4_WMEM		(3*3*(L4_IFM_CH/L4_SIMD)*(L4_OFM_CH/L4_PE))
#define L5_WMEM		(3*3*(L5_IFM_CH/L5_SIMD)*(L5_OFM_CH/L5_PE))
#define L6_WMEM		(3*3*(L6_IFM_CH/L6_SIMD)*(L6_OFM_CH/L6_PE))
#define L7_WMEM		(3*3*(L7_IFM_CH/L7_SIMD)*(L7_OFM_CH/L7_PE))
#define L8_WMEM		(3*3*(L8_IFM_CH/L8_SIMD)*(L8_OFM_CH/L8_PE))
#define L9_WMEM		(3*3*(L9_IFM_CH/L9_SIMD)*(L9_OFM_CH/L9_PE))
#define L10_WMEM	(3*3*(L10_IFM_CH/L10_SIMD)*(L10_OFM_CH/L10_PE))


// hwkim modified for balancing pipeline stage latency
//#define L0_TMEM 	4
//#define L1_TMEM 	2
//#define L2_TMEM 	8
//#define L3_TMEM 	4
//#define L4_TMEM 	16
//#define L5_TMEM 	8
//#define L6_TMEM 	8
//#define L7_TMEM 	4
//#define L8_TMEM 	4
//#define L9_TMEM 	2
//#define L10_TMEM 	1
#define L0_TMEM 	(L0_OFM_CH/L0_PE)
#define L1_TMEM 	(L1_OFM_CH/L1_PE)
#define L2_TMEM 	(L2_OFM_CH/L2_PE)
#define L3_TMEM 	(L3_OFM_CH/L3_PE)
#define L4_TMEM 	(L4_OFM_CH/L4_PE)
#define L5_TMEM 	(L5_OFM_CH/L5_PE)
#define L6_TMEM 	(L6_OFM_CH/L6_PE)
#define L7_TMEM 	(L7_OFM_CH/L7_PE)
#define L8_TMEM 	(L8_OFM_CH/L8_PE)
#define L9_TMEM 	(L9_OFM_CH/L9_PE)
#define L10_TMEM 	(L10_OFM_CH/L10_PE)

//#define L0_WPI 1

// hwkim commented: activation accumulation precision
#define L0_API 2
#define L1_API 2
#define L2_API 2
#define L3_API 2
#define L4_API 2
#define L5_API 2
#define L6_API 2
#define L7_API 2
#define L8_API 2
#define L9_API 2
#define L10_API 1

// hwkim added for ternary
//#define L0_NUMTH	2
//#define L1_NUMTH	2
//#define L2_NUMTH	2
//#define L3_NUMTH	2
//#define L4_NUMTH	2
//#define L5_NUMTH	2
//#define L6_NUMTH	2
//#define L7_NUMTH	2
//#define L8_NUMTH	2
//#define L9_NUMTH	2
//#define L10_NUMTH	1	// there's no activation function, threshold is offset (bias)

//#define L0_WPF 0
//#define L0_APF 0

// hwkim added for segmentation
// hwkim modified for padding
//#define L0_IFM_HEIGHT 	(360+2)
//#define L1_IFM_HEIGHT 	(360+2)
//#define L2_IFM_HEIGHT 	(360+1)
//#define L3_IFM_HEIGHT 	(180+2)
//#define L4_IFM_HEIGHT 	(180+1)
//#define L5_IFM_HEIGHT 	(90+2)
//#define L6_IFM_HEIGHT 	(90)
//#define L7_IFM_HEIGHT 	(180+2)
//#define L8_IFM_HEIGHT 	(180+1)
//#define L9_IFM_HEIGHT 	(360+2)
//#define L10_IFM_HEIGHT 	(360+2)

#define L0_IFM_HEIGHT 	(360)
#define L1_IFM_HEIGHT 	(360)
#define L2_IFM_HEIGHT 	(360)
#define L3_IFM_HEIGHT 	(180)
#define L4_IFM_HEIGHT 	(180)
#define L5_IFM_HEIGHT 	(90)
#define L6_IFM_HEIGHT 	(90)
#define L7_IFM_HEIGHT 	(180)
#define L8_IFM_HEIGHT 	(180)
#define L9_IFM_HEIGHT 	(360)
#define L10_IFM_HEIGHT 	(360)

#define L0_OFM_HEIGHT 	(360)
#define L1_OFM_HEIGHT 	(360)
#define L2_OFM_HEIGHT 	(180)
#define L3_OFM_HEIGHT 	(180)
#define L4_OFM_HEIGHT 	(90)
#define L5_OFM_HEIGHT 	(90)
#define L6_OFM_HEIGHT 	(180)
#define L7_OFM_HEIGHT 	(180)
#define L8_OFM_HEIGHT 	(360)
#define L9_OFM_HEIGHT 	(360)
#define L10_OFM_HEIGHT 	(360)

#define L0_WAY	3
#define L1_WAY	4
#define L2_WAY	4
#define L3_WAY	4
#define L4_WAY	4
#define L5_WAY	4
#define L6_WAY	4
#define L7_WAY	4
#define L8_WAY	4
#define L9_WAY	4
#define L10_WAY	4

// log2(fan-in/WAY)*WAY
#define L0_FANWIDTH	5	//3*3*3/(3/3)=27 -> 5-bit
#define L1_FANWIDTH	8	//3*3*64/(32/4)= -> 7-bit
#define L2_FANWIDTH	8
#define L3_FANWIDTH	9
#define L4_FANWIDTH	9
#define L5_FANWIDTH	10
#define L6_FANWIDTH	10
#define L7_FANWIDTH	9
#define L8_FANWIDTH	9
#define L9_FANWIDTH	8
#define L10_FANWIDTH	8


#endif //__LAYER_CONFIG_H_
