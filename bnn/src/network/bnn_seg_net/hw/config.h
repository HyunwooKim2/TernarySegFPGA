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

#define L0_SIMD 	3
#define L1_SIMD 	32
#define L2_SIMD 	32
#define L3_SIMD 	32
#define L4_SIMD 	32
#define L5_SIMD 	32
#define L6_SIMD 	32
#define L7_SIMD 	32
#define L8_SIMD 	32
#define L9_SIMD 	32
#define L10_SIMD 	32

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
#define L0_PE 	64
#define L1_PE 	64
#define L2_PE 	32
#define L3_PE 	32
#define L4_PE 	16
#define L5_PE 	32
#define L6_PE 	8
#define L7_PE 	32
#define L8_PE 	8
#define L9_PE 	64
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
//#define L10_WMEM 	18
#define L0_WMEM 	9
#define L1_WMEM 	18
#define L2_WMEM 	72
#define L3_WMEM 	144
#define L4_WMEM 	576
#define L5_WMEM 	576
#define L6_WMEM 	1152
#define L7_WMEM 	144
#define L8_WMEM 	288
#define L9_WMEM 	18
#define L10_WMEM 	18

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
#define L0_TMEM 	1
#define L1_TMEM 	1
#define L2_TMEM 	4
#define L3_TMEM 	4
#define L4_TMEM 	16
#define L5_TMEM 	8
#define L6_TMEM 	16
#define L7_TMEM 	4
#define L8_TMEM 	8
#define L9_TMEM 	1
#define L10_TMEM 	1

//#define L0_WPI 1

// NumTH
#define L0_API 1
#define L1_API 1
#define L2_API 1
#define L3_API 1
#define L4_API 1
#define L5_API 1
#define L6_API 1
#define L7_API 1
#define L8_API 1
#define L9_API 1
#define L10_API 1

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



#endif //__LAYER_CONFIG_H_
