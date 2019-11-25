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
#define L10_SIMD 	16

// hwkim modified for balancing pipeline stage latency
#define L0_PE 	16
#define L1_PE 	32
#define L2_PE 	16
#define L3_PE 	32
#define L4_PE 	16
#define L5_PE 	32
#define L6_PE 	16
#define L7_PE 	32
#define L8_PE 	16
#define L9_PE 	32
#define L10_PE 	11

// hwkim modified for balancing pipeline stage latency
#define L0_WMEM 	36
#define L1_WMEM 	36
#define L2_WMEM 	144
#define L3_WMEM 	144
#define L4_WMEM 	576
#define L5_WMEM 	576
#define L6_WMEM 	576
#define L7_WMEM 	144
#define L8_WMEM 	144
#define L9_WMEM 	36
#define L10_WMEM 	36

// hwkim modified for balancing pipeline stage latency
#define L0_TMEM 	4
#define L1_TMEM 	2
#define L2_TMEM 	8
#define L3_TMEM 	4
#define L4_TMEM 	16
#define L5_TMEM 	8
#define L6_TMEM 	8
#define L7_TMEM 	4
#define L8_TMEM 	4
#define L9_TMEM 	2
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

#ifdef FPGA_DEBUG

#define L0_ACT_SIZE_IN_64B	((L0_OFM_DIM*L0_OFM_HEIGHT*L0_OFM_CH)/64)
#define L1_ACT_SIZE_IN_64B	((L1_OFM_DIM*L1_OFM_HEIGHT*L1_OFM_CH)/64)
#define L2_ACT_SIZE_IN_64B	((L2_OFM_DIM*L2_OFM_HEIGHT*L2_OFM_CH)/64)
#define L3_ACT_SIZE_IN_64B	((L3_OFM_DIM*L3_OFM_HEIGHT*L3_OFM_CH)/64)
#define L4_ACT_SIZE_IN_64B	((L4_OFM_DIM*L4_OFM_HEIGHT*L4_OFM_CH)/64)
#define L5_ACT_SIZE_IN_64B	((L5_OFM_DIM*L5_OFM_HEIGHT*L5_OFM_CH)/64)
#define L6_ACT_SIZE_IN_64B	((L6_OFM_DIM*L6_OFM_HEIGHT*L6_OFM_CH)/64)
#define L7_ACT_SIZE_IN_64B	((L7_OFM_DIM*L7_OFM_HEIGHT*L7_OFM_CH)/64)
#define L8_ACT_SIZE_IN_64B	((L8_OFM_DIM*L8_OFM_HEIGHT*L8_OFM_CH)/64)
#define L9_ACT_SIZE_IN_64B	((L9_OFM_DIM*L9_OFM_HEIGHT*L9_OFM_CH)/64)
#define L10_ACT_SIZE_IN_64B	((L10_OFM_DIM*L10_OFM_HEIGHT*(L10_OFM_CH*24 + 64 - (L10_OFM_CH*24 % 64)))/64)
#define CAT_SIZE_IN_64B		((L10_OFM_DIM*L10_OFM_HEIGHT*16)/64)


#define L1_WORD_OFFSET		(L0_ACT_SIZE_IN_64B)
#define L2_WORD_OFFSET		(L1_WORD_OFFSET + L1_ACT_SIZE_IN_64B)
#define L3_WORD_OFFSET		(L2_WORD_OFFSET + L2_ACT_SIZE_IN_64B)
#define L4_WORD_OFFSET		(L3_WORD_OFFSET + L3_ACT_SIZE_IN_64B)
#define L5_WORD_OFFSET		(L4_WORD_OFFSET + L4_ACT_SIZE_IN_64B)
#define L6_WORD_OFFSET		(L5_WORD_OFFSET + L5_ACT_SIZE_IN_64B)
#define L7_WORD_OFFSET		(L6_WORD_OFFSET + L6_ACT_SIZE_IN_64B)
#define L8_WORD_OFFSET		(L7_WORD_OFFSET + L7_ACT_SIZE_IN_64B)
#define L9_WORD_OFFSET		(L8_WORD_OFFSET + L8_ACT_SIZE_IN_64B)
#define L10_WORD_OFFSET		(L9_WORD_OFFSET + L9_ACT_SIZE_IN_64B)
#define CAT_WORD_OFFSET		(L10_WORD_OFFSET + L10_ACT_SIZE_IN_64B)

#endif

#endif //__LAYER_CONFIG_H_
