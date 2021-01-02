/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY 
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include "tiny_cnn/util/util.h"
#include <fstream>
#include <cstdint>
#include <algorithm>

#define CIFAR10_IMAGE_DEPTH (3)
// hwkim modified for segmentation
//#define CIFAR10_IMAGE_WIDTH (32)
//#define CIFAR10_IMAGE_HEIGHT (32)
#define CIFAR10_IMAGE_WIDTH (480)
#define CIFAR10_IMAGE_HEIGHT (360)

#define CIFAR10_IMAGE_AREA (CIFAR10_IMAGE_WIDTH*CIFAR10_IMAGE_HEIGHT)
#define CIFAR10_IMAGE_SIZE (CIFAR10_IMAGE_AREA*CIFAR10_IMAGE_DEPTH)

// hwkim modified for debug
//#define ACTIVATION_LOG
#ifdef ACTIVATION_LOG
#include <iostream>
#include <fstream>
#endif

namespace tiny_cnn {

/**
 * parse CIFAR-10 database format images
 *
 * @param filename [in] filename of database(binary version)
 * @param train_images [out] parsed images
 * @param train_labels [out] parsed labels
 * @param scale_min  [in]  min-value of output
 * @param scale_max  [in]  max-value of output
 * @param x_padding  [in]  adding border width (left,right)
 * @param y_padding  [in]  adding border width (top,bottom)
 **/
inline void parse_cifar10(const std::string& filename,
                          std::vector<vec_t> *train_images,
//						  	  const std::string& filename_mask,	// hwkim added for ternary
//						  	  std::vector<vec_t> *train_mask,	// hwkim added for ternary
                          std::vector<label_t> *train_labels,
                          float_t scale_min,
                          float_t scale_max,
                          int x_padding,
                          int y_padding)
{
    if (x_padding < 0 || y_padding < 0)
        throw nn_error("padding size must not be negative");
    if (scale_min >= scale_max)
        throw nn_error("scale_max must be greater than scale_min");

    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    if (ifs.fail() || ifs.bad())
        throw nn_error("failed to open file:" + filename);

    // hwkim added for ternary
//    std::ifstream imaskfs(filename_mask.c_str(), std::ios::in | std::ios::binary);
//    if (imaskfs.fail() || imaskfs.bad())
//        throw nn_error("failed to open file:" + filename_mask);

    // hwkim commented for segmentation
    //uint8_t label;
    
    std::vector<unsigned char> buf(CIFAR10_IMAGE_SIZE);

    // hwkim modified for segmentation - removing category
    //while (ifs.read((char*) &label, 1)) {
    while(!ifs.eof()){
        vec_t img;

        if (!ifs.read((char*) &buf[0], CIFAR10_IMAGE_SIZE)) break;

        if (x_padding || y_padding)
         {
            int w = CIFAR10_IMAGE_WIDTH + 2 * x_padding;
            int h = CIFAR10_IMAGE_HEIGHT + 2 * y_padding;

            img.resize(w * h * CIFAR10_IMAGE_DEPTH, scale_min);

            for (int c = 0; c < CIFAR10_IMAGE_DEPTH; c++) {
                for (int y = 0; y < CIFAR10_IMAGE_HEIGHT; y++) {
                    for (int x = 0; x < CIFAR10_IMAGE_WIDTH; x++) {
                        img[c * w * h + (y + y_padding) * w + x + x_padding]
                            = scale_min + (scale_max - scale_min) * buf[c * CIFAR10_IMAGE_AREA + y * CIFAR10_IMAGE_WIDTH + x] / 255;
                       }
                  }
              }
            // hwkim modified for padding
//            for(int c=0; c<CIFAR10_IMAGE_DEPTH; c++){
//            	for(int y=0; y<CIFAR10_IMAGE_HEIGHT+2; y++){
//            		for(int x=0; x<CIFAR10_IMAGE_WIDTH+2; x++){
//            			if(x==0 || x==CIFAR10_IMAGE_WIDTH+1
//            				|| y==0 || y==CIFAR10_IMAGE_HEIGHT+1)
//            				img[c*w*h + y*w + x] = 0;
//             		}
//             	}
//             }
// hwkim modified for debugging
#ifdef ACTIVATION_LOG
            std::ofstream parse_cifar10_log_file("parse_cifar10_log.txt");
            if(!parse_cifar10_log_file.is_open()){
            	std::cout << "parse_cifar_log_file open error!!" << std::endl;
            }
            parse_cifar10_log_file.precision(8);
            for(int c=0; c<CIFAR10_IMAGE_DEPTH; c++){
            	for(int y=0; y<h; y++){
            		for(int x=0; x<w; x++){
            			parse_cifar10_log_file << std::setw(15) << img[c*w*h + y*w + x] << "|";
            		}
            		parse_cifar10_log_file << std::endl;
            	}
            	parse_cifar10_log_file << std::string(100, '=') << std::endl;
            }
            parse_cifar10_log_file.close();
#endif
        }
        else
        {
            std::transform(buf.begin(), buf.end(), std::back_inserter(img),
            		// hwkim modified to match kjh network
//                [=](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 255; });
            		[=](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 256; });
        }

        train_images->push_back(img);
        // hwkim commented for segmentation
        //train_labels->push_back(label);
    }

    // hwkim added for ternary
//    while(!imaskfs.eof()){
//        vec_t img_mask;
//        if (!imaskfs.read((char*) &buf[0], CIFAR10_IMAGE_SIZE)) break;
//        if (x_padding || y_padding){
//        	std::cout << "error! padding for mask is not implemented!!" << std::endl;
//         }
//        else{
//            std::transform(buf.begin(), buf.end(), std::back_inserter(img_mask),
//					[=](unsigned char c) { return c; });
//         }
//        train_mask->push_back(img_mask);
//    }

}

} // namespace tiny_cnn
