#include <cv.h>
#include <cmath>
#include <iostream>
using namespace std;

class CylindricalWarping{
public:
	CylindricalWarping(){ };
	void warp(IplImage *src_img, IplImage *tar_img, float f){
		int width = src_img->width;
		int height = src_img->height;
		float half_w = (float)width/2;
		float half_h = (float)height/2;
		assert(width==tar_img->width && height==tar_img->height);
#pragma omp parallel for
		for(int y=0 ; y<height ; y++){
			for(int x=0 ; x<width ; x++){
				float tar_x = x - half_w;  // origin at (w/2, h/2)
				float tar_y = half_h - y;
				float src_x = f*tan(tar_x/f) + half_w;
				float src_y = half_h - sqrt(tar_x*tar_x + f*f)*tar_y/f;
				if(src_x<0 || src_x>=width || src_y<0 || src_y>=height)
					continue;
				CvScalar scalar = cvGet2D(src_img, (int)src_y, (int)src_x) ;
#pragma omp critical
				{
				cvSet2D(tar_img, y, x, scalar);
				}
			}
		}
	};
private:
};