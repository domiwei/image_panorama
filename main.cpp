#include <cv.h>
#include <iostream>
#include <highgui.h>
#include "warping.h"
#include "matching.h"
#define INF 100000;
//#define DEBUG
int main(int argv, char *argc[])
{
	IplImage **images = new IplImage*[20];
	IplImage **warp_images = new IplImage*[20];
	std::vector<FeaturePoint> feature[20];
	//string path = string(argc[1]);
	int num_img = atoi(argc[3]);
	int channel = 1;
	int focal = atoi(argc[4]);
	CylindricalWarping warping;
	for(int i=0 ; i<num_img ; i++){
		char filename[100];
		//if(i<10)
			sprintf(filename, "%s/%s (%d).jpg", argc[1], argc[2], i+1);
		//else
			//sprintf(filename, "%s/grail%d.jpg", path.c_str(), i);
		images[i] = cvLoadImage(filename, CV_LOAD_IMAGE_COLOR);
			if(images[i]==NULL){
				std::cout << "loading error" << std::endl;
				exit(0);
			}
		warp_images[i] = cvCreateImage(cvSize(images[i]->width, images[i]->height), IPL_DEPTH_8U, 3);
		warping.warp(images[i], warp_images[i], focal);
		std::cout << "warping done." << std::endl;
		MSOP msop = MSOP(1, 700);
		msop.find_feature(warp_images[i], feature[i], channel);
		cout << "---------------Image " << i << " done-----------------" << endl; 
	}
	int dx, dy;
	int match;
	Match matching;
	//matching.translate_matching(feature[5], feature[6], 64, dx, dy, warp_images[5], warp_images[6], match);
	//cout <<"dx = " << dx << ", dy = " << dy << endl;
	
#ifdef DEBUG
	for(int i=0 ; i<num_img ; i++){
		double min = INF
		int nb = 0;
		for(int j=0 ; j<num_img ; j++){
			if(j==i)
				continue;
			double err = matching.translate_matching(feature[i], feature[j], 64, dx, dy, warp_images[i], warp_images[j], match); 
			if(min > err/match){
				min = err/match;
				nb = j;
			}
			cout << "err/match = " << err/match << ", dx = " << dx << ", dy = " << dy << endl;
		}
		cout << "img " << i << ", neighborhood is img " << nb << endl;
	}
#endif
#ifndef DEBUG
	int shift_x[20], shift_y[20];
	shift_x[num_img-1] = shift_y[num_img-1] = 0;
	for(int i=0 ; i<num_img-1 ; i++){
		double err = matching.translate_matching(feature[i], feature[i+1], 64, dx, dy, 
												warp_images[i], warp_images[i+1], match);
		shift_x[i] = dx;
		shift_y[i] = dy;
		cout << "err/match = " << err/match << ", dx = " << dx << ", dy = " << dy << endl;
	}
	IplImage *mask = cvCreateImage(cvSize(images[0]->width, images[0]->height), IPL_DEPTH_8U, 1);
	IplImage *warp_mask = cvCreateImage(cvSize(images[0]->width, images[0]->height), IPL_DEPTH_8U, 1);
	for(int y=0 ; y<mask->height ; y++)
		for(int x=0 ; x<mask->width ; x++){
			CvScalar s = cvGet2D(mask, y, x);
			s.val[0] = 1;
			cvSet2D(mask, y, x, s);
			s = cvGet2D(warp_mask, y, x);
			s.val[0] = 0;
			cvSet2D(warp_mask, y, x, s);
		}
	warping.warp(mask, warp_mask, focal);
	Stitching stitching;
	stitching.stitching(shift_x, shift_y, num_img, images, warp_mask, argc[1]); 
#endif
	/*IplImage *out = cvCreateImage(cvSize(test->width, test->height), IPL_DEPTH_8U, 3);
	CylindricalWarping warping;
	warping.warp(test, out, 705);
	std::cout << "warping done." << std::endl;
	MSOP msop = MSOP(3, 500);
	std::vector<FeaturePoint> feature;

	msop.find_feature(out, feature);*/
	//cvSaveImage("out.jpg", warp_images[0]);
	//cvSaveImage("out2.jpg", warp_images[1]);
	system("pause");
}