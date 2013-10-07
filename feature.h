#include <cv.h>
#include <vector>
#include <highgui.h>
#include <iostream>
#include <cmath>
#include "haar.h"
#include "featurepoint.h"
#define SIGMA_L 3
#define SIGMA_I 3
#define HARRIS_THRESHOLD 1

/////////////// class HarrisCorner //////////////// 
class HarrisCorner{
public:
	void find_feature(IplImage *img, std::vector<FeaturePoint> &feature, int level, int ch){
		cv::Mat x_grad= cv::Mat(img->height, img->width, CV_32FC1);
		cv::Mat y_grad= cv::Mat(img->height, img->width, CV_32FC1);
		gradient_map(img, x_grad, y_grad);
		cv::Mat hm = cv::Mat(img->height, img->width, CV_32FC1);
		hessian_feature(x_grad, y_grad, img->width, img->height, level, hm);
		int rate = pow(2.0,(double)level);
		int count = 0;
		 for(int y=1 ; y<img->height-1 ; y++){
			for(int x=1 ; x<img->width-1 ; x++){
				float value = hm.at<float>(y, x);
				if( value <= HARRIS_THRESHOLD || value != value)
					continue;
				bool flag = true;
				for(int dy=-1 ; dy<=1 ; dy++)
					for(int dx=-1 ; dx<=1 ; dx++)
						if(hm.at<float>(y+dy, x+dx) > value){
							flag = false;
							break;
						}
				if(flag){
					feature.push_back(FeaturePoint(64*ch, x*rate, y*rate, level));
					count++;
				}
			}
		 }
		 cout << "level = " << level << ", count = " << count << endl;
		 cout << "find feature done" << endl;
	}
private:
	void gradient_map(IplImage *src_img, cv::Mat &x_img, cv::Mat &y_img){
		for(int y=1 ; y<src_img->height-1 ; y++){
			for(int x=1 ; x<src_img->width-1 ; x++){
				CvScalar s0 = cvGet2D(src_img, y, x);
				//CvScalar sx0 = cvGet2D(src_img, y, x-1);
				CvScalar sx = cvGet2D(src_img, y, x+1);
				//CvScalar sy0 = cvGet2D(src_img, y-1, x);
				CvScalar sy = cvGet2D(src_img, y+1, x);
				x_img.at<float>(y, x) = sx.val[0] - s0.val[0];
				y_img.at<float>(y, x) = -sy.val[0] + s0.val[0];
			}
		}
	}
	void hessian_feature(cv::Mat &x_grad, cv::Mat &y_grad,  
						int w, int h, int level, cv::Mat &hm){
		 cv::Mat xx = cv::Mat(h, w, CV_32FC1);
		 cv::Mat xy = cv::Mat(h, w, CV_32FC1);
		 cv::Mat yy = cv::Mat(h, w, CV_32FC1);
		 for(int y=0 ; y<h ; y++){
			for(int x=0 ; x<w ; x++){
				float px = x_grad.at<float>(y, x);
				float py = y_grad.at<float>(y, x);
				xx.at<float>(y, x) = px*px;
				xy.at<float>(y, x) = px*py;
				yy.at<float>(y, x) = py*py;
			}
		 }
		 cv::Mat xblur = cv::Mat(h, w, CV_32FC1);
		 cv::Mat yblur = cv::Mat(h, w, CV_32FC1);
		 cv::Mat xyblur = cv::Mat(h, w, CV_32FC1);
		 cv::Size ksize;
		 ksize.width = 5;
		 ksize.height = 5;
		 cv::GaussianBlur(xx, xblur, ksize, SIGMA_I);
		 cv::GaussianBlur(xy, yblur, ksize, SIGMA_I);
		 cv::GaussianBlur(yy, xyblur, ksize, SIGMA_I);
		 for(int y=0 ; y<h ; y++){
			for(int x=0 ; x<w ; x++){
				float a00 = xblur.at<float>(y, x);
				float a01 = xyblur.at<float>(y, x);
				float a10 = xyblur.at<float>(y, x);
				float a11 = yblur.at<float>(y, x);
				float lamda_product = ((a00+a11)*(a00+a11)-(a00-a11)*(a00-a11)-4*a10*a01)/4;
				float lamda_add = a00+a11;
				hm.at<float>(y, x) = lamda_product - 0.05*lamda_add*lamda_add;
			}
		 }
		 
		 
	}
};


/////////////////////class Multi-scale Oriented Patch///////////////////////
class MSOP{
public:
	MSOP(int level, int max_num_f){  
		_num_level = level;
		_max_num_feature = max_num_f;
	};
	void find_feature(IplImage *image, std::vector<FeaturePoint> &feature, int ch){
		IplImage *gray_img = cvCreateImage(cvSize(image->width, image->height), IPL_DEPTH_8U, 1);
		//IplImage *all_img = cvCreateImage(cvSize(image->width, image->height), IPL_DEPTH_8U, ch);
		cvCvtColor( image, gray_img, CV_RGB2GRAY );
		//cvSmooth(gray_img, gray_img, CV_GAUSSIAN, 1.5, 1.5);
		cvSaveImage("gray.jpg", gray_img);
		std::vector<FeaturePoint> first_feature;
		HarrisCorner harris;
		int rate = 1;
		for(int i=0 ; i<_num_level ; i++){
			IplImage *img = cvCreateImage(cvSize(image->width/rate+1, image->height/rate+1), IPL_DEPTH_8U, 1);
			downsample(gray_img, img, rate);
			cvSmooth(img, img, CV_GAUSSIAN, SIGMA_L, SIGMA_L);
			//cvSaveImage("gray.jpg", img);
			harris.find_feature(img, first_feature, i, ch);
			rate *= 2;
		}
		non_max_supression(gray_img, first_feature, feature);
		
		if(ch==1)
			feature_descriptor(gray_img, feature, 1);
		else
			feature_descriptor(image, feature, ch);
		cv::Mat mat(image, 0);
		for(int i=0 ; i<feature.size() ; i++){
			cv::circle(mat, cv::Point(feature.at(i).x, feature.at(i).y), 3, CV_RGB(255,0,0));
		/*	CvScalar s;
			switch(feature.at(i).scale_level){
			case 0:
				s.val[2] = 255;
				break;
			case 1:
				s.val[0] = 255;
				break;
			case 2:
				s.val[1] = 255;
				break;
			}
			cvSet2D(image, feature.at(i).y, feature.at(i).x, s); */
		}

	}
private:
	int _num_level;
	int _max_num_feature;
	void downsample(IplImage *gray_img, IplImage *img, int rate){
		for(int y=0 ; y<gray_img->height ; y+=rate){
			for(int x=0 ; x<gray_img->width ; x+=rate){
				cvSet2D(img, y/rate, x/rate, cvGet2D(gray_img, y, x));
			}
		}
	}
	void non_max_supression(IplImage *gray_img, 
							std::vector<FeaturePoint> &feature, 
							std::vector<FeaturePoint> &refine_feature){
		float *intensity = new float[feature.size()];
		bool *tag = new bool[feature.size()];
		for(int i=0 ; i<feature.size() ; i++){
			CvScalar s = cvGet2D(gray_img, feature.at(i).y, feature.at(i).x);
			intensity[i] = s.val[0];
		}
		int radius = 25;
		// non-maximum supression
		while(radius>0){
			int count=0;
			for(int i=0 ; i<feature.size() ; i++)
				tag[i] = true;
			//begin
#pragma omp parallel for
			for(int i=0 ; i<feature.size() ; i++){
				if(!tag[i])
					continue;
				bool flag = true;
				for(int j=0 ; j<feature.size() ; j++){
					if((feature.at(i).x-feature.at(j).x)*(feature.at(i).x-feature.at(j).x)+
						(feature.at(i).y-feature.at(j).y)*(feature.at(i).y-feature.at(j).y) > radius*radius)
						continue;
					if(intensity[j]>intensity[i]){
#pragma omp critical
{
						tag[i] = false;
}
						flag = false;
						break;
					}
					else
#pragma omp critical
{
						tag[j] = false;
}
					}
				if(flag){
#pragma omp critical
{
					count++;
					tag[i] = true;
}
				}
			}
			// done
			cout << "radius = " << radius << ", ";
			cout << "# of features = " << count << endl;
			if(count > _max_num_feature)
				break;
			radius -= 1;
		}
		// refine features
		for(int i=0 ; i<feature.size() ; i++)
				if(tag[i])
					refine_feature.push_back(feature.at(i));
	}
	// find featuure descriptor
	void feature_descriptor(IplImage *gray_img, std::vector<FeaturePoint> &feature, int channel){
		IplImage **img = new IplImage*[10];
		int rate = 1;
		for(int i=0 ; i<_num_level ; i++){
			img[i] = cvCreateImage(cvSize(gray_img->width/rate+1, gray_img->height/rate+1), IPL_DEPTH_8U, channel);
			downsample(gray_img, img[i], rate);
			cvSmooth(img[i], img[i], CV_GAUSSIAN, (2 + 2*i)*2 + 1, (2 + 2*i)*2 + 1);
			rate *= 2;
		}
		bool *feature_tag = new bool[feature.size()];
		for(int i=0 ; i<feature.size() ; i++)
			feature_tag[i] = true;
		// begin
#pragma omp parallel for
		for(int i=0 ; i<feature.size() ; i++){
			int level = feature.at(i).scale_level;
			int rate = pow(2.0, (double)level);
			int y = feature.at(i).y/rate;
			int x = feature.at(i).x/rate;
			CvScalar s0 = cvGet2D(img[level], y, x);
			CvScalar sx = cvGet2D(img[level], y, x+1);
			CvScalar sy = cvGet2D(img[level], y+1, x);
			for(int ch=0 ; ch<channel ; ch++){
				float r = sqrt((sx.val[ch]-s0.val[ch])*(sx.val[ch]-s0.val[ch]) 
						+ (sy.val[ch]-s0.val[ch])*(sy.val[ch]-s0.val[ch]));
				//float cosine = -(sy.val[ch]-s0.val[ch]) / r;
				//float sine = (sx.val[ch]-s0.val[ch]) / r;
				float cosine = 1;
				float sine = 0;
				int count = 0;
				float vector_buf[64];
				float mean = 0, sr = 0, square_sum = 0;
				bool flag = true;
				for(int dy=-20 ; dy<20 ; dy+=5){
					for(int dx=-20 ; dx<20 ; dx+=5){
						float sum = 0;
						for(int ddy=dy ; ddy<dy+5 ; ddy++){
							for(int ddx=dx ; ddx<dx+5 ; ddx++){
								int trans_x = cosine*ddx - sine*ddy + x;
								int trans_y = sine*ddx + cosine*ddy + y;
								if(trans_x<0 || trans_y<0 || trans_x>=img[level]->width || trans_y>=img[level]->height){
									feature_tag[i] = false; // erase if out of boundary
									flag = false;
									break;
									}
								float intensity = cvGet2D(img[level], trans_y, trans_x).val[ch];
								if(intensity != intensity){
									feature_tag[i] = false; // erase if not a number
									flag = false;
									break;
									}
								sum += intensity;
								}
								if(!flag)
									break;
							}
							if(!flag)
								break;
							float intensity = sum/25;
							mean += intensity;
							square_sum += intensity*intensity;
							vector_buf[count] = intensity;
							count++;
					}
					if(!flag)
						break;
				}
				if(!flag)
					continue;
				mean /= 64;
				sr = sqrt(square_sum/64 - mean*mean);
				// Normalize
				float *haar_buf[8];
				for(int x=0 ; x<8 ; x++)
					for(int y=0 ; y<8 ; y++){
						haar_buf[x] = new float[8];
						haar_buf[x][y] = (vector_buf[x*8+y]-mean)/sr;
						//if(haar_buf[x][y]<-100 || haar_buf[x][y]>100)
						//cout << haar_buf[x][y] << " "; 
					}
				//cout << endl;
				//haar2(haar_buf, 8, 8);
				for(int x=0 ; x<8 ; x++)
					for(int y=0 ; y<8 ; y++){
						 feature.at(i).feature_vector[ch*64+x*8+y] = haar_buf[x][y];
					 //cout << haar_buf[x][y] << " "; 
					}
				//cout << endl;
				}
			}
			// end
			cout << "descriptor done" << endl;
			for(int i=feature.size()-1 ; i>=0 ; i--){
				if(!feature_tag[i])
					feature.erase(feature.begin()+i);
			}
			cout << "# of feature = " << feature.size() << endl;
		}
};