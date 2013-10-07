#include <vector>
#include "feature.h"
#include <ctime>
#include <cmath>
#include <cv.h>
#define INF 10000000000;
#define THRESHOLD 0.9
class Match{
public:
	double translate_matching(std::vector<FeaturePoint> &feature1, 
							std::vector<FeaturePoint> &feature2, 
							int vector_size, int &shift_x, int &shift_y,
							IplImage *img1, IplImage *img2, int &match){
		srand(time(NULL));
		/// debug
		IplImage *img = cvCreateImage(cvSize(img1->width*2, img1->height), IPL_DEPTH_8U, 3);
		for(int y=0 ; y<img1->height ; y++)
			for(int x=0 ; x<img1->width ; x++)
				cvSet2D(img, y, x, cvGet2D(img1, y, x));
		for(int y=0 ; y<img1->height ; y++)
			for(int x=0 ; x<img1->width ; x++)
				cvSet2D(img, y, x+img1->width, cvGet2D(img2, y, x));
		
		cv::Mat mat(img, 0);
		/// debug
		int *best_match = new int[feature1.size()];
		int *best_match2 = new int[feature2.size()];
		for(int i=0 ; i<feature2.size() ; i++)
			best_match2[i] = 0;

		for(int i=0 ; i<feature1.size() ; i++){
			double min = INF;
			double min2 = INF;
			int index = 0;
			for(int j=0 ; j<feature2.size() ; j++){
				double dis = norm2(feature1.at(i).feature_vector, feature2.at(j).feature_vector, vector_size);
				if(min > dis){
					min2 = min;
					min = dis;
					index = j;
				}else{
					if(min2 > dis)
						min2 = dis;
				}
			}
			best_match[i] = index;
			//cout << min2 << endl;
			if(min/min2>0.5)
				best_match[i] = -1;
		}
		for(int i=0 ; i<feature2.size() ; i++){
			double min = INF;
			double min2 = INF;
			int index = 0;
			for(int j=0 ; j<feature1.size() ; j++){
				double dis = norm2(feature2.at(i).feature_vector, feature1.at(j).feature_vector, vector_size);
				if(min > dis){
					min2 = min;
					min = dis;
					index = j;
				}else{
					if(min2 > dis)
						min2 = dis;
				}
			}
			best_match2[i] = index;
			if(min/min2>0.5)
				best_match2[i] = -1;
		}
		for(int i=0 ; i<feature1.size() ; i++){
			if(best_match[i]<0)
				continue;
			else{
				if(best_match2[best_match[i]] != i)
					best_match[i] = -1;
			}
		}
		int count=0;
		for(int i=0 ; i<feature1.size() ; i++){
			if(best_match[i]>=0){
				//cv::line(mat, cv::Point(feature2.at(best_match[i]).x + img2->width, feature2.at(best_match[i]).y),
				//			  cv::Point(feature1.at(i).x, feature1.at(i).y), CV_RGB(0,0,255), 1);
				count++;
			}
		}
		//cout << "count = " << count << endl;
		//cout << "matching done!" << endl;
		IplImage final(mat);
		
		// ransac
		double min = INF;
		int best_x = 0, best_y = 0, best_index;
		bool *chosen = new bool[feature1.size()];
		for(int i=0 ; i<feature1.size() ; i++)
			chosen[i] = false;
		for(int k=0 ; k<count ; k++){
			int ran = rand()%(feature1.size());
			while(chosen[ran])
				ran = rand()%(feature1.size());
			chosen[ran] = true;
			if(best_match[ran]<0){
				k--;
				continue;
			}
			//cout << best_match[ran] << endl;
			int dx = feature2.at(best_match[ran]).x - feature1.at(ran).x;
			int dy = feature2.at(best_match[ran]).y - feature1.at(ran).y;
			double err = compute_error(feature1, feature2, best_match, dx, dy);
			if(min > err){
				min = err;
				best_x = dx;
				best_y = dy;
				best_index = ran;
			}
			//cout << "lala" << endl;
		}
		shift_x = best_x;
		shift_y = best_y;
		cv::line(mat, cv::Point(feature1.at(best_index).x + img2->width +shift_x, feature1.at(best_index).y+shift_y),
					  cv::Point(feature1.at(best_index).x, feature1.at(best_index).y), CV_RGB(0,0,255), 1);
		cvSaveImage("big.jpg", &final);	
		match = count;
		return min;
	}
private:
	double norm2(float *v1, float *v2, int size){
		double ret = 0;
		for(int i=0 ; i<size ; i++){
			ret += (v1[i]-v2[i])*(v1[i]-v2[i]);
			//cout << "ret = " << v1[i] << ", " << v2[i] << endl;
		}
		//cout << "ret = " << ret << endl;
		return ret;
	}
	double compute_error(std::vector<FeaturePoint> feature1, 
						 std::vector<FeaturePoint> feature2, 
						 int *match_index, int dx, int dy){
		double ret = 0;
		for(int i=0 ; i<feature1.size() ; i++){
			if(match_index[i]<0)
				continue;
			double x = (feature1.at(i).x + dx) - feature2.at(match_index[i]).x;
			double y = (feature1.at(i).y + dy) - feature2.at(match_index[i]).y;
			ret += sqrt(x*x + y*y);
		}
		return ret;
	}
};


class Stitching{
public:
	void stitching(int *shift_x, int *shift_y, int num_img, IplImage **img, IplImage *mask, char *path){
		int width = img[0]->width;
		int height = img[0]->height;
		int w = width+1, ph = 0, nh = 0, h = 0;
		for(int i=0 ; i<num_img-1 ; i++){
			w += shift_x[i];
			if(shift_y[i]>0)
				ph += shift_y[i];
			else
				nh += shift_y[i];
		}
		h = height + ph - nh + 3;
		//cout << "maxh = " << ph << ", minh = " << nh << endl;
		IplImage *panorama = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 3);
		for(int y=0 ; y<h ; y++)
			for(int x=0 ; x<w ; x++){
				CvScalar s = cvGet2D(panorama, y, x);
				for(int i=0 ; i<3 ; i++)
					s.val[i] = 0;
				cvSet2D(panorama, y, x, s);
			}
		//cout << cvGet2D(panorama, 0, 0).val[0] << endl;
			//make_mask(mask, num_img);
		//add_mask(panorama, mask, 0, 0);
		//for(int y=0 ; y<img[num_img-1].height ; y++)
			//for(int x=0 ; )
		
		int x = 0, y = -nh;
		//cout << "begin to stitch" << ", y = " << y << ", h = " << h << endl;
		for(int i = num_img-1 ; i>=0 ; i--){
			cout << "i = " << i << endl;
			x += shift_x[i];
			y += shift_y[i];
			cout << "begin to stitch" << ", y = " << y << ", h = " << h << endl;
			int end_x = (i>0 ? shift_x[i-1] : w - x - 2);
			make_mask(mask, i+1);
			add_mask(panorama, mask, x, y);	
			int countx=0, county=0;
			for(int dy=0 ; dy<panorama->height ; dy++){
				countx = 0;
				for(int dx=x ; dx<x + end_x ; dx++){
					int m = cvGet2D(panorama, dy, dx).val[0];
					if(m==i+1){
						assert(county-y>=0 && countx<width);
						cvSet2D(panorama, dy, dx, cvGet2D(img[i], county-y, countx));
					}else if(m==i+2){
						assert(county-(y-shift_y[i])>=0 && countx+shift_x[i]>=0);
						cvSet2D(panorama, dy, dx, cvGet2D(img[i+1], county-(y-shift_y[i]), countx+shift_x[i]));
					}else if(m==2*i+3){
						if(county-(y-shift_y[i])>=height || countx+shift_x[i]>= width || 
							county-(y-shift_y[i])<0 || countx+shift_x[i]<0 ||
							county-y>=height || county-y<0 || countx >= width)
							continue;
						CvScalar sr = cvGet2D(img[i], county-y, countx);
						CvScalar sl = cvGet2D(img[i+1], county-(y-shift_y[i]), countx+shift_x[i]);
						CvScalar s;
						float weight_r = countx;
						float weight_l = width-shift_x[i]-countx;
						for(int ch=0 ; ch<3 ; ch++)
							s.val[ch] = (sr.val[ch]*weight_r + sl.val[ch]*weight_l)/(weight_r + weight_l);
						cvSet2D(panorama, dy, dx, s);
					}else if(m==0){
						continue;
					}else{
						if(county-(y-shift_y[i])>=height || county-(y-shift_y[i])<0 || county-y>=height || county-y<0)
							continue;
						CvScalar sr = cvGet2D(img[i], county-y, countx);
						CvScalar sl = cvGet2D(img[i+1], county-(y-shift_y[i]), countx+shift_x[i]);
						CvScalar s;
						float weight_r = countx;
						float weight_l = width-shift_x[i]-countx;
						for(int ch=0 ; ch<3 ; ch++)
							s.val[ch] = (sr.val[ch]*weight_r + sl.val[ch]*weight_l)/(weight_r + weight_l);
						cvSet2D(panorama, dy, dx, s);
					}
					countx++;
				}
				county++;
			}
		}
		char file[200];
		sprintf(file, "%s/panorama.jpg", path);
		cvSaveImage(file, panorama);
	};
private:
	void make_mask(IplImage *mask, int value){
		for(int y=0 ; y<mask->height ; y++)
			for(int x=0 ; x<mask->width ; x++)
				if(cvGet2D(mask, y, x).val[0]){
					CvScalar s = cvGet2D(mask, y, x);
					s.val[0] = value;
					cvSet2D(mask, y, x, s);
				}
	}
	// 無聊斗咪陪你呀
	void add_mask(IplImage *pano, IplImage *mask, int startx, int starty){
		for(int y=starty ; y<starty+mask->height ; y++){
			for(int x=startx ; x<startx+mask->width ; x++){
				CvScalar s = cvGet2D(mask, y-starty, x-startx);
				CvScalar sp = cvGet2D(pano, y, x);
				sp.val[0] += s.val[0];
				cvSet2D(pano, y, x, sp);
			}
		}
		// 晚安，睡睡平安 :)
		//	 /-\
		//	 | |											   |
		//----/----------|---|---|---|---|---|---|---|----|----|--------|-------||
		//--/-_|_---4----|---|---|---|---|---|---|---|----|----|----|---|-------||
		//-|-|-|-\-------|---|---|---|---|---|---|---|----|----|----|---|-------||
		//-|-\-|-/--4----|---|---|---|---|---|---|---|----|---O-----|---|-------||
		//--\--|--------O---O---O----|--O---O---O----|---O----------|---|---O---||
		//	   |												   O
		//    Q
		//

	}
};