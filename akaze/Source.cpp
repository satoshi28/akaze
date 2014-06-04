/**
 * @file akaze_demo
 * @brief AKAZE detector + descritpor + BruteForce Matcher + drawing matches with OpenCV functions
 * @author A. Huaman
 * @updated Takahiro Poly Horikawa
 */

#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "akaze\akaze_features.h"

using namespace cv;


/**
 * @function main
 * @brief Main function
 */

void extractFeatures(cv::Mat image, std::vector<KeyPoint>& keypoints, cv::Mat& descriptors)
{
	Ptr<FeatureDetector> detector = FeatureDetector::create("AKAZE");
	double s, e, t;
	cv::Mat grayImage;

	if (image.channels()  == 3)
        cv::cvtColor(image, grayImage, CV_BGR2GRAY);
    else if (image.channels() == 4)
        cv::cvtColor(image, grayImage, CV_BGRA2GRAY);
    else if (image.channels() == 1)
        grayImage = image;

	//-- Step 1: Detect the keypoints using AKAZE Detector
	s = getTickCount();

	detector->detect( grayImage, keypoints );

	e = getTickCount();
	t = 1000.0 * (e-s) / getTickFrequency();

	printf("Detect keypoints: %f msec\n", t);
	
	//-- Step 2: Calculate descriptors (feature vectors)
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("AKAZE");
	
	s = getTickCount();
	
	extractor->compute( grayImage, keypoints, descriptors );
	e = getTickCount();
	t = 1000.0 * (e-s) / getTickFrequency();

	printf("Extract descriptors: %f msec\n", t);

}
int main( int argc, char** argv )
{

	const char *imageName1 = "C:\\Users\\satoshi\\Documents\\Image\\opencv.jpg";
	const char *imageName2 = "C:\\Users\\satoshi\\Documents\\Image\\tyuson\\2.jpg";

	cv::Mat img_1 = cv::imread(imageName1,0);
	cv::Mat img_2;

	img_1.resize(320,240);

	bool videoFlag = false;

	cv::VideoCapture cap(0);
	if(cap.open(0))
	{
		cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

		cap >> img_2;
		videoFlag = true;
	}
	else
		img_2 = cv::imread(imageName2,0);


  if( !img_1.data || !img_2.data )
  {
    std::cerr << " Failed to load images." << std::endl;
    return -1;
  }

  double s, e, t;

  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;
  std::vector< DMatch > matches;
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
  Mat img_matches;

  //‰æ‘œ1
  extractFeatures(img_1, keypoints_1, descriptors_1);


  //-- Step 3: Matching descriptor vectors with a brute force matcher
  while (true)
  {

	    //‰æ‘œ2
		extractFeatures(img_2, keypoints_2, descriptors_2);

		s = getTickCount();
  
		matcher->match( descriptors_1, descriptors_2, matches );
		e = getTickCount();
		t = 1000.0 * (e-s) / getTickFrequency();
		printf("Match descriptors: %f msec\n", t);
		
		//-- Draw matches
		drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches );
		
		//-- Show detected matches
		imshow("Matches", img_matches );


		//ƒJƒƒ‰‰æ‘œ‚Å‚Í‚È‚¢‚È‚çˆê‰ñ‚¾‚¯
		if(videoFlag == false)
		{
			break;
		}else
		{
			cap >> img_2;
		}

		waitKey(10);

  }

  

  waitKey(0);

  return 0;
}

