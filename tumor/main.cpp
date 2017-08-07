using namespace std;

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

Mat filter_custom(Mat img,Mat strel)
{
	Mat tempImg =  img.clone();
	for (int i=1; i<=img.rows-2; i++)
		for (int j=1; j<=img.cols-2; j++)
		{
			int sum =0;
			int p=0;
			for (int m=-((strel.rows-1)/2); m<=(strel.rows-1)/2; m++)
			{
				int q=0;
				for (int n=-((strel.cols-1)/2); n<=(strel.cols-1)/2; n++)
				{
					sum = sum + ((int)img.at<cv::Vec3b>(i+m,j+n)[0] * (int)strel.at<int>(p,q));
					//tempImg.at<cv::Vec3b>(i+m,j+n)[1] = (int)img.at<cv::Vec3b>(i+m,j+n)[1] + (int)strel.at<int>(p,q);
					//tempImg.at<cv::Vec3b>(i+m,j+n)[2] = (int)img.at<cv::Vec3b>(i+m,j+n)[2] + (int)strel.at<int>(p,q);
					q++;
				}
				p++;
			}

			tempImg.at<cv::Vec3b>(i,j)[0] = sum;
			tempImg.at<cv::Vec3b>(i,j)[1] = sum;
			tempImg.at<cv::Vec3b>(i,j)[2] = sum;
      	}

	return tempImg;
}

//driver function
void main()
{
	//reading input
	Mat img = imread("input.bmp");
	//imshow("Input - Medical Image",img);
	waitKey(0);

	//converting to grayscale
	Mat gs_bgr(img.size(), CV_8UC1);
	cvtColor(img, gs_bgr, CV_BGR2GRAY);
	imshow("Grayscale",img);
	waitKey(0);

	//Applying highpass filter for noise reduction
	Mat kern = (Mat_<int>(3,3) << 0,-1,0,-1,4,-1,0,-1,0);
	Mat highpassImg;
	//filter2D(img, highpassImg, img.depth(), kern);
	highpassImg = filter_custom(img, kern);
	imshow("Highpass - Noise reducted", highpassImg);
	waitKey(0);

	cout<< "Press any key to exit...";
	getchar();
}