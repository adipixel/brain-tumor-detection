using namespace std;

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

Mat convertBinary(Mat img, int threshold)
{
	Mat binImg = Mat(img.rows,img.cols,img.type());

	for(int i =0; i<img.rows; i++)
		for(int j=0; j<img.cols; j++)
		{
			cv::Vec3b intensity = img.at<cv::Vec3b>(i,j);

			if(intensity.val[0] > threshold)
			{
				binImg.at<cv::Vec3b>(i,j).val[0] = 255;
				binImg.at<cv::Vec3b>(i,j).val[1] = 255;
				binImg.at<cv::Vec3b>(i,j).val[2] = 255;
			}
			else
			{
				binImg.at<cv::Vec3b>(i,j).val[0] = 0;
				binImg.at<cv::Vec3b>(i,j).val[1] = 0;
				binImg.at<cv::Vec3b>(i,j).val[2] = 0;
			}
		}

	return binImg;
}

Mat addImg( Mat highpassImg, Mat img)
{
	for (int i=1; i<=img.rows-2; i++)
		for (int j=1; j<=img.cols-2; j++)
		{
			img.at<cv::Vec3b>(i,j)[0] = img.at<cv::Vec3b>(i,j)[0] +  highpassImg.at<cv::Vec3b>(i,j)[0];
			img.at<cv::Vec3b>(i,j)[1] = img.at<cv::Vec3b>(i,j)[1] +  highpassImg.at<cv::Vec3b>(i,j)[1];
			img.at<cv::Vec3b>(i,j)[2] = img.at<cv::Vec3b>(i,j)[2] +  highpassImg.at<cv::Vec3b>(i,j)[2];

			if(img.at<cv::Vec3b>(i,j)[0]>255)
			{
				img.at<cv::Vec3b>(i,j)[0] = 255;
				img.at<cv::Vec3b>(i,j)[1] = 255;
				img.at<cv::Vec3b>(i,j)[2] = 255;
			}

		}

		return img;
}

Mat toDouble(Mat img)
{
	for (int i=1; i<=img.rows-2; i++)
		for (int j=1; j<=img.cols-2; j++)
		{
			img.at<cv::Vec3b>(i,j)[0] = (double)img.at<cv::Vec3b>(i,j)[0];
			img.at<cv::Vec3b>(i,j)[1] = (double)img.at<cv::Vec3b>(i,j)[1];
			img.at<cv::Vec3b>(i,j)[2] = (double)img.at<cv::Vec3b>(i,j)[2];
		}

		return img;
}

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

//erosion - morphological operation
Mat erosion(Mat img,Mat strel)
{
	Mat tempImg =  Mat(img.rows,img.cols,img.type(),Scalar(0,0,0,0));

	for (int i=0; i<=img.rows-1; i++)
		for (int j=0; j<=img.cols-1; j++)
		{
			int validFlag = true;
			for (int m=0; m<=strel.rows-1; m++)
				for (int n=0; n<=strel.cols-1; n++)
				{
					if((int)img.at<cv::Vec3b>(i+m,j+n)[0] != (int)strel.at<int>(m,n))
					{
						validFlag = false;
					}
				}
				if (validFlag)  
				{
					tempImg.at<Vec3b>(i,j)[0] = 255;
					tempImg.at<Vec3b>(i,j)[1] = 255;
					tempImg.at<Vec3b>(i,j)[2] = 255;
				}
      	}

	return tempImg;
}

//dilation - morphological operation
Mat dilation(Mat img, Mat strel)
{
	Mat outImg = Mat(img.rows,img.cols,img.type(),Scalar(0,0,0,0));
	for (int i=0; i<=img.rows-1; i++)
	{
		for (int j=0; j<=img.cols-1; j++) 
		{
			if ((int)img.at<cv::Vec3b>(i,j)[0] == 255) 
			{     
				for (int m=0; m<=strel.rows-1; m++)
					for (int n=0; n<=strel.cols-1; n++)
					{
          				outImg.at<cv::Vec3b>(i+m, j+n)[0] = (int)strel.at<int>(m,n);
						outImg.at<cv::Vec3b>(i+m, j+n)[1] = (int)strel.at<int>(m,n);
						outImg.at<cv::Vec3b>(i+m, j+n)[2] = (int)strel.at<int>(m,n);
					}
			}
		}
	}
	return outImg;
}

void label(Mat& img,int i,int j,int r,int g,int b)
{
		cv::Vec3b intensity = img.at<cv::Vec3b>(i,j);
		int o_r = intensity.val[0];
		if(i < img.rows  && j < img.cols &&  i >= 0 && j >= 0 && o_r == 255 )
		{
			cv::Vec3b intensity = img.at<cv::Vec3b>(i,j);

			intensity.val[0] = r;
			intensity.val[1] = g;
			intensity.val[2] = b;

			img.at<cv::Vec3b>(i,j) = intensity;
				
			label(img,i+1,j,r,g,b);
			label(img,i,j+1,r,g,b);
			label(img,i,j-1,r,g,b);
			label(img,i-1,j,r,g,b);
		}
}

//counts connected components (4 connected)
Mat fourConn (Mat& img)
{
	int r = 0, g = 100, b = 200;
	int cells=0;
	for(int i =0;i<img.rows ;i++)
		for(int j=0;j<img.cols;j++)
			{
					cv::Vec3b intensity = img.at<cv::Vec3b>(i,j);
					int o_r = intensity.val[0];
					int o_g = intensity.val[1];
					int o_b = intensity.val[2];

				if(o_r == 255)
					{
						label(img,i,j,r,g,b);
						r = r + 100;
						g = g + 70;
						b = b + 30;

						cells++;
					}
			}

	cout<<"Number of tumors found = " << cells;
		
	return img;
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
	Mat kern = (Mat_<int>(3,3) << 0,1,0,1,-4,1,0,1,0);
	Mat highpassImg;
	filter2D(img, highpassImg, img.depth(), kern);
	//highpassImg = filter_custom(img, kern);
	highpassImg = addImg(highpassImg, img);
	imshow("Highpass filtered", highpassImg);
	waitKey(0);


	//kernel = np.ones((5,5),np.float32)/25



	//converting to threshold image
	Mat binImg = convertBinary(highpassImg, 170);
	imshow("Binary Image", binImg);
	waitKey(0);


	// applying OPEN morphological operation
	// part 1 - erosion
	Mat strel = (Mat_<int>(13,13) << 255,255,255,255,255,255,255,255,255,255,255,255,255,
	255,255,255,255,255,255,255,255,255,255,255,255,255,
	255,255,255,255,255,255,255,255,255,255,255,255,255,
	255,255,255,255,255,255,255,255,255,255,255,255,255,
	255,255,255,255,255,255,255,255,255,255,255,255,255,
	255,255,255,255,255,255,255,255,255,255,255,255,255,
	255,255,255,255,255,255,255,255,255,255,255,255,255,
	255,255,255,255,255,255,255,255,255,255,255,255,255,
	255,255,255,255,255,255,255,255,255,255,255,255,255,
	255,255,255,255,255,255,255,255,255,255,255,255,255,
	255,255,255,255,255,255,255,255,255,255,255,255,255,
	255,255,255,255,255,255,255,255,255,255,255,255,255,
	255,255,255,255,255,255,255,255,255,255,255,255,255
	);
	Mat outImg = erosion(binImg,strel);
	imshow("After Erosion",outImg);
	imwrite("Erorded.bmp",outImg);
	waitKey(0);

	outImg = dilation(outImg,strel);
	imshow("After Dilation",outImg);
	imwrite("Dilated.bmp",outImg);
	waitKey(0);

	//counting connected components
	outImg = fourConn(outImg);
	imshow("Component mark and count",outImg);
	imwrite("componentmark.bmp",outImg);
	waitKey(0);



	cout<< "\n\nPress any key to exit...";
	getchar();
}