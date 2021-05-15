#include "main_functions.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() 
{
	setup(); /* setup the object detection library */
	bool m_enable = true;
	VideoCapture capture("rtsp://localhost:1/live");

	if (!capture.isOpened()) {
		printf("Capture open fail !\n");
		return 1;
	}

	//namedWindow("Object Detection Test");

	Mat frame, resized;

	while (m_enable) {
		if (!capture.read(frame)) {
			cout << "Error in reading frame\n";
			break;
		}
		cout << "shape:" << frame.cols << " " << frame.rows << endl;
		resize( frame, resized, cv::Size(640,480), 0, 0, cv::INTER_AREA);
		printf("frame ready\n");
		loop();
		imshow("Object Detection Test", resized);
		if ( waitKey(1) != -1 ) 
			break;
	}
	capture.release();

	return 0;
}
