#include <opencv2/opencv.hpp>
#include <unistd.h>
#include "main_functions.h"
#include "capture.h"

using namespace std;
using namespace cv;

int main() 
{
	setup(); 
	startCapture("http://localhost:56000/mjpeg");

	while (true) {
		Mat frame = getImage();
		if (frame.cols >0 && frame.rows > 0) {
			Rect roi;
			if (frame.cols > frame.rows) {
				roi.x = (frame.cols - frame.rows)/2;
				roi.y = 0;
				roi.width = frame.rows;
				roi.height = frame.rows;
			} else {
				roi.x = 0;
				roi.y = (frame.rows - frame.cols)/2;
				roi.width = frame.cols;
				roi.height = frame.cols;
			}
			Mat crop = frame(roi);
			Mat resizedImage;
			resize(frame, resizedImage, Size(300, 300));	
			ssd_mobilenet(resizedImage.data);
			imshow("SSD mobilenet", resizedImage);
			if (waitKey(1) != -1)
				break;
		} else {
			usleep(100000);
		}
	}
	stopCapture();
	cout << "Finished..." << endl;

	return 0;
}
