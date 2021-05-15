#ifndef _OPENCV_CAPTURE_RTSP_H_
#define _OPENCV_CAPTURE_RTSP_H_

#include "opencv2/opencv.hpp"

using namespace cv;
/* Support only one instance */
void startCapture( const char *url );
Mat &getImage();
void stopCapture();

#endif
