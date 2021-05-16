#include <opencv2/opencv.hpp>
#include <unistd.h>
#include "nms.h"
#include "main_functions.h"
#include "capture.h"
#include "model_settings.h"

using namespace std;
using namespace cv;

class DetectionCb : public NmsCb
{
	private:
		Mat *frame;

	public:
		DetectionCb():NmsCb() {}
		virtual ~DetectionCb() {}

	public:
		inline void setFrame( Mat &input_frame ) { frame = &input_frame; }
		int callback(BoundingBox &boundingBox);
};

int DetectionCb::callback( BoundingBox &b )
{
    printf("class=%d (%s), score=%d%%, box=(%d,%d)-(%d,%d)\n",
        b.classId, kCategoryLabels[b.classId], b.score, b.minX, b.minY, b.maxX, b.maxY);

	/* draw the class bounding box */
	rectangle( *frame, Rect(Point(b.minX, b.minY), Point(b.maxX, b.maxY)), Scalar(0, 0, 255), 1);

    putText(*frame, kCategoryLabels[b.classId], Point(b.minX, b.minY), FONT_HERSHEY_PLAIN, 0.8, Scalar(255, 0, 0), 1, CV_MSA);

	return 0;
}


extern  int get_current_ticks();
int main() 
{
	ssd_mobilenet_setup();
	startCapture("http://localhost:56000/mjpeg");
	DetectionCb detCb;

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
			resize(frame, resizedImage, Size(kNumCols, kNumRows));	

			int tick = get_current_ticks();
			detCb.setFrame(resizedImage);
			ssd_mobilenet_detect(resizedImage.data, detCb);
			cout << get_current_ticks() - tick << " ms" << endl;

			imshow("SSD mobilenet", resizedImage);
			if (waitKey(1) != -1)
				break;
		} else {
			cout << "waiting for capture ready..." << endl;
			usleep(500000);
		}
	}
	stopCapture();
	cout << "Finished..." << endl;

	return 0;
}
