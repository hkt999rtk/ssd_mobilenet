#include <opencv2/opencv.hpp>
#include <cstdio>
#include <unistd.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include "httpserver.h"
#include "nms.h"
#include "util.h"

using namespace std;
using namespace cv;

constexpr int kNumCols = 320;
constexpr int kNumRows = 320;
constexpr int kNumChannels = 3;
constexpr int kMaxImageSize = kNumCols * kNumRows * kNumChannels;
constexpr int kCategoryCount = 90;
extern const char* kCategoryLabels[kCategoryCount];

const char* kCategoryLabels[kCategoryCount] = {
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", " boat", "traffic light",
  "fire hydrant", "???", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
  "cow", "elephant", "bear", "zebra", "giraffe", "???", "backpack", "umbrella", "???", "???",
  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
  "skateboard", "surfboard", "tennis racket", "bottle", "???", "wine glass", "cup", "fork", "knife", "spoon",
  "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
  "cake", "chair", "couch", "potted plant", "bed", "???", "dining table", "???", "???", "toilet",
  "???", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
  "sink", "refrigerator", "???", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};


static std::unique_ptr<tflite::Interpreter> interpreter;
void ssd_mobilenet_setup()
{
	tflite::ErrorReporter* error_reporter = nullptr;
	static tflite::StderrReporter my_error_reporter;
	error_reporter = &my_error_reporter;
	static std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("ssd_mobilenet_v3_large_coco_2020_01_14/model.tflite");

	if ( !model ) {
		clog << "failed to mmap model" << endl;
		exit(0);
	}

	static tflite::ops::builtin::BuiltinOpResolver resolver;
	tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	if (allocate_status != kTfLiteOk) {
		TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
		exit(0);
	}
}

class NmsProc : public NmsCb
{
	protected:
		Mat *pImage;

	public:
		NmsProc(Mat *img) { pImage = img; }
		virtual ~NmsProc() {}

	public:
		virtual int callback(BoundingBox &boundingBox);
};

int NmsProc::callback(BoundingBox &bb)
{
	char s[128];
	Rect rect(bb.minX, bb.minY, bb.maxX-bb.minX+1, bb.maxY-bb.minY+1);
	rectangle(*pImage, rect, (180,105,255), 6);
	sprintf(s, "%s (%d%%)", kCategoryLabels[bb.classId], bb.score);
	putText(*pImage, s, Point(bb.minX, bb.minY-5), FONT_HERSHEY_SIMPLEX, 3, (180,105,255), 1.5);
	clog << "callback minX=" << bb.minX << ", minY=" << bb.minY << ", maxX=" << bb.maxX 
		<< ", maxY=" << bb.maxY << ", score=" << bb.score << ", class=" << bb.classId <<
		", name=" << kCategoryLabels[bb.classId] << endl;

	return 0;
}

void ssd_mobilenet_detect(Mat &img, const string &output)
{
	Mat dstImg;

	float x_ratio = (float)img.cols / (float)kNumCols;
	float y_ratio = (float)img.rows / (float)kNumRows;
    resize(img, dstImg, Size(kNumCols, kNumRows));
    cvtColor(dstImg, dstImg, COLOR_BGR2RGB);

	uint8_t *data = interpreter->typed_input_tensor<uint8_t>(0);
	memcpy(data, dstImg.data, kMaxImageSize);

	interpreter->Invoke();

	NmsPostProcess nms;
	/* 4 output tensors */
	float_t *out0 = interpreter->typed_output_tensor<float_t>(0);
	float_t *out1 = interpreter->typed_output_tensor<float_t>(1);
	float_t *out2 = interpreter->typed_output_tensor<float_t>(2);
	float_t *out3 = interpreter->typed_output_tensor<float_t>(3);
	int numBoxes = (int)*out3;

	for (int i=0; i<numBoxes; i++) {
		int minX = (int)(x_ratio * kNumCols * out0[i*4+1]);
		int minY = (int)(y_ratio * kNumRows * out0[i*4]);
		int maxX = (int)(x_ratio * kNumCols * out0[i*4+3]);
		int maxY = (int)(y_ratio * kNumRows * out0[i*4+2]);
		int score = (int)(out2[i] * 100.0);
		int classId = (int) out1[i];
		if (score > 50) {
			BoundingBox bb(minX, minY, maxX, maxY, score, classId);
			nms.AddBoundingBox(bb);
		}
	}
	NmsProc nmsCall(&img);
	nms.Go(50, nmsCall); // overlay threshold

	auto rc = imwrite(output, img);
	clog << "write image to " << output << " rc=" << rc << endl;
}

class MyCgiTest : public MyFastCgi
{
    public:
        int run(QueryString &qs, ostream &os);
};

void returnValue(int ms, ostream &os)
{
	os << "{\"status\":\"ok\", \"elapsed_time\":" << ms << "}";
}

void returnFail(const char *reason, ostream &os)
{
	clog << reason << endl;
	os << "{\"status\":\"fail\", \"reason\":\"" << reason << "\"}";
}

int MyCgiTest::run(QueryString &qs, ostream &os)
{
    os << "Content-Type: application/json" << endl << endl;
	if (qs.numParams()>=1) {
		if (qs.hasParam("input") && qs.hasParam("output")) {
			auto pi = qs.getParam("input");
			Mat img = imread(pi->firstValue());
			clog << "input:" << pi->firstValue() << endl;
			if (!img.data) {
				returnFail("error: canot read file", os);
				return -1;
			}
			auto po = qs.getParam("output");
			int start = get_current_ticks();
   			ssd_mobilenet_detect(img, po->firstValue());
			returnValue( get_current_ticks() - start, os );
		} else {
			returnFail("error: need input and output parameter", os);
			return -1;
		}
	} else {
		returnFail("error: no parameter", os);
		return -1;
	}

    return 0;
}


int main(int argc, char **argv)
{
	ssd_mobilenet_setup();

	HttpServer server(".");
    MyCgiTest mycgi;
    
    server.registerMyCgi("detect", mycgi);
	server.run(2);

	return 0;
}

