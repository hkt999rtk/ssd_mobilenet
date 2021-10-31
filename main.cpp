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
#include "INIReader.h"

#define INPUT_ORIGINAL	0
#define INPUT_CROP		1
#define INPUT_PADDING	2

using namespace std;
using namespace cv;

class ODInference
{
	protected:
		string kModelName;
		int kNumCols;
		int kNumRows;
		int kNumChannels;
		int kMaxImageSize;
		int kImageMode;
		int kCategoryCount;
		const char **kCategoryLabels;
		std::unique_ptr<tflite::Interpreter> interpreter;
		tflite::StderrReporter my_error_reporter;
		std::unique_ptr<tflite::FlatBufferModel> model;
		tflite::ops::builtin::BuiltinOpResolver resolver;

	public:
		ODInference(int iWidth, int iHeight, int iNumCh, string modelName, int numCategory, const char **labels);
		virtual ~ODInference() {}

		void modelSetup();
		string &getModelName() { return kModelName; }
		string detect(Mat &img, const string &output, int &width, int &height);
		virtual string getClassName(int classId);
};

class MobileNetSSD : public ODInference
{
	public:
		MobileNetSSD(int iWidth, int iHeight, int iNumCh, string modelName, int numCategory, const char **labels);
		~MobileNetSSD() {}
};

class InferenceManager
{
	public:
		vector<ODInference *> vec;

	public:
		InferenceManager();
		virtual ~InferenceManager();

	public:
		void add(ODInference *od);
		ODInference *findOd(string name);
		ODInference *getDefault();
};

InferenceManager::InferenceManager()
{
}

InferenceManager::~InferenceManager()
{
}

void InferenceManager::add(ODInference *od)
{
	vec.push_back(od);
}

ODInference *InferenceManager::findOd(string name)
{
	if (vec.size()>=1) {
		if (name=="default")
			return vec[0];

		for (int i=0; i<vec.size(); i++) {
			if (name == vec[i]->getModelName()) {
				return vec[i];
			}
		}
	}

	return NULL;
}

ODInference::ODInference(int iWidth, int iHeight, int iNumCh, string modelName, int numCategory, const char **labels)
{
	kModelName = modelName;
	kNumCols = iWidth;
	kNumRows = iHeight;
	kNumChannels = iNumCh;
	kMaxImageSize = kNumCols * kNumRows * kNumChannels;
	kCategoryCount = numCategory;
	kCategoryLabels = labels;
}

string ODInference::getClassName(int classId)
{
	return string(kCategoryLabels[classId]);
}

MobileNetSSD::MobileNetSSD(int iWidth, int iHeight, int iNumCh, string modelName, int numCategory, const char **labels):
	ODInference(iWidth, iHeight, iNumCh, modelName, numCategory, labels)
{
	kImageMode = INPUT_PADDING;
	modelSetup();
}

void ODInference::modelSetup()
{
	tflite::ErrorReporter* error_reporter = nullptr;
	error_reporter = &my_error_reporter;
	string filename = kModelName + "/model.tflite";
	model = tflite::FlatBufferModel::BuildFromFile(filename.c_str());

	if ( !model ) {
		clog << "failed to mmap model" << endl;
		exit(0);
	}

	tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	if (allocate_status != kTfLiteOk) {
		TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
		exit(0);
	}
}

#define MAX_NUM_BOX 100
class NmsProc : public NmsCb
{
	protected:
		Mat *pImage;
		ODInference *infEngine;
		int numBox;
		int x_offset;
		int y_offset;
		vector<BoundingBox> bboxVec;

	public:
		NmsProc(Mat *img, int x_off, int y_off, ODInference *inference) {
			pImage = img;
			numBox = 0;
			x_offset = x_off;
			y_offset = y_off; 
			infEngine = inference;
		}
		virtual ~NmsProc() {}

	public:
		virtual int callback(BoundingBox &boundingBox);
		void addBox(BoundingBox bbox);
		string packJson();
};

void NmsProc::addBox(BoundingBox bbox)
{
	bboxVec.push_back(static_cast<BoundingBox>(bbox));
}

int NmsProc::callback(BoundingBox &bb)
{
	Rect rect(bb.minX-x_offset, bb.minY-y_offset, bb.maxX-bb.minX+1, bb.maxY-bb.minY+1);
	rectangle(*pImage, rect, Scalar(180,105,255), 3);

#if 0
	char s[128];
	snprintf(s, sizeof(s), "%s (%d%%)", kCategoryLabels[bb.classId], bb.score);
	putText(*pImage, s, Point(bb.minX, bb.minY-5),
		FONT_HERSHEY_SIMPLEX, 1, Scalar(180,105,255), 1.5);
#endif

	addBox(bb);
	return 0;
}

string NmsProc::packJson()
{
	ostringstream s;
	s << "[";
	for (int i=0; i<bboxVec.size(); i++) {
		BoundingBox bb = bboxVec[i];
		s << "{\"minx\":" << bb.minX << ", \"maxx\":" << bb.maxX <<
		     ", \"miny\":" << bb.minY << ", \"maxy\":" << bb.maxY <<
			 ", \"score\":" << bb.score << 
			 ", \"id\":" << bb.classId <<
			 ", \"class\":\"" << infEngine->getClassName(bb.classId) << "\"}";
		if (i<bboxVec.size()-1) {
			s << ",";
		}
	}
	s << "]";

	return s.str();
}

string ODInference::detect(Mat &img, const string &output,
	int &width, int &height)
{
	Mat dstImg, outputImg;
	float x_ratio, y_ratio;
	int x_offset = 0, y_offset = 0;

	switch (kImageMode) {
		case INPUT_ORIGINAL:
			resize(img, dstImg, Size(kNumCols, kNumRows));
			x_ratio = (float)img.cols / (float)kNumCols;
			y_ratio = (float)img.rows / (float)kNumRows;
			break;

		case INPUT_CROP: {
				Rect cropRect;
				if (img.cols >= img.rows) {
					cropRect = Rect((img.cols - img.rows)/2, 0, img.rows, img.rows);
				} else {
					cropRect = Rect(0, (img.rows - img.cols)/2, img.cols, img.cols);
				}

				outputImg = Mat(img, cropRect);
				x_ratio = (float)outputImg.cols / (float)kNumCols;
				y_ratio = (float)outputImg.rows / (float)kNumRows;
				resize(outputImg, dstImg, Size(kNumCols, kNumRows));
			}
			break;

		case INPUT_PADDING: {
				Mat dst;
				if (img.cols >= img.rows) {
					dst = Mat(img.cols,  img.cols, CV_8UC3, Scalar(0,0,0));
					y_offset = (img.cols - img.rows)/2;
					img.copyTo(dst.rowRange(y_offset, img.rows+y_offset).colRange(0, img.cols));
				} else {
					dst = Mat(img.rows,  img.rows, CV_8UC3, Scalar(0,0,0));
					x_offset = (img.rows - img.cols)/2;
					img.copyTo(dst.rowRange(0, img.rows).colRange(x_offset, img.cols+x_offset));
				}
				x_ratio = (float)dst.cols / (float)kNumCols;
				y_ratio = (float)dst.rows / (float)kNumRows;
				resize(dst, dstImg, Size(kNumCols, kNumRows));
				outputImg = img;
			}
			break;
	}
    cvtColor(dstImg, dstImg, COLOR_BGR2RGB);

	int8_t *int8_data = 0;
	uint8_t *uint8_data = 0;
	float_t *float_data = 0;
	uint8_data = interpreter->typed_input_tensor<uint8_t>(0);
	int8_data = interpreter->typed_input_tensor<int8_t>(0);
	float_data = interpreter->typed_input_tensor<float_t>(0);
	uint8_t *in = dstImg.data;

	printf("uint8_data=%p, int8_data=%p, float_data=%p\n", uint8_data, int8_data, float_data);
	if (uint8_data) {
		memcpy(uint8_data, in, kMaxImageSize);
	} else if (int8_data) {
		int count = kMaxImageSize;
		while (count-->0) {
			*int8_data++ = (int8_t)(((int)(*in++))-128);
		}
	} else if (float_data) {
		int count = kMaxImageSize;
		while (count-->0) {
			*float_data++ = ((float)(*in++)) / 256.0;
		}
	}

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
	NmsProc nmsCall(&outputImg, x_offset, y_offset, this);
	nms.Go(50, nmsCall); // overlay threshold
	string json = nmsCall.packJson();

	auto rc = imwrite(output, outputImg);
	clog << "write image to " << output << endl;
	width = outputImg.cols;
	height = outputImg.rows;

	return json;
}

class ODCGI : public MyFastCgi
{
	protected:
		pthread_mutex_t m_mutex;
		InferenceManager *m_im;

	public:
		ODCGI(InferenceManager *im) {
			m_im = im;
			pthread_mutex_init(&m_mutex, NULL);
		}
		virtual ~ODCGI() {
			pthread_mutex_destroy(&m_mutex);
		}
};

class DetectCGI : public ODCGI
{
	public:
		DetectCGI(InferenceManager *im);

    public:
        int run(QueryString &qs, ostream &os);
};

DetectCGI::DetectCGI(InferenceManager *im):ODCGI(im)
{
}

class NameCGI : public ODCGI
{
	public:
		NameCGI(InferenceManager *m_im):ODCGI(m_im) {}

	public:
		int run(QueryString &qs, ostream &os);
};

void _returnValue(int width, int height, string &modelName, string &result, int ms, ostream &os)
{
	os << "{\"status\":\"ok\", \"elapsed_time\":" << ms 
	   << ",\"model\":" << "\"" << modelName << "\""
	   << ",\"width\":" << width << ",\"height\":" << height
	   << ",\"detection\":" << result << "}";
}
#define returnValue(width, height, modelName, result, ms, os, rc) \
	_returnValue(width, height, modelName, result, ms, os); return rc;

void _returnFail(const char *reason, ostream &os)
{
	clog << reason << endl;
	os << "{\"status\":\"fail\", \"reason\":\"" << reason << "\"}";
}

#define returnFail(reason, os, rc) _returnFail(reason, os); return rc;

int DetectCGI::run(QueryString &qs, ostream &os)
{
    os << "Content-Type: application/json" << endl << endl;
	if (qs.numParams()>=1) {
		if (qs.hasParam("input") && qs.hasParam("output")) {
			auto pi = qs.getParam("input");
			Mat img = imread(pi->firstValue());
			clog << "input:" << pi->firstValue() << endl;
			if (!img.data) {
				returnFail("error: canot read file", os, -1);
			}
			auto po = qs.getParam("output");
			int start = get_current_ticks();
			string modelName = "default";
			if (qs.hasParam("model")) {
				auto pi = qs.getParam("model");
				modelName = pi->firstValue();
			}
			ODInference *infEngine = m_im->findOd(modelName);
			if (infEngine) {
				int width = 0, height = 0;
				string result = infEngine->detect(img, po->firstValue(), width, height);
				returnValue(width, height, infEngine->getModelName(),
					result, get_current_ticks() - start, os, 0);
			}
			returnFail("inference engine not found", os, -1);
		} else {
			returnFail("need input and output parameter", os, -1);
		}
	} else {
		returnFail("error: no parameter", os, -1);
		return -1;
	}

    return 0;
}

int NameCGI::run(QueryString &qs, ostream &os)
{
    os << "Content-Type: application/json" << endl << endl;

	os << "{[";
	for (int i=0; i<m_im->vec.size(); i++) {
		m_im->vec[i]->getModelName();
		os << "\"" << m_im->vec[i]->getModelName() << "\"";
		if (i<m_im->vec.size()-1) {
			os << ",";
		}
	}
	os << "]}";

	return 0;
}


static const char *cocoLabels[] = {
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", " boat", "traffic light",
	"fire hydrant", "???", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
	"cow", "elephant", "bear", "zebra", "giraffe", "???", "backpack", "umbrella", "???", "???",
	"handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "???", "wine glass", "cup", "fork", "knife", "spoon",
	"bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
	"cake", "chair", "couch", "potted plant", "bed", "???", "dining table", "???", "???", "toilet",
	"???", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
	"sink", "refrigerator", "???", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

#define COUNT_LABELS	(sizeof(cocoLabels)/sizeof(char *))
string sections(INIReader &reader)
{
    stringstream ss;
    set<string> sections = reader.Sections();
    for (set<string>::iterator it = sections.begin(); it != sections.end(); ++it)
        ss << "    " << *it << endl;

    return ss.str();
}

int main(int argc, char **argv)
{
	INIReader reader("models.ini");
	if (reader.ParseError() < 0) {
        clog << "error: can't load 'models.ini'\n";
        return 1;
    }
	clog << "config loaded from 'models.ini', models:" << endl << sections(reader);

	HttpServer server(".", 8120);
	InferenceManager im;

	set<string> sections = reader.Sections();
	for (set<string>::iterator it = sections.begin(); it != sections.end(); ++it) {
		int width = reader.GetInteger(*it, "width", -1);
		int height = reader.GetInteger(*it, "height", -1);
		int channels = reader.GetInteger(*it, "channels", -1);
		if (width < 0 || height < 0 || channels < 0)
			continue;
		MobileNetSSD *ssd = new MobileNetSSD(width, height, channels, *it, COUNT_LABELS, cocoLabels);
		im.add(ssd);
	}

    DetectCGI detectCGI(&im);
	NameCGI nameCGI(&im);
    
    server.registerCgi("detect", detectCGI);
	server.registerCgi("model_list", nameCGI);
	server.run(5); // acceptance 5 current request

	return 0;
}

