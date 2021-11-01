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
		string kTitle;
		string kModelName;
		int kNumCols;
		int kNumRows;
		int kNumChannels;
		int kMaxImageSize;
		int kImageMode;
		int kCategoryCount;
		int kScoreThreshold;
		int kIouThreshold;
		bool kDefault;
		string kPostProcess;
		const char **kCategoryLabels;
		std::unique_ptr<tflite::Interpreter> interpreter;
		tflite::StderrReporter my_error_reporter;
		std::unique_ptr<tflite::FlatBufferModel> model;
		tflite::ops::builtin::BuiltinOpResolver resolver;

	public:
		ODInference(string title, int iWidth, int iHeight, int iNumCh,
			int score, int iou, string modelName,
			int numCategory, const char **labels);
		virtual ~ODInference() {}

		void modelSetup();
		string &getModelName() { return kModelName; }
		inline string getTitle() { return kTitle; }
		string detect(Mat &img, const string &output, int &width, int &height);
		inline int getScore() { return kScoreThreshold; }
		inline int getIou() { return kIouThreshold; }
		inline bool isDefault() { return kDefault; }
		inline void setDefault() { kDefault = true; }
		inline void setPostProcess(string pp) { kPostProcess = pp; }
		virtual string getClassName(int classId);
};

class MobileNetSSD : public ODInference
{
	public:
		MobileNetSSD(string title, int iWidth, int iHeight, int iNumCh,
			int score, int iou, string modelName,
			int numCategory, const char **labels);
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
		if (name=="default") {
			for (int i=0; i<vec.size(); i++) {
				if (vec[i]->isDefault()) {
					return vec[i];
				}
			}
			return vec[0];
		} else {
			for (int i=0; i<vec.size(); i++) {
				if (name == vec[i]->getTitle()) {
					return vec[i];
				}
			}
		}
	}

	return NULL;
}

ODInference::ODInference(string title, int iWidth, int iHeight, int iNumCh,
	int score, int iou, string modelName,
	int numCategory, const char **labels)
{
	kModelName = modelName;
	kTitle = title;
	kNumCols = iWidth;
	kNumRows = iHeight;
	kNumChannels = iNumCh;
	kMaxImageSize = kNumCols * kNumRows * kNumChannels;
	kScoreThreshold = score;
	kIouThreshold = iou;
	kCategoryCount = numCategory;
	kCategoryLabels = labels;
	kDefault = false;
}

string ODInference::getClassName(int classId)
{
	return string(kCategoryLabels[classId]);
}

MobileNetSSD::MobileNetSSD(string title, int iWidth, int iHeight, int iNumCh,
	int score, int iou,
	string modelName, int numCategory, const char **labels):
	ODInference(title, iWidth, iHeight, iNumCh, score, iou, 
		modelName, numCategory, labels)
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

	int numOutputTensor = interpreter->outputs().size();
	if (numOutputTensor != 4) {
		// TODO: error here
	}

	int numBoxes, numClasses;
	NmsPostProcess nms;
	if (kPostProcess == "yolo") {
		for (int i=0; i<numOutputTensor; i++) {
			int outputTensorIndex = interpreter->outputs()[i];
			TfLiteIntArray *outputDims = interpreter->tensor(outputTensorIndex)->dims;
			for (int j=0; j<outputDims->size; j++) {
				if (j == 1) numBoxes = outputDims->data[j];
				if (i == 1 && j == 2) numClasses = outputDims->data[j];
			}
		}
		/* Return a mutable pointer into the data of a given output tensor. */
		float_t *boxes = interpreter->typed_output_tensor<float_t>(0);
		float_t *scores = interpreter->typed_output_tensor<float_t>(1);
		for (int i=0; i<numBoxes; i++) {
			for (int j=0; j<numClasses; j++) {
				int score = (int)(scores[i * numClasses + j] * 100);
				if (score > kScoreThreshold && j == 0) {
					float x = boxes[i*4];
					float y = boxes[i*4+1];
					float w = boxes[i*4+2];
					float h = boxes[i*4+3];
					int minX = (x - w / 2.0);
					int maxX = (x + w / 2.0);
					int minY = (y - h / 2.0);
					int maxY = (y + h / 2.0);
					printf("minX=%d, minY=%d, maxX=%d, maxY=%d, score=%d, classid=%d\n", minX, minY, maxX, maxY, score, j);
					BoundingBox box(minX, minY, maxX, maxY, int(score * 100), j);
					nms.AddBoundingBox(box);
				}
			}
		}
	} else {
		// Google SSD
		float_t *boxes = interpreter->typed_output_tensor<float_t>(0);
		float_t *classIds = interpreter->typed_output_tensor<float_t>(1);
		float_t *scores = interpreter->typed_output_tensor<float_t>(2);
		int numBoxes = (int)(*interpreter->typed_output_tensor<float_t>(3));

		for (int i=0; i<numBoxes; i++) {
			int minX = (int)(x_ratio * kNumCols * boxes[i*4+1]);
			int minY = (int)(y_ratio * kNumRows * boxes[i*4]);
			int maxX = (int)(x_ratio * kNumCols * boxes[i*4+3]);
			int maxY = (int)(y_ratio * kNumRows * boxes[i*4+2]);
			int score = (int)(scores[i] * 100.0);
			int classId = (int) classIds[i];
			if (score > kScoreThreshold ) {
				BoundingBox bb(minX, minY, maxX, maxY, score, classId);
				nms.AddBoundingBox(bb);
			}
		}
	}

	NmsProc nmsCall(&outputImg, x_offset, y_offset, this);
	nms.Go(kIouThreshold, nmsCall); // overlay threshold
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

void _returnValue(string title, int width, int height, int score, int iou,
	string &modelName, string &result, int ms, ostream &os)
{
	os << "{\"status\":\"ok\", \"elapsed_time\":" << ms 
	   << ",\"title\":\"" << title << "\""
	   << ",\"model\":" << "\"" << modelName << "\""
	   << ",\"width\":" << width << ",\"height\":" << height
	   << ",\"score\":" << score << ",\"iou\":" << iou
	   << ",\"detection\":" << result << "}";
}
#define returnValue(title, width, height, score, iou, modelName, result, ms, os, rc) \
	_returnValue(title, width, height, score, iou, modelName, result, ms, os); return rc;

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
				returnValue(infEngine->getTitle(), width, height,
					infEngine->getScore(), infEngine->getIou(),
					infEngine->getModelName(), result, get_current_ticks() - start, os, 0);
			}
			// engine not found, fallback to default
			infEngine = m_im->findOd(string("default"));
			if (infEngine) {
				int width = 0, height = 0;
				string result = infEngine->detect(img, po->firstValue(), width, height);
				returnValue(infEngine->getTitle(), width, height,
					infEngine->getScore(), infEngine->getIou(),
					infEngine->getModelName(), result, get_current_ticks() - start, os, 0);
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

	os << "[";
	for (int i=0; i<m_im->vec.size(); i++) {
		os << "\"" << m_im->vec[i]->getTitle() << "\"";
		if (i<m_im->vec.size()-1) {
			os << ",";
		}
	}
	os << "]";

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
int main(int argc, char **argv)
{
	INIReader reader("models.ini");
	if (reader.ParseError() < 0) {
        clog << "error: can't load 'models.ini'\n";
        return 1;
    }

	HttpServer server(".", 8110);
	InferenceManager im;

	set<string> sections = reader.Sections();
	for (set<string>::iterator it = sections.begin(); it != sections.end(); ++it) {
		int width = reader.GetInteger(*it, "width", -1);
		int height = reader.GetInteger(*it, "height", -1);
		int channels = reader.GetInteger(*it, "channels", -1);
		int score = reader.GetInteger(*it, "score", 50);
		int iou = reader.GetInteger(*it, "iou", 50);
		int isDefault = reader.GetInteger(*it, "default", 0);
		string title = reader.Get(*it, "title", "unknown");
		string post = reader.Get(*it,"post", "ssd");
		if (width < 0 || height < 0 || channels < 0)
			continue;
		MobileNetSSD *ssd = new MobileNetSSD(title, width, height, channels,
			score, iou, *it, COUNT_LABELS, cocoLabels);
		if (isDefault)
			ssd->setDefault();

		for_each(post.begin(), post.end(), [](char & c) {
			c = ::tolower(c);
		});
		ssd->setPostProcess(post);
		im.add(ssd);
		clog << "load model: " << *it;
		if (isDefault) {
			clog << " (default)";
		}
		clog << endl;
	}

    DetectCGI detectCGI(&im);
	NameCGI nameCGI(&im);
    
    server.registerCgi("detect", detectCGI);
	server.registerCgi("model_list", nameCGI);
	server.run(5); // acceptance 5 current request

	return 0;
}

