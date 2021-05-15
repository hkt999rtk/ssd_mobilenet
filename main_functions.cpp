#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include "test-image.h"
#include "nms.h"

extern const char* kCategoryLabels[];
static std::unique_ptr<tflite::Interpreter> interpreter;

extern "C" void setup()
{
	tflite::ErrorReporter* error_reporter = nullptr;
	static tflite::StderrReporter my_error_reporter;
	error_reporter = &my_error_reporter;

	static std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("model/lite-model_ssd_mobilenet_v1_1_metadata_2.tflite");

	if ( !model ) {
		printf("Failed to mmap model\n");
		exit(0);
	}

	static tflite::ops::builtin::BuiltinOpResolver resolver;
	tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	if (allocate_status != kTfLiteOk) {
		TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
		return;
	}
}

extern "C" void loop()
{
	uint8_t *data = interpreter->typed_input_tensor<uint8_t>(0);
	memcpy(data, test_image_data, test_image_data_len);

	interpreter->Invoke();

	NmsPostProcess nms;
	/* 4 output tensors */
	float_t *out0 = interpreter->typed_output_tensor<float_t>(0);
	float_t *out1 = interpreter->typed_output_tensor<float_t>(1);
	float_t *out2 = interpreter->typed_output_tensor<float_t>(2);
	float_t *out3 = interpreter->typed_output_tensor<float_t>(3);
	int numBoxes = (int)*out3;

	for (int i=0; i<numBoxes; i++) {
		int minX = (int)(300.0 * out0[i*4+1]);
		int minY = (int)(300.0 * out0[i*4]);
		int maxX = (int)(300.0 * out0[i*4+3]);
		int maxY = (int)(300.0 * out0[i*4+2]);
		int score = (int)(out2[i] * 100.0);
		int classId = (int) out1[i];

		#define SCORE_THRESHOLD 60
		if (score > SCORE_THRESHOLD) {
			BoundingBox box( minX, minY, maxX, maxY, score, classId);
			nms.AddBoundingBox(box);
		}
	}

	#define OVERLAY_THRESHOLD  50
	nms.Go(OVERLAY_THRESHOLD); /* overlay threshold */
}

