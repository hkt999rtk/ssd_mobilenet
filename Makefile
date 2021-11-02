INS_PATH = /srv/linebot

all:
	@echo "Usage: make [config/start/stop]"

config:
	@echo "Copy ssd_mobilenet_v3.service"
	@sudo cp ./ssd_mobilenet_v3.service /lib/systemd/system
	@echo "Enable ssd_mobilenet_v3 service"
	@sudo systemctl enable ssd_mobilenet_v3.service

install:
	@echo "Installation path" $(INS_PATH)
	@mkdir -p $(INS_PATH)/ssd_mobilenet_v3_large_coco_2020_01_14
	@mkdir -p $(INS_PATH)/ssd_mobilenet_v3_small_coco_2020_01_14
	@mkdir -p $(INS_PATH)/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19
	@mkdir -p $(INS_PATH)/lite-model_efficientdet_lite0_int8
	@mkdir -p $(INS_PATH)/yolov3-416-int8
	@mkdir -p $(INS_PATH)/yolov4-416-fp32
	@mkdir -p $(INS_PATH)/yolov4-416-fp16
	@mkdir -p $(INS_PATH)/yolov3-tiny-416-int8
	@cp build/ssd_mobilenet_detection $(INS_PATH)
	@cp models.ini $(INS_PATH)
	@cp ssd_mobilenet_v3_large_coco_2020_01_14/model.tflite $(INS_PATH)/ssd_mobilenet_v3_large_coco_2020_01_14
	@cp ssd_mobilenet_v3_small_coco_2020_01_14/model.tflite $(INS_PATH)/ssd_mobilenet_v3_small_coco_2020_01_14
	@cp ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/model.tflite $(INS_PATH)/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19
	@cp lite-model_efficientdet_lite0_int8/model.tflite $(INS_PATH)/lite-model_efficientdet_lite0_int8
	@cp yolov3-416-int8/model.tflite $(INS_PATH)/yolov3-416-int8
	@cp yolov4-416-fp32/model.tflite $(INS_PATH)/yolov4-416-fp32
	@cp yolov4-416-fp16/model.tflite $(INS_PATH)/yolov4-416-fp16
	@cp yolov3-tiny-416-int8/model.tflite $(INS_PATH)/yolov3-tiny-416-int8
	@echo "Done"

start:
	systemctl start ssd_mobilenet_v3.service

stop:
	systemctl stop ssd_mobilenet_v3.service

restart:
	systemctl restart ssd_mobilenet_v3.service
	@echo 'NEW PID:' `cat $(INS_PATH)/ssd_mobilenet_v3.pid`

