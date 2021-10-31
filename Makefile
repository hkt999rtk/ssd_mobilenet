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
	@cp build/ssd_mobilenet_detection $(INS_PATH)
	@cp models.ini $(INS_PATH)
	@cp ssd_mobilenet_v3_large_coco_2020_01_14/model.tflite $(INS_PATH)/ssd_mobilenet_v3_large_coco_2020_01_14
	@cp ssd_mobilenet_v3_small_coco_2020_01_14/model.tflite $(INS_PATH)/ssd_mobilenet_v3_small_coco_2020_01_14
	@cp ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/model.tflite $(INS_PATH)/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19
	@echo "Done"

start:
	systemctl start ssd_mobilenet_v3.service

stop:
	systemctl stop ssd_mobilenet_v3.service

restart:
	systemctl restart ssd_mobilenet_v3.service
	@echo 'NEW PID:' `cat $(INS_PATH)/ssd_mobilenet_v3.pid`

