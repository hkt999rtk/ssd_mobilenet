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
	@cp build/ssd_mobilenet_detection $(INS_PATH)
	@cp ssd_mobilenet_v3_large_coco_2020_01_14/model.tflite $(INS_PATH)/ssd_mobilenet_v3_large_coco_2020_01_14
	@echo "Done"

start:
	systemctl start ssd_mobilenet_v3.service

stop:
	systemctl stop ssd_mobilenet_v3.service

restart:
	systemctl restart ssd_mobilenet_v3.service
	@echo 'NEW PID:' `cat $(INS_PATH)/ssd_mobilenet_v3.pid`

