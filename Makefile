INS_PATH = /srv/linebot

all:
	@echo "Usage: make [config/start/stop]"

config:
	@echo "Copy ssd_mobilenet_v3.service"
	@sudo cp ./ssd_mobilenet_v3.service /lib/systemd/system
	@echo "Enable ssd_mobilenet_v3 service"
	@sudo systemctl enable ssd_mobilenet_v3.service

quick_install:
	@echo "Installation path" $(INS_PATH)
	@cp build/ssd_mobilenet_detection $(INS_PATH)
	@cp models.ini $(INS_PATH)
	@echo "Done"

install:
	@echo "Installation path" $(INS_PATH)
	@mkdir -p $(INS_PATH)/deploy
	@cp build/ssd_mobilenet_detection $(INS_PATH)
	@cp models.ini $(INS_PATH)
	@cp -r deploy/* $(INS_PATH)/deploy
	@echo "Done"

start:
	systemctl start ssd_mobilenet_v3.service

stop:
	systemctl stop ssd_mobilenet_v3.service

restart:
	systemctl restart ssd_mobilenet_v3.service
	@echo 'NEW PID:' `cat $(INS_PATH)/ssd_mobilenet_v3.pid`

