#!/bin/sh
#curl "http://localhost:8110/detect?input=/srv/linebot/images/upload-2021-11-01-11-34-40.jpg&output=/tmp/a.jpg" -o result.json
curl "http://localhost:8120/detect?input=/srv/linebot/images/upload-2021-11-01-11-34-40.jpg&output=/tmp/a.jpg&model=yolov3-416-int8" -o result.json
cp /tmp/a.jpg .
curl "http://localhost:8120/model_list" -o name.json
