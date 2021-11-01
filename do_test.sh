#!/bin/sh
curl "http://localhost:8120/detect?input=/srv/linebot/images/upload-2021-10-30-23-21-01.jpg&output=/tmp/a.jpg" -o result.json
cp /tmp/a.jpg .
curl "http://localhost:8120/model_list" -o name.json
