#!/bin/sh
curl "http://localhost:8120/detect?input=/srv/linebot/images/upload-2021-10-27-11-36-12.jpg&output=/tmp/a.jpg" -o result.json
cp /tmp/a.jpg .
curl "http://localhost:8120/model_list" -o name.json
