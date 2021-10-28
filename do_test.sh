#!/bin/sh
curl "http://localhost:8120/detect?input=/srv/linebot/images/upload-2021-10-25-11-02-43.jpg&output=/tmp/a.jpg" -o result.json
