# SSD MobileNet Detection Service

這個專案提供一個以 C++ 實作的物件偵測服務，使用 TensorFlow Lite 模型進行推論，並透過 HTTP CGI 介面提供偵測 API。

目前程式啟動後會提供：

- `GET /model_list`：列出可用模型（`models.ini` 內已啟用的 section）
- `GET /detect`：針對輸入圖片做偵測並輸出結果 JSON，且可同時輸出標註後圖片

## 專案結構

- `main.cpp`：HTTP 服務入口、模型載入、推論流程
- `models.ini`：模型清單與推論參數設定
- `deploy/`：模型資料夾（每個模型一個子資料夾，內含 `model.tflite`）
- `libs/`：靜態函式庫（TensorFlow Lite 與相依套件）
- `ssd_mobilenet_v3.service`：systemd 服務檔
- `Makefile`：部署與服務控制指令

## 環境需求

- CMake `>= 3.12`
- C++20 編譯器（例如 `g++`）
- OpenCV
- Boost
- cgicc
- pthread / dl
- TensorFlow Lite 相關靜態庫（放在 `libs/`）

## 建置

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j
```

建置完成後執行檔為：

- `build/ssd_mobilenet_detection`

## 啟動方式

正常模式（預設埠 `8110`）：

```bash
./build/ssd_mobilenet_detection
```

除錯模式（埠 `8120`）：

```bash
./build/ssd_mobilenet_detection debug
```

## API 使用方式

### 1) 列出模型

```bash
curl "http://127.0.0.1:8110/model_list"
```

### 2) 進行偵測

必要參數：

- `input`：輸入圖片路徑
- `output`：輸出標註圖路徑

可選參數：

- `model`：模型名稱（對應 `models.ini` section 名稱；未提供時會走 `default`）
- `x_left` / `x_right` / `y_top` / `y_bottom`：裁切比例（`0.0 ~ 1.0`）

範例：

```bash
curl "http://127.0.0.1:8110/detect?input=/tmp/in.jpg&output=/tmp/out.jpg&model=yolov4-416-fp32&x_left=0.05&x_right=0.05" -o result.json
```

回傳 JSON 範例欄位：

- `status`
- `ms`（推論耗時）
- `title`
- `model`
- `width` / `height`
- `score` / `iou`
- `detection`

## models.ini 說明

每個 section 代表一個模型，例如：

- `[yolov4-416-fp32]`

常用欄位：

- `title`：模型顯示名稱
- `width` / `height` / `channels`：輸入尺寸與通道數
- `score`：分數門檻（百分比）
- `iou`：NMS IoU 門檻（百分比）
- `post`：後處理類型（`ssd` 或 `yolo`）
- `default=1`：標記為預設模型（可多個，服務會輪詢）
- `classes`：類別清單（逗號分隔）

模型檔預設路徑規則：

- `deploy/<section_name>/model.tflite`

## systemd 部署（選用）

`Makefile` 內提供快速部署與服務控制：

```bash
make config       # 安裝並啟用 systemd service
make quick_install
make install
make start
make stop
make restart
```

預設部署路徑為 `/srv/linebot`（可在 `Makefile` 的 `INS_PATH` 修改）。

## Git LFS

本專案的 `.tflite` 模型檔已改用 Git LFS 管理。首次 clone 後建議執行：

```bash
git lfs install
git lfs pull
```
