import os
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
from torchvision import transforms, models
from torchvision.transforms.functional import crop
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import glob
import numpy as np
import requests
import io
import base64
import json

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# 新增 Google Cloud Storage 相關導入
from google.cloud import storage

# 在所有其他程式碼之前載入 .env 檔案 (本地開發用)
load_dotenv()

# 從 .env 讀取 Google Maps API 金鑰
Maps_API_KEY = os.getenv('API_KEY')
if os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON_BASE64'):
    try:
        cred_json_str = base64.b64decode(os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON_BASE64')).decode('utf-8')
        temp_cred_file = "/tmp/google_credentials.json"
        with open(temp_cred_file, "w") as f:
            f.write(cred_json_str)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_cred_file
        print("已從環境變數載入 Google 服務帳戶憑證。")
    except Exception as e:
        print(f"載入 Google 服務帳戶憑證時發生錯誤: {e}")
        
if not Maps_API_KEY:
    print("警告：未找到 'API_KEY' 環境變數。Google Maps 功能可能無法正常運作。")

# --- 設備設定 (本地電腦中優先使用 CUDA，否則使用 CPU) ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"正在使用 NVIDIA (CUDA) GPU 進行 PyTorch 運算: {torch.cuda.get_device(0)}")
else:
    device = torch.device("cpu")
    print("正在使用 CPU 進行 PyTorch 運算。")

# --- GCS 配置 ---
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME') # 從環境變數獲取儲存桶名稱
GCS_MODELS_PREFIX = 'models/' # GCS 儲存桶中模型檔案的路徑前綴 (例如：gs://your-bucket/models/trained_model_safe.pth)

# 在容器內儲存模型的臨時目錄
# Cloud Run 提供 /tmp 作為可寫目錄
LOCAL_MODELS_DIR = '/tmp/models'
os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)


def download_models_from_gcs():
    """
    從 Google Cloud Storage 下載所有預期的模型檔案到本地臨時目錄。
    """
    if not GCS_BUCKET_NAME:
        print("警告：未設定 'GCS_BUCKET_NAME' 環境變數，無法從 GCS 下載模型。")
        return

    print(f"\n--- 正在從 GCS 儲存桶 '{GCS_BUCKET_NAME}' 下載模型到 '{LOCAL_MODELS_DIR}' ---")
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)

    # 定義您預期的模型檔案名稱
    model_filenames = [f'trained_model_{study_type}.pth' for study_type in ['safe', 'lively', 'clean']]

    for filename in model_filenames:
        gcs_blob_name = f"{GCS_MODELS_PREFIX}{filename}" # 完整的 GCS blob 路徑
        local_file_path = os.path.join(LOCAL_MODELS_DIR, filename)

        try:
            blob = bucket.blob(gcs_blob_name)
            if blob.exists():
                blob.download_to_filename(local_file_path)
                print(f"成功下載：{gcs_blob_name} -> {local_file_path}")
            else:
                print(f"警告：GCS 中找不到 Blob '{gcs_blob_name}'。")
        except Exception as e:
            print(f"下載 '{gcs_blob_name}' 時發生錯誤: {e}")

# --- 圖片轉換函式 (保持不變) ---
def crop_google_logo(img):
    """裁剪 Google 地圖照片底部可能有的 Google Logo。"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    if img.size[1] > 25:
        return crop(img, 0, 0, img.size[1] - 25, img.size[0])
    return img

# ==============================================================================
# CustomImageDatasetFromBytes 類別 (保持不變)
# ==============================================================================
class CustomImageDatasetFromBytes(Dataset):
    """
    用於載入從 URL 或字節流獲取的圖片的 PyTorch 資料集類別。
    圖片將在內部處理為 PIL Image，然後應用轉換。
    """
    def __init__(self, image_data_list):
        self.image_data_list = image_data_list
        self._base_transform = transforms.Compose([
            transforms.Lambda(crop_google_logo),
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_data_list)

    def __getitem__(self, idx):
        original_url, img_bytes = self.image_data_list[idx]

        try:
            # 從 bytes 載入圖片作為 PIL Image
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print(f"警告: 無法打開圖片 bytes 數據，URL: {original_url}, 錯誤: {e}")
            # 返回一個空的或預設的張量，這樣 DataLoader 不會崩潰
            return torch.zeros(3, 224, 224), original_url

        # 應用預處理轉換
        if self._base_transform:
            image = self._base_transform(image)

        return image, original_url


# ==============================================================================
# 模型初始化與載入 (修改為從 LOCAL_MODELS_DIR 載入)
# ==============================================================================
def initialize_and_load_models():
    """
    初始化 VGG16 模型結構並嘗試載入預訓練權重。
    現在從 LOCAL_MODELS_DIR (即 /tmp/models) 載入。
    """
    print("\n--- 1. 初始化模型結構並準備載入權重 ---")

    study_types_to_load = ['safe', 'lively', 'clean']

    trained_models = {}

    for study_type in study_types_to_load:
        # 從容器的臨時目錄載入模型
        model_path = os.path.join(LOCAL_MODELS_DIR, f'trained_model_{study_type}.pth')

        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 1)

        try:
            # 在 Cloud Run 上，通常使用 map_location=device (CPU)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            trained_models[study_type] = model
            print(f"已成功載入 '{study_type}' 模型。")
        except FileNotFoundError:
            print(f"警告：找不到 '{study_type}' 屬性的模型檔案，路徑為 '{model_path}'。該屬性將不會被用於預測。")
        except Exception as e:
            print(f"載入 '{study_type}' 模型時發生錯誤: {e}。該屬性將不會被用於預測。")

    if not trained_models:
        print("沒有找到任何可用的模型進行預測。請確認模型文件已存在於 GCS 儲存桶中並已成功下載到容器。")
    return trained_models


# ==============================================================================
# 模型預測函數 (保持不變)
# ==============================================================================
def predict_scores_from_image_urls(image_urls, trained_models):
    """
    接收圖片 URL 列表，下載圖片，並使用所有訓練好的模型進行預測。
    返回一個包含每個圖片預測結果的列表。
    """
    if not trained_models:
        print("沒有可用的模型進行預測。")
        return []

    study_types_in_order = ['safe', 'lively', 'clean']
    all_predictions = []
    images_to_process = []

    print(f"開始下載 {len(image_urls)} 張圖片以進行預測...")
    for url in tqdm(image_urls, desc="下載圖片"):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            images_to_process.append((url, response.content))
        except requests.exceptions.RequestException as e:
            print(f"下載圖片 '{url}' 時發生錯誤: {e}")
            all_predictions.append({'image_url': url, 'error': f"下載失敗: {e}"})
        except Exception as e:
            print(f"處理圖片 '{url}' 下載時的未知錯誤: {e}")
            all_predictions.append({'image_url': url, 'error': f"下載時發生未知錯誤: {e}"})

    if not images_to_process:
        print("沒有成功下載的圖片可供預測。")
        return all_predictions

    custom_dataset = CustomImageDatasetFromBytes(image_data_list=images_to_process)
    # Cloud Run 環境下 num_workers 建議設定為 0
    custom_loader = DataLoader(custom_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"將預測 {len(images_to_process)} 張圖片的屬性分數。")
    with torch.no_grad():
        for images_tensor, original_urls in tqdm(custom_loader, desc="預測中"):
            if images_tensor.sum() == 0 and images_tensor.shape == torch.Size([1, 3, 224, 224]):
                all_predictions.append({'image_url': original_urls[0], 'error': '圖片載入或處理失敗 (可能為空或格式錯誤)'})
                continue

            image_predictions = {'image_url': original_urls[0]}
            images_tensor = images_tensor.to(device)

            for study_type in study_types_in_order:
                model = trained_models.get(study_type)
                if model:
                    try:
                        outputs = model(images_tensor)
                        predicted_score = outputs.item()
                        image_predictions[f'predicted_score_{study_type}'] = predicted_score
                    except Exception as e:
                        print(f"對圖片 '{original_urls[0]}' 預測屬性 '{study_type}' 時發生錯誤: {e}")
                        image_predictions[f'predicted_score_{study_type}'] = 'Error'
                else:
                    image_predictions[f'predicted_score_{study_type}'] = None
            all_predictions.append(image_predictions)

    print("\n--- 預測完成！ ---")
    return all_predictions


# ==============================================================================
# Flask API 設定 (保持不變)
# ==============================================================================

app = Flask(__name__)
CORS(app)

GLOBAL_LOADED_MODELS = {}

@app.route('/get_maps_api_key', methods=['GET'])
def get_maps_api_key():
    return jsonify({"apiKey": Maps_API_KEY})

@app.route('/predict_street_scores', methods=['POST'])
def handle_predict_street_scores():
    global GLOBAL_LOADED_MODELS

    if not GLOBAL_LOADED_MODELS:
        print("警告：模型尚未載入。嘗試重新載入模型。")
        GLOBAL_LOADED_MODELS = initialize_and_load_models()
        if not GLOBAL_LOADED_MODELS:
            return jsonify({"error": "後端尚未載入任何模型，請檢查後端啟動時的輸出。確保模型檔案存在且未損壞。"}), 503

    data = request.get_json()
    if not data or 'images' not in data:
        return jsonify({"error": "請求中缺少 'images' 列表。請確保發送正確的 JSON 格式。"}), 400

    image_urls = data['images']
    if not isinstance(image_urls, list):
        return jsonify({"error": "'images' 必須是一個 URL 列表。"}), 400

    print(f"從前端接收到 {len(image_urls)} 個圖片 URL 請求。")
    predictions = predict_scores_from_image_urls(image_urls, GLOBAL_LOADED_MODELS)

    successful_predictions = [p for p in predictions if 'error' not in p]
    error_predictions = [p for p in predictions if 'error' in p]

    if error_predictions:
        print(f"警告: 有 {len(error_predictions)} 張圖片處理失敗。")
        return jsonify(predictions), 200
    else:
        return jsonify(predictions), 200


# ==============================================================================
# 本機執行流程 (修改為在啟動時從 GCS 下載模型)
# ==============================================================================
if __name__ == "__main__":
    print("\n--- 載入預訓練模型 ---")
    # 在應用程式啟動時先從 GCS 下載模型
    download_models_from_gcs()
    GLOBAL_LOADED_MODELS = initialize_and_load_models()

    print("\n--- 啟動 Flask API 服務 ---")
    print("服務正在監聽來自前端的圖片 URL 請求。")
    print("請確保你的前端 JavaScript 中的 BACKEND_API_ENDPOINT 指向 Cloud Run 服務的 URL。")
    print("服務啟動中，請等待... (Ctrl+C 終止)")

    app.run(debug=False, host='0.0.0.0', port=os.getenv('PORT', 8080))
