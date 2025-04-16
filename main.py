# main.py
import os
import json
import numpy as np
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import cv2
from augmentations import get_augmentations
from models import build_model
import albumentations as A

# === 設定ファイル読み込み ===
with open("config.json", "r") as f:
    config = json.load(f)

# === 設定値の読み込み ===
AUGMENTATION_NAMES = config["augmentations"]
NUM_AUGS = config["num_augmentations"]
MODEL_NAME = config["model"]
EPOCHS = config["epochs"]
BATCH_SIZE = config["batch_size"]
CONFIDENCE_THRESH = config["confidence_threshold"]

# === パス定義 ===
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
OUTPUT_DIR = "data/output"
AUG_TRAIN_DIR = "data/train_aug"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUG_TRAIN_DIR, exist_ok=True)

# === 前処理の取得 ===
augmentations = get_augmentations(AUGMENTATION_NAMES)

# === データ拡張の適用 ===
def apply_augmentations_to_dataset():
    for cls in os.listdir(TRAIN_DIR):
        src_cls_path = os.path.join(TRAIN_DIR, cls)
        dst_cls_path = os.path.join(AUG_TRAIN_DIR, cls)
        os.makedirs(dst_cls_path, exist_ok=True)
        for fname in os.listdir(src_cls_path):
            src_path = os.path.join(src_cls_path, fname)
            img = cv2.imread(src_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 元画像もコピー
            base_dst = os.path.join(dst_cls_path, fname)
            cv2.imwrite(base_dst, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            for i in range(NUM_AUGS):
                for aug in augmentations:
                    aug_img = aug(image=img)["image"]
                    aug_fname = f"{os.path.splitext(fname)[0]}_aug{i}_{aug.__class__.__name__}.jpg"
                    aug_path = os.path.join(dst_cls_path, aug_fname)
                    cv2.imwrite(aug_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

apply_augmentations_to_dataset()

# === データジェネレータ ===
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_gen = datagen.flow_from_directory(
    AUG_TRAIN_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# === ラベルの取得 ===
class_indices = train_gen.class_indices
class_labels = {v: k for k, v in class_indices.items()}

# === モデル構築（外部ファイルから） ===
model = build_model(MODEL_NAME, num_classes=len(class_labels))
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# === 学習 ===
model.fit(train_gen, epochs=EPOCHS, verbose=1)

# === 推論関数 ===
def predict_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx]
    return class_labels[class_idx], confidence

# === 推論実行と保存 ===
for fname in os.listdir(TEST_DIR):
    img_path = os.path.join(TEST_DIR, fname)
    label, conf = predict_image(img_path)
    if conf >= CONFIDENCE_THRESH:
        shutil.copy(img_path, os.path.join(OUTPUT_DIR, fname))
        print(f"[OK] {fname} -> {label} ({conf:.2f})")
    else:
        print(f"[NG] {fname} -> {label} ({conf:.2f})")

print("\n推論完了！ 正しく分類された画像は data/output に保存されました。")
