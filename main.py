# main.py
import os
import json
import numpy as np
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

# === 設定ファイル読み込み ===
with open("config.json", "r") as f:
    config = json.load(f)

# === 設定値の読み込み ===
AUGMENTATIONS = config["augmentations"]  # 今後の拡張で使う予定
NUM_AUGS = config["num_augmentations"]
MODEL_NAME = config["model"]
EPOCHS = config["epochs"]
BATCH_SIZE = config["batch_size"]
CONFIDENCE_THRESH = config["confidence_threshold"]

# === パス定義 ===
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
OUTPUT_DIR = "data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === データジェネレータ ===
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# === ラベルの取得 ===
class_indices = train_gen.class_indices
class_labels = {v: k for k, v in class_indices.items()}

# === モデル構築（MobileNetV2） ===
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(len(class_labels), activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)
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

print("\n推論完了\n\n正しく分類された画像は data/output に保存されました。")

