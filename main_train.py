# main_train.py
#その１　ここを完成させてください
#!!osというモジュールをインポートする!!

import json
#!!numpyというモジュールをnpという名前でインポートする!!

import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import cv2
from augmentations import get_augmentations
from models import build_model
import albumentations as A
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
            #その2 ここを完成させてください
            #!!for文を使ってリストaugmentationsの先頭から順にaugに代入して繰り返し処理を行う!!
                #この列から書いてください
                
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
#その3　ここを完成させてください
#!!predict_imageという関数を定義して、引数をimg_pathにする!!
    
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx]
    return class_labels[class_idx], confidence
# === 推論結果保存先の作成 ===
for label in class_labels.values():
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)

# === 推論結果保存 + 結果収集 ===
results = []

# === 画像ファイル拡張子の定義 ===
VALID_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

for fname in os.listdir(TEST_DIR):

    if not any(fname.lower().endswith(ext) for ext in VALID_EXTENSIONS):
        print(f"[SKIP] {fname} は画像ファイルではありません。")
        continue
        
    img_path = os.path.join(TEST_DIR, fname)
    label, conf = predict_image(img_path)
    results.append((fname, label, conf))
  
#その4　ここを完成させてください
    #この段落から書く
    #!!もしconfの値がCONFIDENCE_THRESH以上であれば!!
  　#ヒント：以上は>=
    
        dst_dir = os.path.join(OUTPUT_DIR, label)
        shutil.copy(img_path, os.path.join(dst_dir, fname))
        print(f"[OK] {fname} -> {label} ({conf:.2f})")
    #それ以外の時は
    
        print(f"[NG] {fname} -> {label} ({conf:.2f})")

#その5　ここを完成させてください
#!!段落を変えて、「推論完了！ 正しく分類された画像は data/output/配下に保存されました。」というフレーズを表示させる!!
#ヒント　段落を変えるときは \n

# === ヒートマップ描画 ===

# DataFrame 作成
df = pd.DataFrame(results, columns=["filename", "label", "confidence"])
df["filename"] = df["filename"].astype(str)

# ヒートマップ用に行列作成（行：画像、列：クラス）
heatmap_data = pd.DataFrame(0, index=df["filename"], columns=class_labels.values())
for _, row in df.iterrows():
    heatmap_data.at[row["filename"], row["label"]] = row["confidence"]

# ヒートマップ表示
plt.figure(figsize=(10, len(heatmap_data) * 0.3 + 1))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar=True, fmt=".2f")
plt.title("Prediction Confidence Heatmap")
plt.xlabel("Class")
plt.ylabel("Image Filename")
plt.tight_layout()
plt.show()
