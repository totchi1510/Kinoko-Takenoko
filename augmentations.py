# augmentations.py
import albumentations as A

# 利用可能な前処理を定義
AUGMENTATION_OPTIONS = {
    "HorizontalFlip": A.HorizontalFlip(p=1.0),
    "Rotate": A.Rotate(limit=30, p=1.0),
    "Blur": A.Blur(blur_limit=3, p=1.0),
    "Sharpen": A.Sharpen(p=1.0),
    "RandomBrightnessContrast": A.RandomBrightnessContrast(p=1.0),
    "GaussNoise": A.GaussNoise(p=1.0),
    "HueSaturationValue": A.HueSaturationValue(p=1.0),
    "RandomScale": A.RandomScale(scale_limit=0.2, p=1.0)

}

# 指定された拡張名リストから、適用する拡張をリストで返す関数
def get_augmentations(names):
    return [AUGMENTATION_OPTIONS[name] for name in names if name in AUGMENTATION_OPTIONS]
