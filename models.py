# models.py
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# 利用可能なモデルを定義
AVAILABLE_MODELS = ["MobileNetV2", "ResNet50", "EfficientNetB0"]

def build_model(name, num_classes):
    if name == "MobileNetV2":
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    elif name == "ResNet50":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    elif name == "EfficientNetB0":
        base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError(f"Unsupported model: {name}. Choose from {AVAILABLE_MODELS}")

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=output)
    return model
