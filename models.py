from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, MaxPooling2D, Flatten, Input, Reshape
from tensorflow.keras.models import Model, Sequential

# 利用可能なモデルを定義
AVAILABLE_MODELS = ["MobileNetV2", "ResNet50", "EfficientNetB0", "SimpleCNN", "SimpleMLP"]

def build_model(name, num_classes):
    if name == "MobileNetV2":
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        x = base.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        output = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=base.input, outputs=output)
    
    elif name == "ResNet50":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        x = base.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        output = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=base.input, outputs=output)

    elif name == "EfficientNetB0":
        base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        x = base.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        output = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=base.input, outputs=output)
    
    elif name == "SimpleCNN":
        model = Sequential([
            Input(shape=(224, 224, 3)),
            Conv2D(16, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
    
    elif name == "SimpleMLP":
        model = Sequential([
            Input(shape=(224, 224, 3)),
            Flatten(),  # 画像を1次元に
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
    
    else:
        raise ValueError(f"Unsupported model: {name}. Choose from {AVAILABLE_MODELS}")
    
    return model
