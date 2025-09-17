import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
from PIL import ImageFile
from Constants import *
import os, cv2, datetime
import numpy as np

if USE_AMP:
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy('mixed_float16')

ImageFile.LOAD_TRUNCATED_IMAGES = True
def create_checkpoint():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    curTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoints = [
        ModelCheckpoint(
            filepath=os.path.join('checkpoints', f'efficientnet_{curTime}.keras'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=False,
            verbose=1
        ),
        CSVLogger(
            os.path.join('logs', f'efficientnet_{curTime}.csv'),
            separator=',',
            append=False
        ),
        TensorBoard(
            log_dir=os.path.join('logs', 'tensorboard', f'smth{curTime}'),
            histogram_freq=1,
            write_graph=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
    return checkpoints

def load_latest_checkpoint(model):
    ckpt_dir = "checkpoints"
    if not os.path.exists(ckpt_dir):
        return model
    
    checkpoints = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".keras")]
    if not checkpoints:
        return model
    
    latest_ckpt = max(checkpoints, key=os.path.getctime)
    print(f"Loading weights from {latest_ckpt}")
    model.load_weights(latest_ckpt)
    return model

def preprocess_image(img):
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    clahe_img = np.stack([clahe_img]*3, axis=-1)
    clahe_img = clahe_img.astype(np.float32) / 255.
    return clahe_img

def efficientnet_model():
    base_model = EfficientNetB0(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    for layer in base_model.layers[:100]:  # freeze first 100 layers
        layer.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model

if __name__ == "__main__":
    train_data = ImageDataGenerator(preprocessing_function=preprocess_image).flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    test_data = ImageDataGenerator(preprocessing_function=preprocess_image).flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    model = efficientnet_model()
    model = load_latest_checkpoint(model)
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], jit_compile=True)

    try:
        hist = model.fit(train_data, epochs=EPOCHS, validation_data=test_data, validation_steps=len(test_data), callbacks=create_checkpoint(), initial_epoch=INIT_EPOCH)

        model.save(os.path.join('models', 'efficientnet_model_final.keras'))
    except Exception as e:
        print(f"Error occurred: {e}")