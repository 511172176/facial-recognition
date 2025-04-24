import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.optimizers.experimental import AdamW
#from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.layers import Lambda

from tqdm.keras import TqdmCallback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Reshape, Add, Multiply, Conv2D, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.optimizers.schedules import CosineDecay

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=\"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8\""

############################################
# 清理訓練日誌（logs）
############################################
'''class CleanLogsCallback(tf.keras.callbacks.Callback):
    def _clean(self, logs):
        for k, v in list(logs.items()):
            if isinstance(v, tf.Tensor):
                arr = v.numpy()
                logs[k] = arr.tolist() if arr.ndim > 0 else float(arr)
            elif isinstance(v, np.ndarray):
                logs[k] = v.tolist() if v.ndim > 0 else float(v)

    def on_train_batch_end(self, batch, logs=None):
        if logs: self._clean(logs)

    def on_test_batch_end(self, batch, logs=None):
        if logs: self._clean(logs)

    def on_epoch_end(self, epoch, logs=None):
        if logs: self._clean(logs)'''

############################################
# 1. 參數與資料路徑設定
############################################
IMG_SIZE      = (224, 224)
NUM_CLASSES   = 7
emotion_labels = ['angry','disgust','fear','happy','sad','surprise','neutral']
TRAIN_PATH    = "FER2013/train"
TEST_PATH     = "FER2013/test"
BATCH_SIZE    = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 30
WEIGHT_DECAY  = 1e-4

############################################
# 2. 自訂裁切函式，用於增強
############################################
def random_crop(img):
    h, w, _ = img.shape
    # 隨機裁切 80-100% 範圍
    crop_h = int(np.random.uniform(0.8, 1.0) * h)
    crop_w = int(np.random.uniform(0.8, 1.0) * w)
    y = np.random.randint(0, h - crop_h + 1)
    x = np.random.randint(0, w - crop_w + 1)
    img_c = img[y:y+crop_h, x:x+crop_w, :]
    return tf.image.resize(img_c, IMG_SIZE).numpy()
    '''return tf.image.resize(img_c, IMG_SIZE)'''
# 保留 TensorFlow Tensor，避免在 ImageDataGenerator 中轉型效率低下

############################################
# 3. 資料載入與增強設定（含旋轉、平移、裁切、縮放、亮度、色道偏移）
############################################
'''train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.5,1.5],
    channel_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.2,
    preprocessing_function=random_crop
)'''

# 將 train_datagen 修改為較保守增強
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    directory=TRAIN_PATH,
    target_size=IMG_SIZE,
    class_mode='categorical',
    subset='training',
    batch_size=BATCH_SIZE,
    shuffle=True
)
validation_generator = train_datagen.flow_from_directory(
    directory=TRAIN_PATH,
    target_size=IMG_SIZE,
    class_mode='categorical',
    subset='validation',
    batch_size=BATCH_SIZE,
    shuffle=False
)
test_generator = test_datagen.flow_from_directory(
    directory=TEST_PATH,
    target_size=IMG_SIZE,
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False
)

############################################
# 4. 定義 MixUp + CutMix 生成器
############################################
'''def mix_cutmix_generator(generator, alpha=0.2, cutmix_prob=0.0):
    while True:
        X, y = next(generator)
        batch_size, h, w, c = X.shape
        if np.random.rand() < cutmix_prob:
            # CutMix
            lam = np.random.beta(alpha, alpha)
            idx = np.random.permutation(batch_size)
            # 計算裁切區域
            cut_rat = np.sqrt(1 - lam)
            cut_w = int(w * cut_rat)
            cut_h = int(h * cut_rat)
            cx = np.random.randint(w)
            cy = np.random.randint(h)
            x1 = np.clip(cx - cut_w // 2, 0, w)
            y1 = np.clip(cy - cut_h // 2, 0, h)
            x2 = np.clip(cx + cut_w // 2, 0, w)
            y2 = np.clip(cy + cut_h // 2, 0, h)
            X_cutmix = X.copy()
            X_cutmix[:, y1:y2, x1:x2, :] = X[idx, y1:y2, x1:x2, :]
            # 調整 λ
            lam_adj = 1 - ((x2-x1)*(y2-y1) / (w*h))
            y_cutmix = lam_adj * y + (1 - lam_adj) * y[idx]
            yield X_cutmix, y_cutmix
        else:
            # MixUp
            lam = np.random.beta(alpha, alpha)
            idx = np.random.permutation(batch_size)
            X_mix = lam * X + (1 - lam) * X[idx]
            y_mix = lam * y + (1 - lam) * y[idx]
            yield X_mix, y_mix'''

# 將 ImageDataGenerator 包裝成 tf.data.Dataset（這段沒加）
def generator_wrapper():
    while True:
        X, y = next(train_generator)
        yield X, y

train_ds_base = tf.data.Dataset.from_generator(
    generator_wrapper,
    output_types=(tf.float32, tf.float32),
    output_shapes=([BATCH_SIZE, *IMG_SIZE, 3], [BATCH_SIZE, NUM_CLASSES])
)
# 因 train_ds 為 from_generator 轉換而成的 infinite dataset，需手動設定 steps_per_epoch

'''def mixup_tf(x, y, alpha=0.2):
    beta = tf.random.gamma([], alpha, 1.0)
    beta = tf.random.beta([], alpha, alpha)
    idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))
    x2 = tf.gather(x, idx)
    y2 = tf.gather(y, idx)
    x_mix = beta * x + (1. - beta) * x2
    y_mix = beta * y + (1. - beta) * y2
    return x_mix, y_mix'''

def mixup_tf(x, y, alpha=0.2):
    beta = tf.random.gamma([], alpha, 1.0)
    idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))  # 保留 batch 維度
    x2 = tf.gather(x, idx)
    y2 = tf.gather(y, idx)
    x_mix = beta * x + (1 - beta) * x2
    y_mix = beta * y + (1 - beta) * y2
    return x_mix, y_mix

def mixup_generator(generator, alpha=0.4):
    def gen():
        while True:
            x, y = next(generator)
            # 跳過最後那個少於 BATCH_SIZE 的小 batch
            if x.shape[0] != BATCH_SIZE:
                continue

            beta = np.random.beta(alpha, alpha)
            idx = np.random.permutation(BATCH_SIZE)
            x_mix = beta * x + (1 - beta) * x[idx]
            y_mix = beta * y + (1 - beta) * y[idx]
            yield x_mix, y_mix

    return gen()

train_ds = train_ds_base.map(lambda x, y: mixup_tf(x, y, alpha=0.2), num_parallel_calls=tf.data.AUTOTUNE)
'''train_ds = train_ds.prefetch(tf.data.AUTOTUNE)'''
train_ds = train_ds.repeat().prefetch(tf.data.AUTOTUNE)



############################################
# 5. 定義 CBAM 注意力模組（Channel + Spatial）
############################################
def spatial_attention(inputs):
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(inputs)
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(inputs)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    cb = Conv2D(
        filters=1,
        kernel_size=7,
        strides=1,
        padding='same',
        activation='sigmoid',
        kernel_initializer='he_normal'
    )(concat)
    return Multiply()([inputs, cb])

def cbam_block(inputs, ratio=8):
    channel = inputs.shape[-1]
    # Channel Attention
    shared_dense_one = Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(WEIGHT_DECAY))
    shared_dense_two = Dense(channel, kernel_initializer='he_normal', kernel_regularizer=l2(WEIGHT_DECAY))
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    avg_pool = Reshape((1,1,channel))(avg_pool)
    avg_pool = shared_dense_two(shared_dense_one(avg_pool))
    max_pool = tf.keras.layers.GlobalMaxPooling2D()(inputs)
    max_pool = Reshape((1,1,channel))(max_pool)
    max_pool = shared_dense_two(shared_dense_one(max_pool))
    cb_channel = Add()([avg_pool, max_pool])
    cb_channel = tf.keras.layers.Activation('sigmoid')(cb_channel)
    x = Multiply()([inputs, cb_channel])
    # Spatial Attention
    x = spatial_attention(x)
    return x

############################################
# 6. 定義 Label Smoothing + Focal Loss
############################################
def smooth_focal_loss(gamma=2.0, alpha=0.25, label_smoothing=0.05):
    def loss_fn(y_true, y_pred):
        eps = 1e-8
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
        num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
        y_true_ls = y_true * (1 - label_smoothing) + label_smoothing / num_classes
        ce = -y_true_ls * tf.math.log(y_pred)
        fl = alpha * tf.pow(1 - y_pred, gamma) * ce
        return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))
    return loss_fn

############################################
# 7. 建立模型：ResNet50 + CBAM + L2 正則化
############################################
'''inputs = Input(shape=(*IMG_SIZE, 3))
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_tensor=inputs
)
base_model.trainable = False  # Phase1 凍結

x = cbam_block(base_model.output) #Resnet50 CBAM
#x = base_model.output #EfficientNetB3 SE
x = GlobalAveragePooling2D()(x)
x = Dense(
    256,
    activation='relu',
    kernel_regularizer=l2(WEIGHT_DECAY)
)(x)
x = Dropout(0.3)(x)
outputs = Dense(
    NUM_CLASSES,
    activation='softmax',
    kernel_regularizer=l2(WEIGHT_DECAY)
)(x)
model = Model(inputs=inputs, outputs=outputs)
model.summary()

# 計算 steps
steps_per_epoch = train_generator.samples // BATCH_SIZE
#steps_per_epoch = train_generator.n // BATCH_SIZE
#steps_per_epoch = len(train_generator)
# len(train_generator) 可等同於 train_generator.n // BATCH_SIZE，適用於 tf.data 訓練步數設定'''

############################################
# 7. 建立模型：EfficientNetB3+SE
############################################
inputs = Input(shape=(*IMG_SIZE, 3))
base_model = EfficientNetV2B3(
    weights='imagenet',
    include_top=False,
    input_tensor=inputs
)
base_model.trainable = False  # Phase1 凍結

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(
    256,
    activation='relu',
    kernel_regularizer=l2(WEIGHT_DECAY)
)(x)
x = Dropout(0.5)(x)
outputs = Dense(
    NUM_CLASSES,
    activation='softmax',
    kernel_regularizer=l2(WEIGHT_DECAY)
)(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

steps_per_epoch = train_generator.samples // BATCH_SIZE

############################################
# 8. 計算 Class Weights
############################################
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weight_dict = {
    int(k): float(v) for k, v in enumerate(class_weights)
}

print("Class weights:", class_weight_dict)
#print("Class weights:", class_weight_dict)

############################################
# 9. Phase1 訓練（含 AdamW + CosineDecayRestarts）
############################################
# Learning rate schedule
lr_schedule_phase1 = CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=steps_per_epoch * EPOCHS_PHASE1,
    alpha=1e-6
)
optimizer_phase1 = AdamW(
    learning_rate=lr_schedule_phase1,
    weight_decay=WEIGHT_DECAY
)
model.compile(
    optimizer=optimizer_phase1,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

logdir1 = os.path.join(
    "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S_phase1")
)
callbacks_phase1 = [
    #CleanLogsCallback(),     # 清理 logs
    ModelCheckpoint(
        "best_model_phase1.h5",
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=False
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    ),
    TensorBoard(log_dir=logdir1),
    TqdmCallback(verbose=0)
]

history1 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE1,
    validation_data=validation_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks_phase1,
    verbose=1
)

############################################
# 10. Phase2 Fine-Tuning（解凍後150層 + MixUp+CutMix + AdamW + CosineDecayRestarts）
############################################

# (1) 解凍最後 50 層（視需求可調整）
for layer in base_model.layers[-150:]:
    layer.trainable = True

# (2) 調整學習率排程：讓前兩個 epoch 內完成一次 cosine decay cycle
lr_schedule_phase2 = CosineDecayRestarts(
    initial_learning_rate=1e-5,
    first_decay_steps=steps_per_epoch * 10,
    t_mul=2.0,
    m_mul=1.0,
    alpha=1e-5
)

optimizer_phase2 = AdamW(
    learning_rate=lr_schedule_phase2,
    weight_decay=WEIGHT_DECAY
)

# (3) 重新 compile，用帶標籤平滑的 Focal Loss
model.compile(
    optimizer=optimizer_phase2,
    loss=smooth_focal_loss(gamma=2.0, alpha=0.25, label_smoothing=0.05),
    metrics=['accuracy']
)

# (4) 建立 MixUp generator（直接套用剛才定義好的 mixup_generator）
train_mix = tf.data.Dataset.from_generator(
    lambda: mixup_generator(train_generator, alpha=0.4),
    output_signature=(
        tf.TensorSpec(shape=(BATCH_SIZE, *IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(BATCH_SIZE, NUM_CLASSES), dtype=tf.float32),
    )
).repeat().prefetch(tf.data.AUTOTUNE)

# (5) Phase2 開始訓練
logdir2 = os.path.join(
    "logs",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S_phase2")
)

callbacks_phase2 = [
    ModelCheckpoint(
        "best_model_phase2.h5",
        monitor='val_loss',    # 改成監控 val_loss
        save_best_only=True,
        mode='min'
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True  # 不 restore 回 baseline
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    ),
    TensorBoard(log_dir=logdir2),
    TqdmCallback(verbose=0)
]

history2 = model.fit(
    train_mix,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS_PHASE2,
    validation_data=validation_generator,
    callbacks=callbacks_phase2,
    verbose=1
)

############################################
# 11. 測試集評估
############################################
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"\nTest Loss: {test_loss:.4f}\nTest Accuracy: {test_acc:.4f}")

############################################
# 12. 繪製訓練曲線
############################################
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss_vals = history1.history['loss'] + history2.history['loss']
val_loss_vals = history1.history['val_loss'] + history2.history['val_loss']

plt.figure()
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Val Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend(); plt.tight_layout()
plt.savefig('training_accuracy.png', dpi=300)
plt.close()

plt.figure()
plt.plot(loss_vals, label='Train Loss')
plt.plot(val_loss_vals, label='Val Loss')
plt.title('Training Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.tight_layout()
plt.savefig('training_loss.png', dpi=300)
plt.close()

############################################
# 13. 混淆矩陣 & 分類報告
############################################
y_pred = model.predict(test_generator)
y_pred_lbls = np.argmax(y_pred, axis=1)
y_true_lbls = test_generator.classes

cm = confusion_matrix(y_true_lbls, y_pred_lbls)
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, None]
plt.figure(figsize=(8,6))
plt.imshow(cm_norm, cmap='Blues')
plt.title('Normalized Confusion Matrix')
plt.colorbar()
ticks = np.arange(len(emotion_labels))
plt.xticks(ticks, emotion_labels, rotation=45)
plt.yticks(ticks, emotion_labels)
plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout(); plt.savefig('confusion_matrix_norm.png', dpi=300)
plt.close()

report = classification_report(
    y_true_lbls, y_pred_lbls,
    target_names=emotion_labels,
    zero_division=0
)
print(report)

df_report = pd.DataFrame(
    classification_report(
        y_true_lbls, y_pred_lbls,
        target_names=emotion_labels,
        output_dict=True,
        zero_division=0
    )
).transpose()
fig, ax = plt.subplots(figsize=(8, df_report.shape[0]*0.5))
ax.axis('off')
tbl = ax.table(
    cellText=df_report.round(3).values,
    colLabels=df_report.columns,
    rowLabels=df_report.index,
    cellLoc='center',
    loc='center'
)
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
plt.tight_layout(); plt.savefig('classification_report.png', dpi=300)
plt.close()