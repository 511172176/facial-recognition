import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm.keras import TqdmCallback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Multiply, Reshape, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

############################################
# 1. 參數與資料路徑設定
############################################
IMG_SIZE      = (224, 224)   # ResNet50 輸入尺寸
NUM_CLASSES   = 7
emotion_labels = ['angry','disgust','fear','happy','sad','surprise','neutral']
TRAIN_PATH    = "FER2013/train"
TEST_PATH     = "FER2013/test"
BATCH_SIZE    = 16
EPOCHS_PHASE1 = 30
EPOCHS_PHASE2 = 50

############################################
# 2. 資料載入與增強設定
############################################
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    brightness_range=[0.8,1.2],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
test_datagen = ImageDataGenerator(rescale=1./255)

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
# 3. Mixup 生成器
############################################
def mixup_generator(generator, alpha=0.2):
    while True:
        X, y = next(generator)
        lam = np.random.beta(alpha, alpha)
        idx = np.random.permutation(len(X))
        X_mix = lam * X + (1 - lam) * X[idx]
        y_mix = lam * y + (1 - lam) * y[idx]
        yield X_mix, y_mix

############################################
# 4. 定義 CBAM 注意力模組
############################################
def cbam_block(inputs, ratio=8):
    channel = inputs.shape[-1]
    shared1 = Dense(channel//ratio, activation='relu', kernel_initializer='he_normal')
    shared2 = Dense(channel, kernel_initializer='he_normal')

    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    avg_pool = Reshape((1,1,channel))(avg_pool)
    avg_pool = shared2(shared1(avg_pool))

    max_pool = tf.keras.layers.GlobalMaxPooling2D()(inputs)
    max_pool = Reshape((1,1,channel))(max_pool)
    max_pool = shared2(shared1(max_pool))

    cb = Add()([avg_pool, max_pool])
    cb = tf.keras.layers.Activation('sigmoid')(cb)
    return Multiply()([inputs, cb])

############################################
# 5. 定義 Focal Loss
############################################
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        eps = 1e-8
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
        ce = -y_true * tf.math.log(y_pred)
        fl = alpha * tf.pow(1 - y_pred, gamma) * ce
        return tf.reduce_mean(tf.reduce_sum(fl, axis=1))
    return loss_fn

############################################
# 6. 建立模型：ResNet50 + CBAM
############################################
base_model = tf.keras.applications.ResNet50(
    weights='imagenet', include_top=False,
    input_shape=(*IMG_SIZE, 3)
)
base_model.trainable = False  # Phase1 先凍結

x = cbam_block(base_model.output)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=outputs)
model.summary()

############################################
# 7. Phase1 訓練（僅訓練分類層）
############################################
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

logdir1 = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S_phase1"))
callbacks_phase1 = [
    ModelCheckpoint("best_model_phase1.h5", monitor='val_accuracy', save_best_only=True, mode='max'),
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    TensorBoard(log_dir=logdir1),
    TqdmCallback(verbose=0)
]

history1 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE1,
    validation_data=validation_generator,
    callbacks=callbacks_phase1,
    verbose=1
)

############################################
# 8. 計算 Class Weights
############################################
class_weights = compute_class_weight(
    'balanced', classes=np.unique(train_generator.classes), y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

############################################
# 9. Phase2 Fine-Tuning（解凍後150層 + Mixup + Focal Loss）
############################################
for layer in base_model.layers[-150:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy']
)

train_mix = mixup_generator(train_generator)
steps = train_generator.samples // BATCH_SIZE

logdir2 = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S_phase2"))
callbacks_phase2 = [
    ModelCheckpoint("best_model_phase2.h5", monitor='val_accuracy', save_best_only=True, mode='max'),
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    TensorBoard(log_dir=logdir2),
    TqdmCallback(verbose=0)
]

history2 = model.fit(
    train_mix,
    steps_per_epoch=steps,
    epochs=EPOCHS_PHASE2,
    validation_data=validation_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks_phase2,
    verbose=1
)

############################################
# 10. 測試集評估
############################################
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"\nTest Loss: {test_loss:.4f}\nTest Accuracy: {test_acc:.4f}")

############################################
# 11. 繪製訓練曲線
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
# 12. 混淆矩陣 & 分類報告
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

report = classification_report(y_true_lbls, y_pred_lbls, target_names=emotion_labels, zero_division=0)
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
    cellLoc='center', loc='center'
)
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
plt.tight_layout(); plt.savefig('classification_report.png', dpi=300)
plt.close()
