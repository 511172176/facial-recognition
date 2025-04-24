import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

print("版本：", tf.__version__)
print("GPU 裝置：", tf.config.list_physical_devices('GPU'))
