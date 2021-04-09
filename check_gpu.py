# check_gpu.py
# -------------------------------------------------------------------------------------------------------- 
# Chek GPU
# --------------------------------------------------------------------------------------------------------

import tensorflow as tf
print(f'Tensorflow version: {tf.__version__ }')
print(f'''Detected GPU(s): {tf.config.list_physical_devices('GPU')}''')

from tensorflow import keras
print(f'Keras version: {keras.__version__}')

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())