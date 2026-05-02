import os
import tensorflow as tf

original_from_config = tf.keras.layers.Dense.from_config

def patched_from_config(cls, config):
    if 'quantization_config' in config:
        del config['quantization_config']
    return original_from_config(config)

tf.keras.layers.Dense.from_config = classmethod(patched_from_config)

path = 'brain_tumor_model.keras'
print(f"Loading {path}...")
model = tf.keras.models.load_model(path)
print("Loaded successfully!")
model.summary()
