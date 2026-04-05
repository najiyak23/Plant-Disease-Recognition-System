import tensorflow as tf

model = tf.keras.layers.TFSMLayer("final_model", call_endpoint="serving_default")

print("✅ Model loaded successfully!")

