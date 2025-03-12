import tensorflow as tf

# Define input shape
inputs = tf.keras.Input(shape=(512, 512, 3))

# Load pre-trained models (without top layers)
xception_base = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_tensor=inputs)
densenet_base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_tensor=inputs)

# Global pooling layers
xception_output = tf.keras.layers.GlobalAveragePooling2D()(xception_base.output)  # (2048,)
densenet_output = tf.keras.layers.GlobalAveragePooling2D()(densenet_base.output)  # (1024,)

# Match the feature sizes by adding a Dense(2048) layer to DenseNet output
densenet_output = tf.keras.layers.Dense(2048, activation="relu")(densenet_output)  # Now (2048,)

# Averaging ensemble
merged_output = tf.keras.layers.Average()([xception_output, densenet_output])

# Final classification layer (4 classes)
final_output = tf.keras.layers.Dense(4, activation="softmax")(merged_output)

# Define the model
model = tf.keras.Model(inputs=inputs, outputs=final_output)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Save the full model
model.save("ensemble_model.h5")

print("Model saved as ensemble_model.h5")
