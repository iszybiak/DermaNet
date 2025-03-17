import keras
from keras.src.applications.mobilenet_v2 import MobileNetV2
from keras import layers, models
from src.data_preparing import train_generator, validation_generator

# Loading ready model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Frizzing base layers
base_model.trainable = False

# Adding own net head
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])

# Model compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
history = model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Unfreezing the last layers
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Recompiling model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Further training
history_fine = model.fit(train_generator, validation_data=validation_generator, epochs=5)

# Saving model
model.save("skin_disease_model.keras")




