import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


data_dir = r"C:\Users\HP\Desktop\Himanshu Singh\DataSets\brain_tumor_dataset"  
train_dir = os.path.join(data_dir, "Training")
test_dir = os.path.join(data_dir, "Testing")


img_size = (150, 150)
batch_size = 32
epochs = 10
learning_rate = 0.0001

data_gen = ImageDataGenerator(
    rescale=1.0/255.0,  
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = data_gen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = ImageDataGenerator(rescale=1.0/255.0).flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

base_model.trainable = False

x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
out = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=out)

model.compile(optimizer=Adam(learning_rate=learning_rate), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('brain_tumor_model_vgg16.h5', save_best_only=True)
]

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=epochs,
    callbacks=callbacks
)

for layer in base_model.layers[-4:]: 
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=learning_rate / 10), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

history_fine = model.fit(
    train_data,
    validation_data=test_data,
    epochs=epochs,
    callbacks=callbacks
)


plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.plot(history_fine.history['accuracy'], label='Train Accuracy (Fine-tuning)')
plt.plot(history_fine.history['val_accuracy'], label='Test Accuracy (Fine-tuning)')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save("final_brain_tumor_model_vgg16V2.h5")
