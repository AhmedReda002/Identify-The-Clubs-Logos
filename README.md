# Identify the Clubs' Logos

## Overview
This project uses a pre-trained VGG16 deep learning model to identify football club logos from images. The model is fine-tuned for this specific task and is built with TensorFlow, leveraging transfer learning to achieve high accuracy.

## Libraries Used
- `tensorflow.keras`: For building, training, and utilizing the deep learning model.
- `numpy`: For numerical operations.
- `matplotlib`: For plotting training history and visualization.
- `ImageDataGenerator` from `tensorflow.keras.preprocessing.image`: For generating batches of tensor image data with real-time data augmentation.

## Features
- Data preparation and augmentation.
- Transfer learning with VGG16 for efficient model training.
- Early stopping callback to prevent overfitting.
- Visualization of training and validation accuracy and loss.
- Prediction function to classify club logos from input images.


## Usage

### Training the Model

1. Import the necessary libraries:
    ```python
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    ```

2. Data preparation and augmentation:
    ```python
    data_dir = '/path/to/your/data'  # Replace with the actual path to your data folder

    train_datagen = ImageDataGenerator(rescale=0.255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    ```

3. Model creation using transfer learning with VGG16:
    ```python
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(train_generator.num_classes, activation='softmax'))

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    ```

4. Model training with early stopping:
    ```python
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=20,
        callbacks=[early_stopping]
    )
    ```

5. Plot training and validation accuracy and loss:
    ```python
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'] if 'val_accuracy' in history.history else ['Train'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'] if 'val_loss' in history.history else ['Train'], loc='upper left')

    plt.show()
    ```

6. Save the model:
    ```python
    model.save('/path/to/your/model')  # Replace with the actual path to save your model
    ```

### Loading the Pre-trained Model for Inference

1. Import the necessary libraries:
    ```python
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    import numpy as np
    ```

2. Load the pre-trained model:
    ```python
    model = load_model('/path/to/your/model')  # Replace with the actual path to your saved model
    ```

3. Define the prediction function:
    ```python
    def predict_club(image_path, model, target_size=(224, 224)):
        img = image.load_img(image_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        class_label = train_generator.class_indices
        class_label = {v: k for k, v in class_label.items()}
        return class_label[class_idx]
    ```

4. Example usage:
    ```python
    print(predict_club('/path/to/your/image.png', model))  # Replace with the path to your image file
    ```

## Output
The script will output the predicted football club's name based on the input image.
