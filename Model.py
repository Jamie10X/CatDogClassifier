import os

from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.legacy import Adam

# Path to the dataset
rootPath = '/Users/jamsh/Downloads/CatAndDog'

# Data Augmentation and Preprocessing
imageGenerator = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[.2, .2],
    horizontal_flip=True,
    validation_split=.2
)
# Preparing Training Data
trainGen = imageGenerator.flow_from_directory(
    os.path.join(rootPath, 'training_set'),
    target_size=(128, 128),  # Increased target size
    batch_size=32,  # Added batch size
    subset='training'
)
# Preparing Validation Data
validationGen = imageGenerator.flow_from_directory(
    os.path.join(rootPath, 'training_set'),
    target_size=(128, 128),  # Increased target size
    batch_size=32,  # Added batch size
    subset='validation'
)

# CNN Model Architecture
model = Sequential()
model.add(layers.InputLayer(input_shape=(128, 128, 3)))
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())  # Added Batch Normalization
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())  # Added Batch Normalization
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())  # Added Batch Normalization
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # Adjusted dropout rate
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid'))
# model summary
model.summary()

# Model Compilation
opt = Adam(learning_rate=0.001)
model.compile(
    optimizer=opt,
    loss='binary_crossentropy',
    metrics=['acc'],
)

# Model Training with Early Stopping
epochs = 100  # Adjusted number of epochs
es = EarlyStopping(monitor='val_loss', mode='min', patience=10)  # Adjusted patience
history = model.fit(
    trainGen,
    epochs=epochs,
    steps_per_epoch=trainGen.samples // trainGen.batch_size,
    validation_data=validationGen,
    validation_steps=validationGen.samples // validationGen.batch_size,
    callbacks=[es]
)


# Visualization of Training Results
def show_graph(history_dict):
    accuracy = history_dict['acc']
    val_accuracy = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs_range = range(1, len(loss) + 1)

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, 'bo-', label='Training accuracy')
    plt.plot(epochs_range, val_accuracy, 'ro-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'bo-', label='Training loss')
    plt.plot(epochs_range, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


show_graph(history.history)

# Evaluation
testGenerator = ImageDataGenerator(rescale=1. / 255)
testGen = testGenerator.flow_from_directory(
    os.path.join(rootPath, 'test_set'),
    target_size=(128, 128),  # Increased target size
    batch_size=32  # Added batch size
)
model.evaluate(testGen)
