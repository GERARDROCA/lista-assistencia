import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directorio de datos
data = 'aquí va la direcció de la carpeta' ######################
batch_size = 32

# Data augmentation y preproces
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Dividir el conjunt de dades
split_ratio = 0.8

# Generar el conjunt d'entrenament
train_generator = train_datagen.flow_from_directory(
    data, 
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',  
    shuffle=True
)

# Generar el conjunt de validació
validation_generator = train_datagen.flow_from_directory(
    data, 
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation', 
    shuffle=True
)

# Construir el model 
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3))) 
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(train_generator.num_classes, activation='softmax'))

# Compilar el model
model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=-1),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar el model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=40, #èpoques
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Guardar el modelo entrenado
model.save('aqué va el nom del model.h5') ###################

# Obtener el historial de entrenamiento
training_accuracy = history.history['accuracy']
validation_accuracy = history.history.get('val_accuracy', history.history.get('val_acc', []))
training_loss = history.history['loss']
validation_loss = history.history.get('val_loss', [])
epochs = range(1, len(training_accuracy) + 1)

# Graficar la precisión de entrenamiento y validación


plt.figure(figsize=(12, 4))

#precisió
plt.subplot(1, 2, 1)
plt.plot(epochs, training_accuracy, 'b-', label="Precisió de l'entrenamiento")
if validation_accuracy:
    plt.plot(epochs, validation_accuracy, 'r-', label='Precisió de validació')
plt.title("Precisió de l'entrenamient i validació")
plt.xlabel('Èpoques')
plt.ylabel('Precisió')
plt.legend()

#pèrdua
plt.subplot(1, 2, 2)
plt.plot(epochs, training_loss, 'b-', label="Pèrdua de l'entrenamient")
if validation_loss:
    plt.plot(epochs, validation_loss, 'r-', label='Pèrdua de validació')
plt.title("Pèrdua de l'entrenamient  validació")
plt.xlabel('Èpoques')
plt.ylabel('Pèrdua')
plt.legend()

plt.tight_layout() 
plt.show()  # Mostrar la gráfica