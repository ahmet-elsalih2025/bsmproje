import tensorflow as tf  # TensorFlow, bilgisayara “öğrenmeyi” öğreten ana araçtır.
from tensorflow import keras #yapay sinir ağlarını kolayca kurmamızı sağlayan kütüphanedir.
from tensorflow.keras import layers 
import numpy as np # numpy, matematiksel işlemler için kullanılır.

def train_model():
    print("Loading MNIST data...") # MNIST veri setini indirir
    # Load dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Expand dims to (28, 28, 1) for Conv2D
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print(f"x_train shape: {x_train.shape}")
    print(f"{x_train.shape[0]} train samples")
    print(f"{x_test.shape[0]} test samples")

    
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)), # giriş katmanı oluşturur
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"), # gizli katman oluşturur
            layers.MaxPooling2D(pool_size=(2, 2)),    
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)), 
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"), # çıkış katmanı oluşturur
        ]
    )

    model.summary()

    # Compile
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) 

    # eğitim
    batch_size = 128
    epochs = 5  
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    # değerlendirme
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # eğitilen modeli kaydet
    model.save("mnist_model.h5")
    print("Model saved to mnist_model.h5")

    
if __name__ == "__main__":
    train_model()
