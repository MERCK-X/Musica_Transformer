import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import tensorflow as tf

# Verificar archivos
required_files = ['network_input.npy', 'network_output.npy', 'pitch_to_int.npy', 'int_to_pitch.npy']
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    print(f"Error: Faltan los siguientes archivos: {missing_files}")
    exit(1)
else:
    print("Todos los archivos requeridos están presentes. Continuando...")

def build_model(input_shape, n_vocab):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        
        LSTM(128),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        Dense(n_vocab, activation='softmax')
    ])
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model

def train_model(X, y, epochs=10, batch_size=128):  # Reducido a 30 épocas
    n_vocab = len(np.unique(y))
    input_shape = (X.shape[1], 1)
    
    model = build_model(input_shape, n_vocab)
    model.summary()
    
    callbacks = [
        ModelCheckpoint(
            'model_weights.h5',
            monitor='loss',
            save_best_only=True,
            mode='min'
        ),
        EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001
        )
    ]
    
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        validation_split=0.1,
        shuffle=True
        # Se eliminaron workers y use_multiprocessing
    )
    
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Evolución de la Pérdida')
    plt.ylabel('Pérdida')
    plt.xlabel('Época')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return model

if __name__ == "__main__":
    # Limitar datos a las primeras 50,000 muestras
    X = np.load('network_input.npy')[:50000].astype('float32')
    y = np.load('network_output.npy')[:50000].astype('int32')
    
    model = train_model(X, y, epochs=10)
    model.save('melody_generator.keras')