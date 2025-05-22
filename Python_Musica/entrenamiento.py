import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Embedding, Dense
from tensorflow.keras.models import Model
import numpy as np
from utils.midi_utils import parse_midi, crear_tokenizer

# Configuración
VOCAB_SIZE = 128
MAX_SEQ_LENGTH = 50
EMBEDDING_DIM = 256
NUM_HEADS = 8
FF_DIM = 512

# Cargar datos
notas = []
for file in os.listdir("data/train"):
    notas.extend(parse_midi(f"data/train/{file}"))

note_to_int, vocabulario = crear_tokenizer(notas)

# Preparar secuencias
X = []
y = []
for i in range(len(notas) - MAX_SEQ_LENGTH):
    secuencia = notas[i:i + MAX_SEQ_LENGTH]
    objetivo = notas[i + MAX_SEQ_LENGTH]
    X.append([note_to_int[note] for note in secuencia])
    y.append(note_to_int[objetivo])

X = np.array(X)
y = np.array(y)
# Bloque Transformer personalizado
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="gelu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Modelo Transformer
inputs = Input(shape=(MAX_SEQ_LENGTH,))
x = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(inputs)
x = TransformerBlock(EMBEDDING_DIM, NUM_HEADS, FF_DIM)(x)
x = layers.GlobalAveragePooling1D()(x)
outputs = Dense(VOCAB_SIZE, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Entrenamiento
model.fit(X, y, batch_size=64, epochs=100, validation_split=0.2)
model.save("models/transformer_model.h5")