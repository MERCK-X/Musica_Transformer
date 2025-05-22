import numpy as np
import pretty_midi
from entrenamiento import MAX_SEQ_LENGTH
from utils.midi_utils import parse_midi
from utils.audio_utils import midi_a_wav
import tensorflow as tf
import pickle

# Cargar modelo y tokenizer
model = tf.keras.models.load_model("models/transformer_model.h5")
with open("models/tokenizer.pkl", 'rb') as f:
    note_to_int, int_to_note = pickle.load(f)

def generar_melodia(nota_inicial, estilo, duracion=100):
    # Inicializar contexto
    contexto = [note_to_int[nota_inicial]]
    melodia = []
    
    for _ in range(duracion):
        x = np.array([contexto[-MAX_SEQ_LENGTH:]])
        pred = model.predict(x)[0][-1]
        nota_predicha = int_to_note[np.argmax(pred)]
        melodia.append(nota_predicha)
        contexto.append(note_to_int[nota_predicha])
    
    # Crear archivo MIDI
    midi = pretty_midi.PrettyMIDI()
    instrumento = pretty_midi.Instrument(program=0)  # Piano
    tiempo = 0.0
    for nota_str in melodia:
        if '.' in nota_str:  # Acorde
            notas = [int(n) for n in nota_str.split('.')]
            for pitch in notas:
                nota = pretty_midi.Note(
                    velocity=100, pitch=pitch, start=tiempo, end=tiempo + 0.5
                )
                instrumento.notes.append(nota)
        else:  # Nota individual
            nota = pretty_midi.Note(
                velocity=100, pitch=int(nota_str), start=tiempo, end=tiempo + 0.5
            )
            instrumento.notes.append(nota)
        tiempo += 0.5
    
    midi.instruments.append(instrumento)
    midi.write("melodia_generada.mid")
    midi_a_wav("melodia_generada.mid", f"soundfonts/{estilo}.sf2")

if __name__ == "__main__":
    nota = input("Ingresa una nota inicial (ej: C4): ")
    estilo = input("Estilo (piano/jazz/rock): ")
    generar_melodia(nota, estilo)