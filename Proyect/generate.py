import numpy as np
import pretty_midi
import os
from pathlib import Path
from tensorflow.keras.models import load_model

def generate_melody(model, start_notes, sequence_length, n_notes=50, temperature=0.7):
    """Genera una melodía usando el modelo entrenado"""
    pitch_to_int = np.load('pitch_to_int.npy', allow_pickle=True).item()
    int_to_pitch = np.load('int_to_pitch.npy', allow_pickle=True).item()
    
    # Asegurar que start_notes tenga la longitud correcta
    if len(start_notes) < sequence_length:
        start_notes = [start_notes[0]] * (sequence_length - len(start_notes)) + start_notes
    
    pattern = [pitch_to_int[p] for p in start_notes[-sequence_length:]]
    output = []
    
    for _ in range(n_notes):
        input_seq = np.reshape(pattern[-sequence_length:], (1, sequence_length, 1))
        input_seq = input_seq / float(len(pitch_to_int))
        
        prediction = model.predict(input_seq, verbose=0)[0]
        prediction = np.log(prediction) / temperature
        prediction = np.exp(prediction) / np.sum(np.exp(prediction))
        
        idx = np.random.choice(len(prediction), p=prediction)
        result = int_to_pitch[idx]
        output.append(result)
        pattern.append(idx)
    
    return output

def create_midi(melody, output_file='output.mid', instrument_name='Acoustic Grand Piano', tempo_bpm=120):
    """Crea un archivo MIDI a partir de la melodía generada"""
    midi = pretty_midi.PrettyMIDI()
    instrument_program = pretty_midi.instrument_name_to_program(instrument_name)
    midi_instrument = pretty_midi.Instrument(program=instrument_program)
    
    duration = 0.5  # media nota
    time = 0.0
    
    for pitch in melody:
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=time,
            end=time + duration
        )
        midi_instrument.notes.append(note)
        time += duration
    
    midi.instruments.append(midi_instrument)
    midi.write(output_file)
    return output_file

def save_notes_to_txt(melody, txt_path):
    """Guarda las notas en un archivo TXT con información detallada"""
    int_to_pitch = np.load('int_to_pitch.npy', allow_pickle=True).item()
    
    with open(txt_path, 'w') as f:
        f.write("=== MELODÍA GENERADA ===\n\n")
        f.write(f"Número de notas: {len(melody)}\n")
        f.write("Notas MIDI: " + ", ".join(map(str, melody)) + "\n\n")
        
        f.write("=== DETALLES POR NOTA ===\n")
        for i, pitch in enumerate(melody, 1):
            note_name = pretty_midi.note_number_to_name(pitch)
            f.write(f"Nota {i}: MIDI {pitch} ({note_name})\n")

if __name__ == "__main__":
    try:
        # Configuración de rutas
        output_dir = Path("C:/Users/AlexisOE/Desktop/Proyect/")
        midi_path = output_dir / "output.mid"
        txt_path = output_dir / "notas_melodia.txt"
        
        # Cargar modelo
        model = load_model('melody_generator.keras')
        
        # Generar melodía
        melody = generate_melody(
            model,
            start_notes=[60, 62, 64, 67],  # Do, Re, Mi, Sol
            sequence_length=32,
            n_notes=50,
            temperature=0.7
        )
        
        # Guardar archivos
        create_midi(melody, output_file=str(midi_path))
        save_notes_to_txt(melody, txt_path)
        
        # Verificación
        print("Generación completada:")
        print(f"- MIDI: {midi_path}")
        print(f"- TXT: {txt_path}")
        print(f"\nVista previa de notas:\n{melody[:10]}...")
        
    except Exception as e:
        print(f"Error: {str(e)}")