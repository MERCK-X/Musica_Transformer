import os
import pretty_midi
import numpy as np
from music21 import converter, instrument, note, chord, stream
import matplotlib.pyplot as plt

def extract_notes(midi_path):
    """Extrae notas y acordes de un archivo MIDI"""
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append({
                'pitch': note.pitch,
                'velocity': note.velocity,
                'start': note.start,
                'end': note.end
            })
    
    return notes

def create_midi_dataset(midi_folder):
    """Procesa todos los archivos MIDI en una carpeta"""
    dataset = []
    
    for file in os.listdir(midi_folder):
        if file.endswith('.mid'):
            try:
                midi_path = os.path.join(midi_folder, file)
                notes = extract_notes(midi_path)
                dataset.extend(notes)
                print(f"Procesado: {file} - {len(notes)} notas")
            except Exception as e:
                print(f"Error procesando {file}: {str(e)}")
    
    return dataset

def prepare_sequences(notes, sequence_length=32):
    """Prepara secuencias para el modelo"""
    pitch_names = sorted(set(note['pitch'] for note in notes))
    pitch_to_int = dict((pitch, number) for number, pitch in enumerate(pitch_names))
    
    network_input = []
    network_output = []
    
    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        
        network_input.append([pitch_to_int[note['pitch']] for note in sequence_in])
        network_output.append(pitch_to_int[sequence_out['pitch']])
    
    return (np.array(network_input), 
            np.array(network_output),
            pitch_to_int,
            dict((number, pitch) for number, pitch in enumerate(pitch_names)))

def visualize_notes(notes, title="Distribución de Notas"):
    """Visualiza la distribución de notas"""
    pitches = [note['pitch'] for note in notes]
    plt.hist(pitches, bins=range(min(pitches), max(pitches) + 1))
    plt.title(title)
    plt.xlabel("Nota MIDI")
    plt.ylabel("Frecuencia")
    plt.show()

if __name__ == "__main__":
    midi_folder = "C:/Users/AlexisOE/Desktop/Proyect/midi_files"
    output_dir = os.path.dirname(os.path.abspath(__file__))  # Obtiene la ruta del script
    
    notes = create_midi_dataset(midi_folder)
    
    if notes:
        visualize_notes(notes)
        network_input, network_output, pitch_to_int, int_to_pitch = prepare_sequences(notes)
        print(f"Secuencias de entrada shape: {network_input.shape}")
        print(f"Secuencias de salida shape: {network_output.shape}")
        
        try:
            # Guardar con rutas absolutas y verificando
            np.save(os.path.join(output_dir, 'network_input.npy'), network_input)
            np.save(os.path.join(output_dir, 'network_output.npy'), network_output)
            np.save(os.path.join(output_dir, 'pitch_to_int.npy'), pitch_to_int)
            np.save(os.path.join(output_dir, 'int_to_pitch.npy'), int_to_pitch)
            
            print(f"Archivos guardados en: {output_dir}")
            print("Contenido del directorio:")
            print(os.listdir(output_dir))  # Muestra qué hay realmente en la carpeta
        except Exception as e:
            print(f"Error al guardar archivos: {str(e)}")
    else:
        print("No se encontraron archivos MIDI válidos.")

    print("Notas únicas aprendidas:", len(np.load('pitch_to_int.npy', allow_pickle=True).item()))
    print("Ejemplo de secuencia:", np.load('network_input.npy')[0])
    print("Nota objetivo correspondiente:", np.load('network_output.npy')[0])