import os
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

def get_notes_from_midi(data_dir="data"):
    """Extract notes and chords from all MIDI files in the directory."""
    notes = []
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Directory '{data_dir}' created. Please add some .mid files there before training.")
        return notes

    files = [f for f in os.listdir(data_dir) if f.endswith(".mid")]
    if not files:
        print(f"No MIDI files found in '{data_dir}'. Using synthetic sample data fallback for demonstration.")
        return []

    for file in files:
        file_path = os.path.join(data_dir, file)
        print(f"Parsing {file}...")
        try:
            midi = converter.parse(file_path)
            parts = instrument.partitionByInstrument(midi)
            notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes
            
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"Error parsing {file}: {e}")
            
    return notes

def prepare_sequences(notes, n_vocab, sequence_length=100):
    """Prepare input sequences and corresponding outputs for the LSTM network."""
    pitchnames = sorted(set(item for item in notes))
    note_to_int = {note: number for number, note in enumerate(pitchnames)}

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    
    # Reshape input into format compatible with LSTM layers
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # Normalize input float values between 0 and 1
    normalized_input = normalized_input / float(n_vocab)
    
    network_output = to_categorical(network_output)

    return (normalized_input, network_output, pitchnames, note_to_int)

def create_network(network_input, n_vocab):
    """Build the deep LSTM network architecture."""
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

def train_model():
    notes = get_notes_from_midi()
    
    # Fallback dummy notes if directory is empty so script runs smoothly
    if not notes:
        notes = ['C4', 'E4', 'G4', 'C5'] * 100 

    n_vocab = len(set(notes))
    print(f"Total unique notes (vocabulary size): {n_vocab}")

    sequence_length = 100
    if len(notes) <= sequence_length:
        print("Not enough notes to create sequences. Add more MIDI files.")
        return

    network_input, network_output, pitchnames, note_to_int = prepare_sequences(notes, n_vocab, sequence_length)

    # Save mapping dictionaries
    os.makedirs("model", exist_ok=True)
    with open('model/note_mapping.pkl', 'wb') as filePath:
        pickle.dump({'pitchnames': pitchnames, 'note_to_int': note_to_int, 'n_vocab': n_vocab}, filePath)

    model = create_network(network_input, n_vocab)
    
    filepath = "model/weights-improvement-{epoch:02d}-{loss:.4f}.keras"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    
    print("Starting model training...")
    model.fit(network_input, network_output, epochs=20, batch_size=64, callbacks=[checkpoint])
    model.save("model/music_generator_model.keras")
    print("Training complete! Model saved successfully.")

if __name__ == '__main__':
    train_model()