import os
import pickle
import numpy as np
from music21 import instrument, note, chord, stream
from tensorflow.keras.models import load_model

def generate_music(weights_path="model/music_generator_model.keras", output_midi="output/generated_output.mid"):
    if not os.path.exists('model/note_mapping.pkl'):
        print("Note mapping not found. Please train the model first.")
        return None

    with open('model/note_mapping.pkl', 'rb') as filePath:
        data = pickle.load(filePath)
        
    pitchnames = data['pitchnames']
    note_to_int = data['note_to_int']
    n_vocab = data['n_vocab']

    sequence_length = 100
    network_input, _ = prepare_sequences_for_generation(pitchnames, note_to_int, n_vocab, sequence_length)
    
    if len(network_input) == 0:
        print("Error preparing generation sequences.")
        return None

    start = np.random.randint(0, len(network_input)-1)
    int_to_note = {number: note for number, note in enumerate(pitchnames)}
    pattern = network_input[start]
    
    if not os.path.exists(weights_path):
        print(f"Model weights file {weights_path} not found.")
        return None

    model = load_model(weights_path)
    prediction_output = []

    print("Generating new music sequence...")
    for note_index in range(300):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:]

    create_midi(prediction_output, output_midi)
    return output_midi

def get_inference_notes():
    return ['C4', 'E4', 'G4', 'C5'] * 50

def prepare_sequences_for_generation(pitchnames, note_to_int, n_vocab, sequence_length):
    notes = get_inference_notes()
    network_input = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in if char in note_to_int])
    
    if not network_input:
        dummy_seq = [0] * sequence_length
        network_input = [dummy_seq]
        
    return network_input, n_vocab

def create_midi(prediction_output, filename):
    """Convert the output prediction notes from the LSTM into a MIDI file format."""
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.instrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.instrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    midi_stream.write('midi', fp=filename)
    print(f"MIDI file successfully saved to {filename}")