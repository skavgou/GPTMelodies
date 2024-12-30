
# -*- coding: utf-8 -*-
"""
@author: Giovanni Di Liberto
See description in the assignment instructions.
"""

from pydub import AudioSegment
import numpy as np
import simpleaudio as sa

# Define note frequencies (A4 = 440 Hz)
#NOTE_FREQUENCIES = {
#    'C': 261.63,
#    'D': 293.66,
#    'E': 329.63,
#    'F': 349.23,
#    'G': 392.00,
#    'A': 440.00,
#    'B': 493.88,
#    'R': 0  # Rest (no sound)
#}

NOTE_FREQUENCIES = {
    'C': 261.63,
    'c': 277.18,  # C#
    'D': 293.66,
    'd': 311.13,  # D#
    'E': 329.63,
    'F': 349.23,
    'f': 369.99,  # F#
    'G': 392.00,
    'g': 415.30,  # G#
    'A': 440.00,
    'a': 466.16,  # A#
    'B': 493.88,
    'R': 0     # Rest
}


# Generate a sine wave for a given frequency
def generate_sine_wave(frequency, duration_ms, sample_rate=44100, amplitude=0.5):
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), False)
    wave = 0.5 * amplitude * np.sin(2 * np.pi * frequency * t)
    wave = (wave * 32767).astype(np.int16)
    audio_segment = AudioSegment(
        wave.tobytes(), 
        frame_rate=sample_rate, 
        sample_width=wave.dtype.itemsize, 
        channels=1
    )
    return audio_segment

# Function to create a sequence of notes
def create_sequence(note_sequence, duration_ms=500):
    song = AudioSegment.silent(duration=0)
    for note in note_sequence:
        if note == 'R':  # Handle rest
            segment = AudioSegment.silent(duration=duration_ms)
        else:
            frequency = NOTE_FREQUENCIES[note]
            segment = generate_sine_wave(frequency, duration_ms)
        song += segment
    return song

# Example sequence (You can replace this with your sequence)
sequence = "A G E d C g A a D c F C d d R D R R R d D a C R a d F R A a F R R R A A R R d A d R F C E d d F F R E F d F D a D E g d a g d d R c a R R R R R C R R R R R R R R R R R R R R R R R R R R R R R R R R R R R R g R c R R a R R R G R R R a R a C F a R c F C R R R R R R R R R R R c F d a a c F c R a d R d c R R R R A R R R R d F R F d F d R f F d a a a c F D R d d d C F R C F d C a C F R d d R F B A c c a d a C R a F a d F R C F c d R D d c R C a D E F c a F F C d g c R d d c c d g D F F d c C R d ".split()
#sequence = "BDDgEARagadGCCdddEgfgcDEAGBDEFgA"

# Create the sequence
song = create_sequence(sequence, duration_ms=500)  # 500ms per note

# Save the song to a .wav file
song.export("nursery_rhyme.wav", format="wav")

# Play the .wav file using simpleaudio
wave_obj = sa.WaveObject.from_wave_file("nursery_rhyme.wav")
play_obj = wave_obj.play()
play_obj.wait_done()
