"""
Layer 0: The Naive Baseline ("The 8-Bit Buzzer")
------------------------------------------------
This script generates the absolute baseline sound that every audio beginner starts with:
A pure mathematical sawtooth wave.

Audio textbooks say: "A bowed violin string produces a sawtooth wave because it contains
all integer harmonics with amplitude 1/n."

Here, we implement that exact textbook math. When you run this, listen to the output
in 'audio/layer0_naive_sawtooth.wav'. You'll immediately hear why pure mathematical
waves sound like a harsh electric buzzer rather than an acoustic wooden violin!
"""

import os
import numpy as np
from scipy import signal
import soundfile as sf
import matplotlib.pyplot as plt

def generate_naive_sawtooth_note(freq: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """
    Generates a pure mathematical sawtooth wave for a single note.
    
    Formula:
        phase = 2 * pi * f0 * t
        y(t)  = scipy.signal.sawtooth(phase)
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Pure mathematical sawtooth wave between -1.0 and +1.0
    waveform = signal.sawtooth(2.0 * np.pi * freq * t)
    
    # Simple linear fade-in and fade-out to prevent speaker pops at the ends
    fade_len = int(0.01 * sample_rate)  # 10 ms
    fade_in = np.linspace(0.0, 1.0, fade_len)
    fade_out = np.linspace(1.0, 0.0, fade_len)
    waveform[:fade_len] *= fade_in
    waveform[-fade_len:] *= fade_out
    
    return waveform

def generate_melody():
    sample_rate = 44100
    
    # The universal benchmark melody: Twinkle, Twinkle, Little Star (Suzuki Violin Method)
    # Frequencies (in Hz):
    # C4: 261.63, G4: 392.00, A4: 440.00, F4: 349.23, E4: 329.63, D4: 293.66
    notes = [
        # "Twinkle, twinkle, little star"
        ("C4", 261.63, 0.5),
        ("C4", 261.63, 0.5),
        ("G4", 392.00, 0.5),
        ("G4", 392.00, 0.5),
        ("A4", 440.00, 0.5),
        ("A4", 440.00, 0.5),
        ("G4", 392.00, 1.0),
        # "How I wonder what you are"
        ("F4", 349.23, 0.5),
        ("F4", 349.23, 0.5),
        ("E4", 329.63, 0.5),
        ("E4", 329.63, 0.5),
        ("D4", 293.66, 0.5),
        ("D4", 293.66, 0.5),
        ("C4", 261.63, 1.0),
    ]
    
    audio_segments = []
    gap = np.zeros(int(0.04 * sample_rate))  # 40ms articulation gap between notes
    
    for name, freq, dur in notes:
        note_audio = generate_naive_sawtooth_note(freq, dur, sample_rate)
        audio_segments.append(note_audio)
        audio_segments.append(gap)
        
    full_audio = np.concatenate(audio_segments)
    
    # Normalize to -1.0 dB to prevent clipping
    peak = np.max(np.abs(full_audio))
    if peak > 0:
        full_audio = (full_audio / peak) * 0.85
        
    return full_audio, sample_rate

def plot_waveform_and_spectrum(sample_rate: int):
    """
    Plots the raw sawtooth wave and its harmonic spectrum to illustrate
    why it sounds harsh and buzzing.
    """
    freq = 440.0  # Concert A4
    duration = 0.02  # 20 milliseconds to clearly see the wave shape
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave_sample = signal.sawtooth(2.0 * np.pi * freq * t)
    
    # 1 second for FFT spectrum
    t_long = np.linspace(0, 1.0, sample_rate, endpoint=False)
    wave_long = signal.sawtooth(2.0 * np.pi * freq * t_long)
    fft_vals = np.abs(np.fft.rfft(wave_long))
    fft_freqs = np.fft.rfftfreq(len(wave_long), 1.0 / sample_rate)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.patch.set_facecolor('#121212')
    
    # Top Plot: Time Domain
    ax1.set_facecolor('#1e1e1e')
    ax1.plot(t * 1000, wave_sample, color='#ff5555', linewidth=2.5)
    ax1.set_title("Layer 0: Pure Mathematical Sawtooth Wave (Time Domain)", color='white', fontsize=12, pad=10)
    ax1.set_xlabel("Time (milliseconds)", color='#cccccc')
    ax1.set_ylabel("Amplitude", color='#cccccc')
    ax1.tick_params(colors='#cccccc')
    ax1.grid(True, color='#333333', linestyle='--')
    ax1.set_ylim(-1.2, 1.2)
    
    # Bottom Plot: Frequency Spectrum
    ax2.set_facecolor('#1e1e1e')
    mask = fft_freqs <= 8000
    ax2.plot(fft_freqs[mask], 20 * np.log10(fft_vals[mask] / np.max(fft_vals) + 1e-6), color='#44aaff', linewidth=1.5)
    ax2.set_title("Harmonic Spectrum (Every Integer Harmonic with 1/n Amplitude)", color='white', fontsize=12, pad=10)
    ax2.set_xlabel("Frequency (Hz)", color='#cccccc')
    ax2.set_ylabel("Magnitude (dB)", color='#cccccc')
    ax2.tick_params(colors='#cccccc')
    ax2.grid(True, color='#333333', linestyle='--')
    ax2.set_ylim(-60, 5)
    
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/layer0_waveform.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close()

if __name__ == "__main__":
    os.makedirs("audio", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    print("[VIOLIN JOURNEY] Layer 0: Synthesizing naive baseline sawtooth...")
    audio, sr = generate_melody()
    
    output_path = os.path.join("audio", "layer0_naive_sawtooth.wav")
    sf.write(output_path, audio, sr, subtype='PCM_24')
    print(f"[OK] Generated audio: {output_path}")
    
    print("[INFO] Generating waveform & spectrum plots...")
    plot_waveform_and_spectrum(sr)
    print("[OK] Generated plot: plots/layer0_waveform.png")
    print("[DONE] Layer 0 complete! Listen to the .wav file to hear the raw starting point.")
