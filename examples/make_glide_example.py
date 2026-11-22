"""Regenerate examples/glide.wav: three plucked notes that stand still, then two glides.
A note detector finds the plucks. Only whale-see finds the glides.
Pure synthesis, no samples. Usage: python examples/make_glide_example.py
"""
import wave
from pathlib import Path

import numpy as np

SR = 22050


def pluck(freq, dur=0.55):
    t = np.arange(int(dur * SR)) / SR
    x = sum(a * np.sin(2 * np.pi * freq * k * t) * np.exp(-t / (0.22 / k**0.7)) for k, a in ((1, 1.0), (2, 0.35), (3, 0.12)))
    x[:80] *= np.linspace(0, 1, 80)
    x[-600:] *= np.linspace(1, 0, 600)  # fade the tail so it doesn't click
    return x


def glide(f0, f1, dur=0.6):
    n = int(dur * SR)
    u = np.arange(n) / n
    f = f0 * (f1 / f0) ** (1 - (1 - u) ** 2)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR)
    edge = int(0.06 * SR)
    env = np.ones(n)
    env[:edge] = np.linspace(0, 1, edge)
    env[-edge:] = np.linspace(1, 0, edge)
    return x * env


def main():
    buf = np.zeros(int(4.4 * SR))

    def place(x, at, gain=1.0):
        i = int(at * SR)
        buf[i : i + len(x)] += gain * x

    for i, hz in enumerate((261.63, 329.63, 392.0)):  # C4 E4 G4, standing still
        place(pluck(hz), 0.2 + 0.45 * i)
    place(glide(400.0, 900.0), 1.9)  # one slide up
    place(glide(1000.0, 450.0), 2.9)  # one slide down
    buf = buf / np.max(np.abs(buf)) * 0.8
    out = Path(__file__).with_name("glide.wav")
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((buf * 32767).astype("<i2").tobytes())
    print(out)


if __name__ == "__main__":
    main()
