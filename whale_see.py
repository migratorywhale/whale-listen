"""
whale-see: the other ear.

whale-listen counts notes. Some sounds have no notes: a whale's hum, a voice,
a siren, a slide guitar. They glide. A note detector hears nothing there.

whale-see doesn't count. It draws where the sound goes: time left to right,
pitch bottom to top (log scale), brighter = louder. An AI that can look at an
image can then *see* a glide as a slanted line, and a held note as a flat one.

Usage:
    python whale_see.py clip.mp3                 # writes clip.see.png
    python whale_see.py clip.wav -o out.png
    python whale_see.py clip.wav --min-hz 40 --max-hz 6000 --width 1700
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def load_audio(path, sr=22050):
    """Load any audio file librosa can read, as mono float32."""
    import librosa

    y, sr = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32), sr


def spectrogram_db(y, sr, n_fft=4096, hop=256):
    """Magnitude spectrogram in dB. Returns (db[freq, time], freqs, seconds_per_column)."""
    win = np.hanning(n_fft).astype(np.float32)
    pad = np.concatenate([np.zeros(n_fft // 2, np.float32), y, np.zeros(n_fft // 2, np.float32)])
    n = 1 + (len(pad) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    mag = np.abs(np.fft.rfft(pad[idx] * win, axis=1)).T
    return 20 * np.log10(mag + 1e-6), np.fft.rfftfreq(n_fft, 1 / sr), hop / sr


def render(y, sr, out_path, min_hz=40.0, max_hz=6000.0, width=1700, height=520):
    """Draw the picture. Returns (duration_seconds, out_path)."""
    from PIL import Image, ImageDraw

    db, freqs, _ = spectrogram_db(y, sr)
    dur = len(y) / sr
    max_hz = min(max_hz, sr / 2)

    # log-frequency rows (high at the top), evenly spaced time columns
    row_hz = np.geomspace(max_hz, min_hz, height)
    rows = np.clip(np.searchsorted(freqs, row_hz), 0, len(freqs) - 1)
    cols = np.clip(np.linspace(0, db.shape[1] - 1, width).astype(int), 0, db.shape[1] - 1)
    img = db[rows][:, cols]

    hi = np.percentile(img, 99.7)
    lo = max(np.percentile(img, 35), hi - 70.0)  # never show more than 70 dB below the loudest
    g = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.8
    # deep blue -> cyan -> white
    r = (255 * np.clip(2.2 * g - 1.2, 0, 1)).astype(np.uint8)
    gr = (255 * np.clip(1.6 * g - 0.3, 0, 1)).astype(np.uint8)
    b = (40 + 215 * np.clip(1.3 * g, 0, 1)).astype(np.uint8)

    left, top = 56, 8
    canvas = Image.new("RGB", (width + left + 8, height + top + 26), (8, 10, 24))
    canvas.paste(Image.fromarray(np.dstack([r, gr, b])), (left, top))
    d = ImageDraw.Draw(canvas)

    for hz in (50, 100, 200, 500, 1000, 2000, 5000, 10000):
        if min_hz <= hz <= max_hz:
            yy = top + int(np.argmin(np.abs(row_hz - hz)))
            d.line([(left, yy), (left + width, yy)], fill=(70, 80, 110), width=1)
            label = f"{hz}Hz" if hz < 1000 else f"{hz // 1000}kHz"
            d.text((4, yy - 6), label, fill=(170, 180, 210))

    step = 1 if dur <= 30 else (5 if dur <= 150 else 30)
    s = 0
    while s <= dur:
        xx = left + int(s / dur * width)
        d.line([(xx, top + height), (xx, top + height + 5)], fill=(170, 180, 210), width=1)
        if (s // step) % 2 == 0:
            d.text((xx - 6, top + height + 8), f"{s}s", fill=(170, 180, 210))
        s += step

    canvas.save(out_path)
    return dur, out_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="whale-see",
        description="Draw where a sound goes, so an AI can see glides that have no notes.",
    )
    parser.add_argument("audio", help="path to audio file (MP3, WAV, FLAC, M4A, ...)")
    parser.add_argument("-o", "--output", help="output PNG path (default: <audio>.see.png)")
    parser.add_argument("--min-hz", type=float, default=40.0, help="lowest frequency drawn (default 40)")
    parser.add_argument("--max-hz", type=float, default=6000.0, help="highest frequency drawn (default 6000)")
    parser.add_argument("--width", type=int, default=1700, help="picture width in pixels (default 1700)")
    parser.add_argument("--height", type=int, default=520, help="picture height in pixels (default 520)")
    args = parser.parse_args(argv)

    src = Path(args.audio)
    if not src.exists():
        print(f"not found: {src}", file=sys.stderr)
        return 1
    if args.min_hz <= 0 or args.max_hz <= args.min_hz:
        print("need 0 < --min-hz < --max-hz", file=sys.stderr)
        return 1

    out = Path(args.output) if args.output else src.with_suffix(".see.png")
    y, sr = load_audio(src)
    dur, out = render(y, sr, out, args.min_hz, args.max_hz, args.width, args.height)
    print(f"{src.name}: {dur:.1f}s -> {out}")
    print("flat lines are notes that stand still; slanted lines are sounds that glide.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
