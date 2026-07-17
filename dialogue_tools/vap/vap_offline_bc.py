#!/usr/bin/env python3
"""我々のターン分離ステレオ(L=S1/R=S2)を MaAI VAP-BC に通し、
相槌(continuer=うん/はい)の自然なタイミングを予測する feasibility テスト。
出力: 各時刻の p_bc_react(継続相槌) / p_bc_emo(感情相槌) / p_now(発話中確率)。
"""
import sys, numpy as np, soundfile as sf, librosa
from maai import Maai, MaaiInput

wav = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "bc_2type"

# ダミー入力で構築(start()は使わず process() を手動駆動)
try:
     z1, z2 = MaaiInput.Zero(), MaaiInput.Zero()
except TypeError:
    z1 = MaaiInput.Wav(wav, use_channel=0); z2 = MaaiInput.Wav(wav, use_channel=1)
print(f"[init] Maai(mode={mode}, lang=jp, cpu) 構築中(初回はモデルDL)...", flush=True)
maai = Maai(mode=mode, lang="jp", audio_ch1=z1, audio_ch2=z2, device="cpu")
maai.reset_runtime_state()
print("[init] done", flush=True)

x, sr = sf.read(wav)
if x.ndim == 1:
    x = np.stack([x, x], axis=1)
if sr != 16000:
    x = np.stack([librosa.resample(x[:, c], orig_sr=sr, target_sr=16000) for c in range(2)], axis=1)
    sr = 16000
L = x[:, 0].astype(np.float32); R = x[:, 1].astype(np.float32)
print(f"[audio] {wav}  {len(L)/sr:.1f}s @16k  (L=S1, R=S2)", flush=True)

frame = 1600  # 100ms @16k (frame_rate=10)
rows = []
first_keys = None
for i in range(0, len(L) - frame, frame):
    maai.process(L[i:i+frame], R[i:i+frame])
    while True:
        try:
            r = maai.result_dict_queue.get_nowait()
        except Exception:
            break
        if first_keys is None:
            first_keys = list(r.keys()); print("[result keys]", first_keys, flush=True)
        rows.append(r)

t = np.arange(len(rows)) * 0.1
def series(k):
    return np.array([float(np.mean(r[k])) if k in r else 0.0 for r in rows])
react = series("p_bc_react"); emo = series("p_bc_emo"); pbc = series("p_bc")
print(f"[frames] {len(rows)}  react.max={react.max():.2f} emo.max={emo.max():.2f} pbc.max={pbc.max():.2f}", flush=True)
sig = react if react.max() > 0 else pbc
peaks = [round(float(t[i]), 1) for i in range(1, len(sig)-1)
         if sig[i] > 0.4 and sig[i] >= sig[i-1] and sig[i] > sig[i+1]]
print(f"[相槌候補時刻(s), 継続相槌 p>0.4 のピーク] {peaks[:50]}", flush=True)
