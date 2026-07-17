#!/usr/bin/env python3
"""ターン分離ステレオ(L=S1/R=S2)+ manifest から、VAP-BC が予測する
「自然な相槌時刻」を選び、{time, listener_channel, score} の JSON を出す。
postprocess.py --bc_times_json がこれを読んで相槌を置く(乱数配置の置換)。

使い方:
  python vap_backchannel_times.py <stereo.wav> <manifest.json> <out.json> \
      [--per_min 3.1] [--thr 0.4] [--refractory 2.0]
"""
import sys, json, argparse, numpy as np, soundfile as sf, librosa
from maai import Maai, MaaiInput

ap = argparse.ArgumentParser()
ap.add_argument("wav"); ap.add_argument("manifest"); ap.add_argument("out")
ap.add_argument("--per_min", type=float, default=3.1)
ap.add_argument("--thr", type=float, default=0.4)
ap.add_argument("--refractory", type=float, default=2.0)
ap.add_argument("--mode", default="bc")
a = ap.parse_args()

man = json.load(open(a.manifest)); turns = man["turns"]
def speaker_channel_at(t):
    """時刻 t に話している話者のチャンネル(無ければ None)。"""
    for tr in turns:
        if tr["onset"] <= t < tr["onset"] + tr["duration"]:
            return tr["channel"]
    return None

# --- VAP-BC を offline 実行して p_bc 時系列を得る ---
z1, z2 = MaaiInput.Zero(), MaaiInput.Zero()
maai = Maai(mode=a.mode, lang="jp", audio_ch1=z1, audio_ch2=z2, device="cpu")
maai.reset_runtime_state()
x, sr = sf.read(a.wav)
if x.ndim == 1: x = np.stack([x, x], 1)
if sr != 16000:
    x = np.stack([librosa.resample(x[:, c], orig_sr=sr, target_sr=16000) for c in range(2)], 1); sr = 16000
L = x[:, 0].astype(np.float32); R = x[:, 1].astype(np.float32)
frame = 1600; p = []
for i in range(0, len(L)-frame, frame):
    maai.process(L[i:i+frame], R[i:i+frame])
    while True:
        try: r = maai.result_dict_queue.get_nowait()
        except Exception: break
        p.append(float(np.mean(r.get("p_bc", r.get("p_bc_react", 0.0)))))
p = np.array(p); t = np.arange(len(p)) * 0.1
dur = len(p) * 0.1

# --- ピーク(局所最大 & thr超え)を score 降順に、refractory を守って選ぶ ---
cand = [(p[i], t[i]) for i in range(1, len(p)-1) if p[i] > a.thr and p[i] >= p[i-1] and p[i] > p[i+1]]
cand.sort(reverse=True)
target = max(1, round(a.per_min * dur / 60.0))
chosen = []
for score, tt in cand:
    if len(chosen) >= target: break
    if any(abs(tt - c["time"]) < a.refractory for c in chosen): continue
    spk_ch = speaker_channel_at(tt)
    if spk_ch is None: continue          # 誰も話していない所には置かない
    chosen.append({"time": round(tt, 2), "listener_channel": 1 - spk_ch, "score": round(score, 3)})
chosen.sort(key=lambda c: c["time"])
json.dump(chosen, open(a.out, "w"), ensure_ascii=False, indent=2)
print(f"[vap-bc] {len(p)}frames/{dur:.1f}s  候補{len(cand)}  採用{len(chosen)}(目標{target}, {a.per_min}/分)")
for c in chosen: print(f"  t={c['time']:.2f}s listener_ch={c['listener_channel']} score={c['score']}")
print(f"-> {a.out}")
