#!/usr/bin/env python3
"""ターン分離ステレオ(L=S1/R=S2)+ manifest から、VAP-BC が予測する
「自然な相槌時刻」を選び、{time, listener_channel, score} の JSON を出す。
postprocess.py --bc_times_json がこれを読んで相槌を置く(乱数配置の置換)。

選び方: 全体トップKでなく **ターンごと**。相手の各ターン内の VAP ピークから、
ターン長に応じた本数(dur/spacing, 最低1)を score 上位で採る。
→ 長い独白ターンには確実に複数付き、短い応酬ターンには付かない(自然)。

使い方:
  python vap_backchannel_times.py <stereo.wav> <manifest.json> <out.json> \
      [--min_turn 2.5] [--spacing 4.0] [--thr 0.4] [--refractory 1.5]
"""
import sys, json, argparse, numpy as np, soundfile as sf, librosa
from maai import Maai, MaaiInput

ap = argparse.ArgumentParser()
ap.add_argument("wav"); ap.add_argument("manifest"); ap.add_argument("out")
ap.add_argument("--min_turn", type=float, default=2.5, help="相槌を入れる最小ターン長(秒)")
ap.add_argument("--spacing", type=float, default=4.0, help="ターン内の相槌1個あたりの目安秒(密度)")
ap.add_argument("--thr", type=float, default=0.4, help="p_bc のピーク閾値")
ap.add_argument("--refractory", type=float, default=1.5, help="同一ターン内の相槌最小間隔(秒)")
ap.add_argument("--mode", default="bc")
a = ap.parse_args()

turns = json.load(open(a.manifest))["turns"]

# --- VAP-BC を offline 実行して p_bc 時系列を得る ---
maai = Maai(mode=a.mode, lang="jp", audio_ch1=MaaiInput.Zero(), audio_ch2=MaaiInput.Zero(), device="cpu")
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

# 全ピーク(局所最大 & thr超え)
peaks = [(t[i], float(p[i])) for i in range(1, len(p)-1)
         if p[i] > a.thr and p[i] >= p[i-1] and p[i] > p[i+1]]

# --- ターンごとに、そのターン内ピークから長さに応じた本数を採る ---
chosen = []
for tr in turns:
    if tr["duration"] < a.min_turn:
        continue                                  # 短い応酬には付けない
    lis = 1 - tr["channel"]
    o, e = tr["onset"], tr["onset"] + tr["duration"]
    inturn = sorted([(sc, tt) for (tt, sc) in peaks if o <= tt < e], reverse=True)  # score降順
    n = max(1, round(tr["duration"] / a.spacing))
    sel = []
    for sc, tt in inturn:
        if len(sel) >= n: break
        if any(abs(tt - s) < a.refractory for s in sel): continue
        sel.append(tt)
        chosen.append({"time": round(tt, 2), "listener_channel": lis, "score": round(sc, 3)})
chosen.sort(key=lambda c: c["time"])
json.dump(chosen, open(a.out, "w"), ensure_ascii=False, indent=2)
print(f"[vap-bc] {len(p)}frames/{dur:.1f}s  ピーク{len(peaks)}  採用{len(chosen)}個 ({len(chosen)/dur*60:.1f}/分)")
for c in chosen:
    print(f"  t={c['time']:5.2f}s listener_ch={c['listener_channel']} score={c['score']}")
print(f"-> {a.out}")
