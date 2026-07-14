#!/usr/bin/env python3
"""synthesize.py の manifest+素片を後段処理する(再生成しない純エンジニアリング)。

各話者が別チャンネルに分離済みなので、タイミングを編集して再アセンブルするだけで
オーバーラップや相槌を実現できる。

  overlap     : 隣接ターン(話者交代箇所)の onset を前倒しし、境界に自然な重なりを作る。
  backchannel : 相手の長ターンの最中に、聞き手chへ短い相槌を差し込む。

使い方:
  python postprocess.py --manifest out/sample.manifest.json --op overlap \
      --overlap_ms 250 --jitter_ms 120 --out out/sample_overlap.wav
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torchaudio


def load(manifest_path: str):
    mp = Path(manifest_path)
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    base = mp.parent
    sr = manifest["sample_rate"]
    audios = []
    for t in manifest["turns"]:
        wav = base / t["wav"]
        a, wsr = torchaudio.load(str(wav))
        if wsr != sr:
            a = torchaudio.functional.resample(a, wsr, sr)
        audios.append(a.reshape(-1))  # [T] mono
    return manifest, audios, sr, base


def assemble_stereo(turns, audios, onsets_samp, sr):
    total = max(o + a.shape[0] for o, a in zip(onsets_samp, audios))
    buf = torch.zeros(2, total)
    for t, a, o in zip(turns, audios, onsets_samp):
        ch = t["channel"]
        n = a.shape[0]
        buf[ch, o:o + n] += a  # 別ch同士の重なりは独立。同ch連結は加算=無害(butt-join)
    return buf


def add_overlap(turns, audios, sr, overlap_ms=250.0, jitter_ms=0.0,
                seed=0, max_frac=0.5):
    """話者交代の各境界で、次ターンの onset を overlap 分だけ前倒しする。

    overlap は min(overlap_ms, 前後ターンの max_frac) にクランプ(潰しすぎ防止)。
    同一話者が連続する箇所は重ねない(同ch衝突を避ける)。
    戻り値: onsets_samp(list[int])
    """
    rng = random.Random(seed)
    onsets = [0]
    for i in range(1, len(turns)):
        prev_end = onsets[i - 1] + audios[i - 1].shape[0]
        ov = 0.0
        if turns[i]["speaker"] != turns[i - 1]["speaker"]:
            ov = overlap_ms
            if jitter_ms:
                ov += rng.uniform(-jitter_ms, jitter_ms)
            ov = max(0.0, ov)
            cap = max_frac * min(turns[i]["duration"], turns[i - 1]["duration"]) * 1000.0
            ov = min(ov, cap)
        onset = prev_end - int(ov * sr / 1000.0)
        onsets.append(max(0, onset))
    return onsets


def load_banks(bank_specs):
    """[[ch, dir], ...] を {ch: {"clips":[T...], "weights":[...]}} に読む。

    dir/bank.json があれば Zoom1 頻度重みを使う。無ければ *.wav を均等重み。
    """
    banks = {}
    for ch, d in bank_specs or []:
        ch = int(ch); d = Path(d)
        clips, weights = [], []
        bj = d / "bank.json"
        if bj.exists():
            for e in json.loads(bj.read_text(encoding="utf-8")):
                wav = d / e["wav"]
                if wav.exists():
                    clips.append(torchaudio.load(str(wav))[0].reshape(-1))
                    weights.append(float(e.get("weight", 1.0)))
        else:
            for p in sorted(d.glob("*.wav")):
                clips.append(torchaudio.load(str(p))[0].reshape(-1))
                weights.append(1.0)
        if clips:
            banks[ch] = {"clips": clips, "weights": weights}
        else:
            print(f"[backchannel] ch{ch}: {d} に素片が無い。", flush=True)
    return banks


def inject_backchannels(turns, audios, onsets_samp, banks, bc_per_min=3.0,
                        min_turn_s=1.5, seed=0, gain=0.6, max_per_turn=4):
    """相手のターン中に、聞き手chへ相槌を挿入する(Zoom1頻度・分あたりレート)。

    banks = {ch: {"clips":[T...], "weights":[...]}}。話者ch の相手(聞き手)ch の
    声バンクから Zoom1 頻度重みで抽選。挿入数は「ターン長 × bc_per_min」で決め
    (長ターンほど多い/複数可)、ターン内に散らして置く。onsets_samp は最終タイム
    ライン上の各ターン開始位置(overlap後でも整合)。
    戻り値: list[dict(channel, onset_samp, audio[T])]
    """
    if not banks:
        print("[backchannel] バンク未指定のためスキップ。", flush=True)
        return []
    rng = random.Random(seed)
    inserts = []
    for t, a, onset in zip(turns, audios, onsets_samp):
        n = a.shape[0]
        listener_ch = 1 - t["channel"]
        if t["duration"] < min_turn_s or listener_ch not in banks:
            continue
        # 期待挿入数 = 分あたりレート × ターン長(秒)。端数は確率で切り上げ。
        expected = bc_per_min / 60.0 * t["duration"]
        k = int(expected) + (1 if rng.random() < (expected - int(expected)) else 0)
        k = min(k, max_per_turn)
        bank = banks[listener_ch]
        for j in range(k):
            clip = rng.choices(bank["clips"], weights=bank["weights"], k=1)[0] * gain
            # ターン内に散らす: (j+1)/(k+1) を中心に少しジッタ、端(0.1..0.9)に収める
            base = (j + 1) / (k + 1)
            frac = min(0.9, max(0.1, base + rng.uniform(-0.08, 0.08)))
            pos = onset + int(n * frac)
            inserts.append({"channel": listener_ch, "onset_samp": pos, "audio": clip})
    print(f"[backchannel] {len(inserts)} 個挿入(Zoom1頻度・聞き手の声)。", flush=True)
    return inserts


def main():
    ap = argparse.ArgumentParser(description="FireRedTTS-2 dialogue post-processing")
    ap.add_argument("--manifest", required=True, help="synthesize.py が出した *.manifest.json")
    ap.add_argument("--out", required=True, help="出力 wav")
    ap.add_argument("--op", nargs="+", choices=["overlap", "backchannel"], default=["overlap"])
    ap.add_argument("--overlap_ms", type=float, default=350.0, help="話者交代の重ね幅(Zoom1寄り)")
    ap.add_argument("--jitter_ms", type=float, default=150.0, help="重ね幅のばらつき(±)")
    ap.add_argument("--max_frac", type=float, default=0.5, help="重ね幅の上限=短い方ターンの割合")
    ap.add_argument("--bc_bank", nargs=2, action="append", metavar=("CH", "DIR"),
                    default=[], help="相槌バンク: チャンネル(0=L/1=R) とその声の素片dir。"
                                     "例: --bc_bank 0 bc/s1 --bc_bank 1 bc/s2")
    ap.add_argument("--bc_per_min", type=float, default=3.0, help="相槌の分あたり挿入数(Zoom1≈3.1)")
    ap.add_argument("--bc_min_turn_s", type=float, default=1.5, help="相槌を入れる最小ターン長(秒)")
    ap.add_argument("--bc_max_per_turn", type=int, default=4, help="1ターンに入れる相槌の上限")
    ap.add_argument("--bc_gain", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest, audios, sr, base = load(args.manifest)
    turns = manifest["turns"]
    assert manifest["layout"] == "stereo", "後段処理は layout=stereo が前提"

    onsets = [t["onset"] * 0 for t in turns]  # placeholder, 下で確定
    if "overlap" in args.op:
        onsets = add_overlap(turns, audios, sr, args.overlap_ms, args.jitter_ms,
                             args.seed, args.max_frac)
    else:
        # overlap しない場合は元の連結位置
        onsets, off = [], 0
        for a in audios:
            onsets.append(off)
            off += a.shape[0]

    buf = assemble_stereo(turns, audios, onsets, sr)

    if "backchannel" in args.op:
        banks = load_banks(args.bc_bank)
        inserts = inject_backchannels(turns, audios, onsets, banks,
                                      args.bc_per_min, args.bc_min_turn_s, args.seed,
                                      args.bc_gain, args.bc_max_per_turn)
        if inserts:
            need = max(buf.shape[1], max(i["onset_samp"] + i["audio"].shape[0] for i in inserts))
            if need > buf.shape[1]:
                buf = torch.nn.functional.pad(buf, (0, need - buf.shape[1]))
            for ins in inserts:
                o, a, ch = ins["onset_samp"], ins["audio"], ins["channel"]
                buf[ch, o:o + a.shape[0]] += a

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out), buf, sr)
    print(f"[done] {out}  ({buf.shape[0]}ch, {buf.shape[1]/sr:.1f}s)  ops={args.op}", flush=True)


if __name__ == "__main__":
    main()
