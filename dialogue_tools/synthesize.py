#!/usr/bin/env python3
"""FireRedTTS-2 対話合成ツール — 簡単に推論でき、後段処理の土台(manifest)も出す。

vanilla の公開API(FireRedTTS2.generate / prepare_prompt)だけで書いてあるので、
公式 FireRedTTS2 をそのまま入れた環境でパッチ無しで動く。

出力:
  <out>.wav            アセンブル済み音声 (stereo=話者別ターン分離[2,T] / mono[1,T])
  <out>.manifest.json  ターン単位のタイミング (speaker/channel/text/onset/duration/wav)
  <out_dir>/turns/*.wav 各ターンの素片 (24k, mono)  ← postprocess が再合成に使う

manifest があると、相槌挿入やオーバーラップ化は「再生成せずタイミングを編集して
再アセンブルするだけ」の純エンジニアリングにできる(各話者が別chに分離済みのため)。

例:
  python synthesize.py \
      --pretrained_dir /path/to/ft_model \
      --script input.txt --out out/sample.wav \
      --prompt s1.wav '[S1]プロンプトの書き起こし' \
      --prompt s2.wav '[S2]プロンプトの書き起こし'

入力 script:
  *.txt   … 1行1ターン。行頭に話者タグ  例:  [S1]こんにちは。
  *.json  … {"turns":[{"speaker":"S1","text":"..."}, ...]} か [{...}, ...]
  *.jsonl … 1行1 {"speaker":"S1","text":"..."}
話者タグは S1/S2/S3/S4 または [S1].. のどちらでも可。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torchaudio
from fireredtts2.fireredtts2 import FireRedTTS2
from fireredtts2.llm.utils import Segment

SR = 24000  # FireRedTTS-2 の生成サンプリングレート
# 話者→チャンネル: S1/S3=左(0), S2/S4=右(1)
SPK2CH = {"[S1]": 0, "[S3]": 0, "[S2]": 1, "[S4]": 1}


def norm_speaker(spk: str) -> str:
    """S1 / [S1] を [S1] に正規化。"""
    spk = spk.strip()
    if not spk.startswith("["):
        spk = f"[{spk}]"
    assert spk in SPK2CH, f"未知の話者タグ: {spk} (S1..S4 のみ対応)"
    return spk


def read_turns(path: str) -> list[tuple[str, str]]:
    """script を [(speaker '[S1]', text), ...] に読む。"""
    p = Path(path)
    turns: list[tuple[str, str]] = []
    if p.suffix == ".txt":
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            assert line.startswith("["), f"行頭に [S1] 等の話者タグが必要: {line!r}"
            turns.append((norm_speaker(line[:4]), line[4:].strip()))
    else:  # .json / .jsonl
        text = p.read_text(encoding="utf-8").strip()
        if p.suffix == ".jsonl":
            objs = [json.loads(l) for l in text.splitlines() if l.strip()]
        else:
            data = json.loads(text)
            objs = data["turns"] if isinstance(data, dict) else data
        for o in objs:
            turns.append((norm_speaker(o["speaker"]), o["text"].strip()))
    assert turns, f"ターンが空: {path}"
    return turns


def synthesize(fire: FireRedTTS2, turns, prompts, temperature, topk, max_turn_ms):
    """ターン毎に generate し、各ターンの素片(24k [T])を返す。

    context 管理は generate_dialogue と同一(prompt + これまでの生成を 16k で渡す)。
    戻り値: list[dict(speaker, text, audio[T]@24k)]
    """
    prompt_segments = [
        fire.prepare_prompt(text=t, speaker=t[:4], audio_path=w) for w, t in prompts
    ]
    generated_ctx: list[Segment] = []
    out = []
    for i, (spk, body) in enumerate(turns):
        audio = fire.generate(
            text=body, speaker=spk, context=prompt_segments + generated_ctx,
            max_audio_length_ms=max_turn_ms, temperature=temperature, topk=topk,
        )  # [T]@24k
        a16 = torchaudio.functional.resample(audio.unsqueeze(0), SR, 16000)
        generated_ctx.append(Segment(text=body, speaker=spk, audio=a16))
        out.append({"speaker": spk, "text": body, "audio": audio.detach().cpu().reshape(-1)})
        print(f"  [{i:03d}] {spk} {audio.shape[-1]/SR:5.2f}s  {body[:30]}", flush=True)
    return out


def assemble(segs, layout: str):
    """素片リストを 1本のテンソルにアセンブル。

    stereo: 話者chへ順に配置したターン分離 [2,T](重なり無し)
    mono  : 全連結 [1,T]
    併せて manifest 用の turns(onset/duration 秒) を返す。
    """
    import torch

    total = sum(s["audio"].shape[0] for s in segs)
    manifest_turns = []
    if layout == "stereo":
        buf = torch.zeros(2, total)
    else:
        buf = torch.zeros(1, total)
    off = 0
    for i, s in enumerate(segs):
        n = s["audio"].shape[0]
        ch = SPK2CH[s["speaker"]] if layout == "stereo" else 0
        buf[ch, off:off + n] = s["audio"]
        manifest_turns.append({
            "index": i, "speaker": s["speaker"],
            "channel": SPK2CH[s["speaker"]], "text": s["text"],
            "onset": round(off / SR, 4), "duration": round(n / SR, 4),
        })
        off += n
    return buf, manifest_turns


def main():
    ap = argparse.ArgumentParser(description="FireRedTTS-2 dialogue synthesis (easy CLI)")
    ap.add_argument("--pretrained_dir", required=True, help="ft_model/ 等の組み立て済みモデルdir")
    ap.add_argument("--script", required=True, help="対話台本 (.txt / .json / .jsonl)")
    ap.add_argument("--out", required=True, help="出力 wav パス (.manifest.json も同名で出る)")
    ap.add_argument("--prompt", nargs=2, action="append", metavar=("WAV", "TEXT"),
                    default=[], help="声prompt (wav と '[S1]書き起こし')。S1,S2,.. の順に複数指定")
    ap.add_argument("--layout", choices=["stereo", "mono"], default="stereo",
                    help="stereo=話者別ターン分離[2,T](後段処理向け) / mono[1,T]")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max_turn_ms", type=float, default=30_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_save_turns", action="store_true", help="per-turn 素片wavを保存しない")
    args = ap.parse_args()

    turns = read_turns(args.script)
    prompts = [(w, t) for w, t in args.prompt]  # 順序=S1,S2,..
    print(f"[synthesize] {len(turns)} turns, {len(prompts)} prompts, layout={args.layout}", flush=True)

    fire = FireRedTTS2(pretrained_dir=args.pretrained_dir, gen_type="dialogue", device=args.device)
    segs = synthesize(fire, turns, prompts, args.temperature, args.topk, args.max_turn_ms)
    buf, manifest_turns = assemble(segs, args.layout)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out), buf, SR)

    # per-turn 素片(後段処理の材料)
    if not args.no_save_turns:
        import torch
        tdir = out.parent / "turns"
        tdir.mkdir(exist_ok=True)
        for mt, s in zip(manifest_turns, segs):
            wp = tdir / f"turn{mt['index']:03d}_{mt['speaker'][1:-1]}.wav"
            torchaudio.save(str(wp), s["audio"].reshape(1, -1), SR)
            mt["wav"] = os.path.relpath(wp, out.parent)

    manifest = {
        "sample_rate": SR,
        "layout": args.layout,
        "channel_map": {"[S1]": 0, "[S3]": 0, "[S2]": 1, "[S4]": 1},
        "note": "onset/duration は秒。stereo は話者別ターン分離(重なり無し)。"
                "後段処理は turns の onset を編集し turns/*.wav を再アセンブルする。",
        "turns": manifest_turns,
    }
    mpath = out.with_suffix(".manifest.json")
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out}  ({buf.shape[0]}ch, {buf.shape[1]/SR:.1f}s)\n       manifest -> {mpath}", flush=True)


if __name__ == "__main__":
    main()
