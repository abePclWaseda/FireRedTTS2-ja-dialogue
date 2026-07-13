#!/usr/bin/env python3
"""相槌(backchannel)素片を「その話者の声」で合成してバンク化する。

postprocess.py の backchannel は、聞き手チャンネルにこのバンクの素片を挿入する。
聞き手の声で相槌が鳴るよう、話者ごと(=声prompt ごと)に1バンク作る。

例(S1の声で相槌バンクを作る):
  python synth_backchannel.py --pretrained_dir /path/to/ft_model \
      --prompt s1.wav '[S1]プロンプトの書き起こし' --speaker S1 \
      --out_dir out/bc_bank/s1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torchaudio
from fireredtts2.fireredtts2 import FireRedTTS2

SR = 24000
DEFAULT_PHRASES = ["うん", "うんうん", "そうですね", "なるほど", "へえ", "はい", "ええ"]


def main():
    ap = argparse.ArgumentParser(description="相槌バンクを話者の声で合成")
    ap.add_argument("--pretrained_dir", required=True)
    ap.add_argument("--prompt", nargs=2, metavar=("WAV", "TEXT"), required=True,
                    help="この声で相槌を合成する prompt (wav と '[S1]書き起こし')")
    ap.add_argument("--speaker", default=None, help="話者タグ (S1/[S1] 等)。省略時 prompt から")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--phrases", default=None, help="カンマ区切りの相槌語。省略で既定セット")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    wav, ptext = args.prompt
    spk = args.speaker or ptext[:4]
    if not spk.startswith("["):
        spk = f"[{spk}]"
    phrases = (args.phrases.split(",") if args.phrases else DEFAULT_PHRASES)

    fire = FireRedTTS2(pretrained_dir=args.pretrained_dir, gen_type="dialogue", device=args.device)
    prompt_seg = fire.prepare_prompt(text=ptext, speaker=ptext[:4], audio_path=wav)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, ph in enumerate(phrases):
        ph = ph.strip()
        if not ph:
            continue
        audio = fire.generate(
            text=ph, speaker=spk, context=[prompt_seg],
            max_audio_length_ms=4_000, temperature=args.temperature, topk=args.topk,
        )  # [T]@24k
        wp = out / f"bc{i:02d}.wav"
        torchaudio.save(str(wp), audio.detach().cpu().reshape(1, -1), SR)
        print(f"  {spk} '{ph}' -> {wp} ({audio.shape[-1]/SR:.2f}s)", flush=True)
    print(f"[done] {len([p for p in phrases if p.strip()])} clips -> {out}", flush=True)


if __name__ == "__main__":
    main()
