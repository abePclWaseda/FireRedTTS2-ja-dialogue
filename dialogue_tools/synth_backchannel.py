#!/usr/bin/env python3
"""相槌(backchannel)素片を「その話者の声」で合成してバンク化する。

postprocess.py の backchannel は、聞き手チャンネルにこのバンクの素片を挿入する。
聞き手の声で相槌が鳴るよう、話者ごと(=声prompt ごと)に1バンク作る。

語彙と頻度は LLM-jp-Zoom1 実測(600会話)に合わせて重み付け:
  うん 55% / はい 21% / うんうん 11% / …(圧倒的に うん・はい)。
各語を複数変種(--variants)合成して同一波形の反復を避け、重みを bank.json に保存する。

例(S1の声で相槌バンクを作る):
  python synth_backchannel.py --pretrained_dir /path/to/ft_model \
      --prompt s1.wav '[S1]プロンプトの書き起こし' --speaker S1 \
      --out_dir out/bc_bank/s1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torchaudio
from fireredtts2.fireredtts2 import FireRedTTS2

SR = 24000

# LLM-jp-Zoom1 実測の相槌頻度(相対 count)。postprocess はこの重みで抽選する。
DEFAULT_LEXICON = {
    "うん": 33029,
    "はい": 12648,
    "うんうん": 6350,
    "うんうんうん": 3220,
    "はいはい": 1012,
    "なるほど": 807,
    "ああ": 685,
    "そうですね": 518,
}


def main():
    ap = argparse.ArgumentParser(description="相槌バンクを話者の声で合成(Zoom1頻度重み付き)")
    ap.add_argument("--pretrained_dir", required=True)
    ap.add_argument("--prompt", nargs=2, metavar=("WAV", "TEXT"), required=True,
                    help="この声で相槌を合成する prompt (wav と '[S1]書き起こし')")
    ap.add_argument("--speaker", default=None, help="話者タグ (S1/[S1] 等)。省略時 prompt から")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--phrases", default=None,
                    help="'語:重み,語:重み' で語彙上書き(重み省略=1)。既定は Zoom1 頻度")
    ap.add_argument("--variants", type=int, default=2,
                    help="各語を何変種合成するか(同一波形の反復回避)")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # 語彙+重みを決定
    if args.phrases:
        lexicon = {}
        for tok in args.phrases.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if ":" in tok:
                w, wt = tok.rsplit(":", 1)
                lexicon[w.strip()] = float(wt)
            else:
                lexicon[tok] = 1.0
    else:
        lexicon = dict(DEFAULT_LEXICON)
    total = sum(lexicon.values())

    wav, ptext = args.prompt
    spk = args.speaker or ptext[:4]
    if not spk.startswith("["):
        spk = f"[{spk}]"

    fire = FireRedTTS2(pretrained_dir=args.pretrained_dir, gen_type="dialogue", device=args.device)
    prompt_seg = fire.prepare_prompt(text=ptext, speaker=ptext[:4], audio_path=wav)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bank = []  # [{wav, text, weight}]
    for i, (ph, cnt) in enumerate(lexicon.items()):
        share = cnt / total / max(1, args.variants)  # 変種で重みを分割
        for v in range(args.variants):
            audio = fire.generate(
                text=ph, speaker=spk, context=[prompt_seg],
                max_audio_length_ms=4_000, temperature=args.temperature, topk=args.topk,
            )  # [T]@24k
            name = f"bc{i:02d}_v{v}.wav"
            torchaudio.save(str(out / name), audio.detach().cpu().reshape(1, -1), SR)
            bank.append({"wav": name, "text": ph, "weight": share})
            print(f"  {spk} '{ph}' v{v} w={share:.3f} -> {name} ({audio.shape[-1]/SR:.2f}s)", flush=True)

    (out / "bank.json").write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {len(bank)} clips ({len(lexicon)} 語 x {args.variants} 変種) -> {out}/bank.json", flush=True)


if __name__ == "__main__":
    main()
