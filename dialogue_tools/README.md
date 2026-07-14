# FireRedTTS-2 dialogue_tools

FireRedTTS-2 の対話合成を**簡単に扱う**ための薄いツール層。公式(vanilla)の
FireRedTTS2 をそのまま入れた環境で**パッチ不要**で動く(公開API だけ使用)。

将来の「後段エンジニアリング」— 相槌挿入・ターンずらしオーバーラップ — を
再生成せずに載せられるよう、合成時に**ターン単位のタイミング manifest** を出す。

```
台本([S1]/[S2]) + 声prompt
      │ synthesize.py
      ▼
  out.wav (stereo=話者別ターン分離[2,T] / mono)  +  out.manifest.json  +  turns/*.wav
      │ postprocess.py   ← 再生成しない。manifest と turns/*.wav だけで再アセンブル
      ▼
  overlap 済み / 相槌入り の stereo wav
```

各話者が別チャンネルに分離済みなので、後段は**音源分離不要の純タイミング編集**でよい。

## セットアップ(共同研究者向け・最短)
```bash
# fork を clone して uv で環境構築(torch cu126 込みで uv.lock を再現)
git clone https://github.com/abePclWaseda/FireRedTTS2-ja-dialogue
cd FireRedTTS2-ja-dialogue && uv sync
# 以降のコマンドは uv run python ... で実行。ft_model/ の準備はトップ README の Quickstart 参照。
```

## 1) 合成(easy inference)
```bash
python synthesize.py \
    --pretrained_dir /path/to/ft_model \
    --script input.txt \
    --out out/sample.wav \
    --prompt s1.wav '[S1]プロンプト音声の書き起こし' \
    --prompt s2.wav '[S2]プロンプト音声の書き起こし'
# 声prompt を省くとランダム話者。--layout mono でモノラル。
```
`input.txt`(1行1ターン):
```
[S1]はじめまして。今日は音声合成のテストです。
[S2]よろしくお願いします。自然に聞こえるか確認しましょう。
```
出力:
- `out/sample.wav` … stereo[2,T](L=S1/S3, R=S2/S4, 重なり無し)
- `out/sample.manifest.json` … 各ターンの speaker/channel/text/onset/duration/wav
- `out/turns/*.wav` … 各ターン素片(後段処理の材料)

## 2) 後段処理(将来公開したい機能)
```bash
# ターン境界を前倒しして自然な重なりを作る
python postprocess.py --manifest out/sample.manifest.json --op overlap \
    --overlap_ms 250 --jitter_ms 120 --out out/sample_overlap.wav
```
- **overlap**: 動作する。話者交代の各境界で次ターンの onset を前倒し。重ね幅は
  短い方ターンの `max_frac`(既定50%)でクランプ。`jitter_ms` でばらつき。
- **backchannel**: 相手の長ターン中に、**聞き手の声**で相槌を挿入。相槌素片は
  `synth_backchannel.py` で話者ごとに合成してバンク化する(下記)。挿入位置は
  overlap後の onset に整合。

相槌は聞き手の声で鳴らすので、話者ごとにバンクを作る。語彙は **LLM-jp-Zoom1 実測の
頻度**(うん 55% / はい 21% / うんうん 11% …)で重み付けし、各語を複数変種合成して
`bank.json`(重み付き)に保存する:
```bash
# S1 の声で相槌バンク(既定=Zoom1頻度語彙 x 2変種)
python synth_backchannel.py --pretrained_dir /path/to/ft_model \
    --prompt s1.wav '[S1]書き起こし' --speaker S1 --out_dir out/bc/s1
python synth_backchannel.py --pretrained_dir /path/to/ft_model \
    --prompt s2.wav '[S2]書き起こし' --speaker S2 --out_dir out/bc/s2

# overlap + backchannel をまとめて適用(既定=Zoom1寄り: overlap 350ms/±150, 相槌 3.1/分)
python postprocess.py --manifest out/sample.manifest.json --op overlap backchannel \
    --bc_bank 0 out/bc/s1 --bc_bank 1 out/bc/s2 --bc_per_min 3.1 \
    --out out/sample_final.wav
```
- `--bc_per_min`: 相槌の分あたり挿入数(ターン長に比例配分・長ターンは複数可)。
- `--overlap_ms`/`--jitter_ms`: 話者交代の重ね幅。
- **注意**: 重なり率は**ターン密度に強く依存**する(交代点にしか重なりを作れないため)。
  実測では 疎な台本(7.5ターン/分)→重なり ~5%、密な台本(~40ターン/分)→~28%。
  Zoom1 の ~20% は **ターン ~18/分**(生成側)に合わせると現在の既定で概ね到達する。

## manifest スキーマ
```json
{
  "sample_rate": 24000,
  "layout": "stereo",
  "channel_map": {"[S1]": 0, "[S3]": 0, "[S2]": 1, "[S4]": 1},
  "turns": [
    {"index":0, "speaker":"[S1]", "channel":0, "text":"...",
     "onset":0.0, "duration":1.83, "wav":"turns/turn000_S1.wav"}
  ]
}
```

## 設計メモ / 今後
- overlap と backchannel を併用する際は、backchannel の挿入位置を overlap 後の
  onset 基準にする(現状 backchannel は素の連結位置で概算)。
- backchannel を「その場合成」にする場合、聞き手の prompt 声で "うん/そうですね" 等を
  生成してバンク化 → 挿入、が自然。合成レイテンシとのトレードオフ。
- 公開形態は fork / 独立リポ / パッケージ同梱のいずれか(要決定)。
