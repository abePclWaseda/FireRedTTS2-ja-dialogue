# VAP-guided backchannel placement(文脈・韻律に基づく相槌配置)

`postprocess.py` の相槌配置は既定では「Zoom1頻度・分あたりレート・ターン内散布」の
**確率的(ルールベース)**配置。ここでは京大 **MaAI(VAP)** で相手発話の韻律・文脈から
**自然な相槌タイミングを予測**し、その時刻に置く(乱数配置の置換)。

## なぜ
確率配置は「量」は Zoom1 に合っても「場所」が文脈盲。VAP は相手の発話を見て
「今なら相槌が自然」という点を予測するので、**うん/はい が入る場所が的確**になる。

## 環境(FireRedTTS2 とは別 venv)
`maai` は torch を要し、`pyaudio`(マイク用)に system の portaudio.h が要るが
**オフラインwav処理には不要**。回避してCPUで動かす:
```bash
uv venv --python 3.12 .venv-vap && source .venv-vap/bin/activate
uv pip install --no-deps maai
uv pip install torch torchaudio numpy soundfile librosa einops rich matplotlib scipy transformers huggingface-hub pygame
# pyaudio を最小スタブ化(Mic/TcpMic を使わない前提)
python - <<'PY'
import maai, os, pathlib
sp=pathlib.Path(maai.__file__).parent.parent
(sp/"pyaudio.py").write_text(
  "paFloat32=1;paInt16=8;paContinue=0\n"
  "class PyAudio:\n"
  "    def __init__(self,*a,**k):pass\n"
  "    def get_device_count(self):return 0\n"
  "    def get_device_info_by_index(self,i):return {}\n"
  "    def open(self,*a,**k):raise RuntimeError('stub')\n"
  "    def terminate(self):pass\n")
print('pyaudio stub written')
PY
```
日本語 backchannel モデル: `maai-kyoto/vap_bc_jp`(`bc`, 10hz, 20s context)。初回に自動DL。

## 使い方(2ステップ)
```bash
# 1) VAP で相槌時刻を予測(素のターン分離ステレオ + manifest から)
#    出力 = [{time, listener_channel, score}]
#    ターンごとに長さ応じた本数を配分(--spacing 秒/個)。長い独白は複数・確実に付く。
python vap_backchannel_times.py out/sample.wav out/sample.manifest.json out/bc_points_vap.json \
    --min_turn 2.5 --spacing 4.0 --thr 0.4 --refractory 1.5

# 2) FireRedTTS2 側の venv に戻り、postprocess でその時刻に配置
python ../postprocess.py --manifest out/sample.manifest.json --op overlap backchannel \
    --bc_bank 0 out/bc/s1 --bc_bank 1 out/bc/s2 \
    --bc_times_json out/bc_points_vap.json --out out/sample_final_vap.wav
```
`--bc_times_json` を渡すと分あたり分布でなく **VAP予測時刻**に置く。時刻は overlap 後の
タイムラインへ(ターン相対位置を保って)自動写像される。

## 実測(混在台本, 2026-07)
ターンごと配分で17点(13.6/分, 独白多めの台本では自然)。最初の長ターン(0–10.2s)にも
t≈2.2/4.8/7.6s(listener=S2)が入り、聞き手が「うん/はい」を打つ位置に一致。
密度は `--spacing`(大きく=疎)で調整。※初期実装の「全体トップK(3.1/分)」だと前半の
やや低スコアなピークがランキングで落ち、長ターンが相槌ゼロになる不具合があった→ターンごと配分で解消。

## 補足 / 今後
- `vap_offline_bc.py` は素の feasibility 確認用(p_bc 時系列を出す)。
- ターン交代の食い込み(overlap)も VAP の SHIFT 予測(mode="vap", p_now/p_future)で
  置けるが本スクリプトは相槌のみ。次段で overlap も VAP 化可能。
- 天井: 2話者は別々合成のため「掛け合いの一体感」は full-duplex 生成(Moshi)が本命。
