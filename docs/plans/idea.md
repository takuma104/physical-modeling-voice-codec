# 先行研究の調査

**User:** Anonymous  
**Created:** 11/11/2026 19:55:19  
**Updated:** 11/11/2026 20:00:20  
**Exported:** 11/12/2026 1:36:53  
**Link:** [https://chatgpt.com/c/6af44996-aba4-83e8-9122-5e07000a4cd8](https://chatgpt.com/c/6af44996-aba4-83e8-9122-5e07000a4cd8)  

## Prompt:
2026/11/11 19:56:45

発話の仕組みに人間の声帯、口、舌あたりを模擬したNeural Coderというのを考えたんですが、先行研究はありますか？

## Response:
2026/11/11 19:56:55 · gpt-6-thinking

先行研究を、①声帯・声道を物理的に模擬する方式、②舌・唇などの運動パラメータをニューラルネットで扱う方式、③それらを音声Codecの潜在表現として使う方式、に分けて調べます。特に、波形から発話器官の状態を推定して再合成できるか、という観点を重視します。

> **整理研究方向**
>
> 物理モデルとニューラルデコーダーを組み合わせる構想を整理している。独自性、量子化後のビットレート、解剖学的同定性、子音・韻律の再現性、データ不足、リアルタイム性を評価し、段階的な実装案へ落とし込む。
>
> **整理了研究方向**
>
> 
>
> **検証点を絞る**
>
> ARTI-6の品質評価と入力条件を確認し、6次元制御だけで自然性を説明しないよう注意する。声帯・声道の物理性、話者埋め込み、量子化、比較対象、実装順序を明確に分けて論じる。
>
> **設計を具体化する**
>
> 物理音響シミュレーター、話者埋め込み、F0・励振、軽量残差デコーダーを組み合わせる構成を検討している。ビットレート計算、残差依存の抑制、逆推定の曖昧性、計算量、データ不足を主要な課題として整理する。
>
> **検証点を絞り設計を具体化した**
>
> 
>
> **23件のウェブサイトを検索しました**
>
> 
>
> 2m 13s考えました

あります。しかも、かなり近い発想の研究がすでに進んでいます。

この分野は **Articulatory Speech Synthesis（調音音声合成）** と呼ばれ、特にニューラルネットで音声を圧縮・復元する方向では、**Articulatory Speech Coding（調音ベース音声符号化）** という研究があります。

通常のNeural Codecが音響特徴を潜在表現として学習するのに対し、考えられている方式は、声帯・舌・唇・顎・軟口蓋などの状態を潜在表現にして音声を再構成するものだと思います。

想定されるアーキテクチャ

```svg
<svg viewBox="0 0 340 174" width="100%" xmlns="http://www.w3.org/2000/svg"><rect x="93" y="4" width="154" height="30" rx="7" fill="#64748b" fill-opacity="0.13" stroke="#94a3b8" /><text x="170" y="23" font-size="12" fill="currentColor" text-anchor="middle">入力音声 Waveform</text><path d="M170 34 V51" stroke="#94a3b8" stroke-width="1.5" /><path d="M166 47 L170 53 L174 47" fill="none" stroke="#94a3b8" stroke-width="1.5" /><rect x="93" y="54" width="154" height="30" rx="7" fill="#64748b" fill-opacity="0.13" stroke="#94a3b8" /><text x="170" y="73" font-size="12" fill="currentColor" text-anchor="middle">Neural Encoder</text><path d="M170 84 V100" stroke="#94a3b8" stroke-width="1.5" /><path d="M166 96 L170 102 L174 96" fill="none" stroke="#94a3b8" stroke-width="1.5" /><rect x="18" y="103" width="304" height="32" rx="7" fill="#0d9488" fill-opacity="0.12" stroke="#0d9488" /><text x="170" y="123" font-size="11.5" fill="currentColor" text-anchor="middle">舌・唇・顎・声帯・呼気などの運動パラメータ</text><path d="M170 135 V149" stroke="#94a3b8" stroke-width="1.5" /><path d="M166 145 L170 151 L174 145" fill="none" stroke="#94a3b8" stroke-width="1.5" /><rect x="63" y="152" width="214" height="21" rx="6" fill="#64748b" fill-opacity="0.13" stroke="#94a3b8" /><text x="170" y="166" font-size="11" fill="currentColor" text-anchor="middle">発声モデル → 再構成音声</text></svg>
```

先行研究の中では、まず次の3本が特に重要です。

- **[SPARC — Coding Speech through Vocal Tract Kinematics](https://arxiv.org/abs/2406.12998?utm_source=chatgpt.com)（2024）**：発話器官の運動を音声の内部表現として使う、非常に直接的な先行研究。
- **[ARTI-6 — Towards Six-dimensional Articulatory Speech Encoding](https://arxiv.org/abs/2509.21447?utm_source=chatgpt.com)（2025-2026）**：わずか6次元の調音パラメータから音声を再合成する研究。
- **[Reconstructing the Vocal Tract with Differentiable Acoustic Simulation](https://arxiv.org/abs/2609.36737?utm_source=chatgpt.com)（2026）**：声道内の音響伝播を微分可能な物理シミュレーションとして実装し、音声から声道形状を逆推定する研究。

特に3本目は2026年9月に公開された新しい研究で、物理的な発声モデルそのものをNeural CodecのDecoderに組み込むという方向にかなり近いです。\[[Coding Speech through Vocal Tract Kinematics](https://arxiv.org/abs/2406.12998?utm_source=chatgpt.com); [ARTI-6: Towards Six-dimensional Articulatory Speech Encoding](https://arxiv.org/abs/2509.21447?utm_source=chatgpt.com); [Reconstructing the Vocal Tract with Differentiable Acoustic Simulation](https://arxiv.org/abs/2609.36737?utm_source=chatgpt.com)\]

## 1\. SPARC：最も近いNeural Codecの先行研究

[![Cheol Jun Cho](https://images.openai.com/static-rsc-4/d6ZKwZuj73dFSN6lN3DoKw_Y7mvGuAEz4qCDU1gFpjkZGQhL1cYoYOt2je8tYQDy0tpMnfdPf6jZOVMnfIDvSjGtEkuWaJ3TCkzCUE8fq_cj79jwKtv89Fc3E0qusG5wQ5EisYq2oADYIsH1oCfPEbZXff4P6pU2lANVRdFgTkPfjM1yS2QGLr-HgRHolGe_?purpose=fullsize)](https://cheoljun95.github.io/)

**SPARC（UC Berkeley, 2024）**

音声 → 調音パラメータ → 音声

[GitHub実装](https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding?utm_source=chatgpt.com)

[音声サンプル・デモ](https://berkeley-speech-group.github.io/sparc-demo/?utm_source=chatgpt.com)

SPARCでは、音声を以下のようにエンコードしています。

| 潜在表現 | 次元 |
|--------|--------|
| 舌（前・中央・後）、上唇・下唇、下顎の位置（各XY） | 12 |
| Pitch (F0) | 1 |
| Loudness | 1 |
| 話者Embedding（時間方向で共有） | 64 |

時間方向は50Hzなので、**14次元 × 50Hz** の連続表現になります。

アーキテクチャは以下です。

- Encoder：WavLM-Largeの中間特徴から調音位置を線形射影で推定
- Decoder：調音パラメータと話者Embeddingを条件としたHiFi-GAN
- 学習：Mel reconstruction loss + GAN loss

LibriTTS-Rでの再合成結果は、原音MOS 4.01に対して再合成MOS 3.82。かなり実用的な音質です。\[[Coding Speech through Vocal Tract Kinematics](https://arxiv.org/html/2406.12998?utm_source=chatgpt.com); [Coding Speech through Vocal Tract Kinematics](https://arxiv.org/html/2406.12998?utm_source=chatgpt.com)\]

ただし重要なのは、**SPARCは発声器官の物理シミュレーションをしていない** という点です。あくまで発声器官の動きを潜在表現にして、HiFi-GANで音声波形に戻しています。

## 2\. ARTI-6：6次元まで圧縮

[![Realistic Dynamic Numerical Phantom for MRI of the Upper Vocal Tract](https://images.openai.com/static-rsc-4/EMOkS-29vu2D_9lXZUP7GQRbWQnOulgfZoIWNaTwBvwCZvyNgPriJNsqakpSrck_EPEqOEzl-VtU9gAMJpAxTORQxzIYv8MaWAcvUb33HbwA3zs0vj01noGRfJzCC8xAG6k4DMjXZQu2PY02zf_AeBicQ-MSZeZiE4W5xcmrZYC0k4wMfOjwF4GVddDudERN?purpose=fullsize)](https://www.mdpi.com/2313-433X/6/9/86)

**ARTI-6（USC, ICASSP 2026）**

[GitHub実装](https://github.com/lee-jhwn/arti-6?utm_source=chatgpt.com)

[音声サンプル](https://lee-jhwn.github.io/arti-6/?utm_source=chatgpt.com)

SPARCよりもさらに少ない6個の調音特徴を使っています。

1. Lip Aperture（唇の開き）
2. Tongue Tip（舌先）
3. Tongue Body（舌中央）
4. Velum（軟口蓋）
5. Tongue Root（舌根）
6. Larynx（喉頭）

これらは厳密な3D座標ではなく、MRI画像の関心領域から得られる狭窄度の代理指標です。実際には別途192次元の話者Embeddingも使用します。

論文での再合成性能（LibriTTS-R）

| 表現 | MOS ↑ | WER ↓ |
|--------|--------|--------|
| 原音 | 4.60 | 3.4% |
| Mel 80次元 | 4.45 | 3.8% |
| SPARC型 14次元 | 4.27 | 4.8% |
| ARTI-6 6次元 | 3.95 | 12.5% |

\[[Arti-6: Towards six-dimensional articulatory speech encoding](https://arxiv.org/html/2509.21447?utm_source=chatgpt.com); [Arti-6: Towards six-dimensional articulatory speech encoding](https://arxiv.org/html/2509.21447?utm_source=chatgpt.com)\]

6次元だけでも発話内容の大部分を表現できるのは興味深いですが、情報損失は大きくなっています。

## 3\. diffTLM：実際に声道の物理をシミュレーション

[![PPT - Perspectives for Articulatory Speech Synthesis PowerPoint Presentation - ID:4508490](https://images.openai.com/static-rsc-4/rJ0Y3ASZx2Oj7cs8uF5LdNFdpwsFXulAdtdfSvzeKfaoe1VnCnjqZcv5IG_p_bya4f1Q1ON3Rg3yPC_jjG_cwxddDiZZ7Y2TEfeeYiN_rt7vDyFdqniuH0cVjXFKqQSWCRlZ3Xl-lxT58Bv811aZVBsKYOAHUCpGQWh8t8PVGRrk9NOQ-pSouY_6QqkqYC7s?purpose=fullsize)](https://www.slideserve.com/wind/perspectives-for-articulatory-speech-synthesis-powerpoint-ppt-presentation)

**diffTLM（MIT, NeurIPS 2026）**

[インタラクティブなWebデモ](https://echen01.github.io/vocal-tract/?utm_source=chatgpt.com)

[GitHub実装](https://github.com/echen01/diffTLM?utm_source=chatgpt.com)

こちらはさらに物理寄りです。

- 声道を断面積が時間変化する音響管として近似
- 声道内の音響伝播を計算
- 子音の乱流成分も微分可能にモデル化
- 出力音声からBackpropagationで声道形状を逆推定

周波数領域でのシミュレーションにより、従来の時間領域ソルバと比較してRTX 4090上で約70倍の高速化を報告しています。\[[Reconstructing the Vocal Tract · Eric Ming Chen, Jin Woo Lee &amp; Vincent Sitzmann](https://echen01.github.io/vocal-tract/?utm_source=chatgpt.com); [GitHub - echen01/diffTLM: Code for "Reconstructing the Vocal Tract with Differentiable Acoustic Simulation", NeurIPS 2026 paper](https://github.com/echen01/diffTLM?utm_source=chatgpt.com)\]

ただし、現状公開されているコードは音声サンプルごとの最適化が中心で、Neural Codecとしてそのままリアルタイム運用できる完成形ではありません。

物理的に人間の声道を模擬したいのであれば、こちらの実装は特に参考になりそうです。

## 4\. ほかにも近い研究

**[Articulatory Source-Filter TTS](https://arxiv.org/abs/2610.00735?utm_source=chatgpt.com)（2026年10月）** は、さらに最近の研究です。声帯に相当する音源と、舌・唇の運動による声道フィルタを分離し、調音軌跡でフィルタを制御するTTSです。生成器自体はニューラルな音源・フィルタモデルですが、提案されている構成は今回のアイデアと相性が良いと思います。\[[Articulatory Source-Filter TTS: Physically Grounded Control through Vocal Tract Kinematics](https://arxiv.org/html/2610.00735?utm_source=chatgpt.com)\]

**[VocalTractLab](https://www.vocaltractlab.de/?utm_source=chatgpt.com)** は、NN以前からの物理ベース音声合成システムです。舌・唇・顎などを制御する19個の声道パラメータと、11個の声門パラメータを持つモデルがあり、発声器官を物理的に再現するための参考実装として使えます。\[[VocalTractLab](https://www.vocaltractlab.de/?utm_source=chatgpt.com); [2. Approach](https://www.isca-archive.org/interspeech_2022/sun22b_interspeech.pdf?utm_source=chatgpt.com)\]

## 5\. Neural Codecとしての実現性と研究余地

かなり可能性があると思います。特に低ビットレートCodecとして考える場合、発声器官の運動には強い時間的・構造的な制約があるため、それを利用した圧縮が期待できます。

例えば8bitの単純量子化を仮定すると、

| 表現 | 仮のビットレート |
|--------|--------|
| SPARC 14次元 × 50Hz × 8bit | 5.6 kbps |
| ARTI-6 6次元 × 50Hz × 8bit | 2.4 kbps |

これは未圧縮の特徴系列についての概算で、話者Embeddingなどは含みません。実際には運動の予測符号化やEntropy Codingで、さらに圧縮できる余地があります。

一方で大きな問題が3つあります。

- **逆問題の非一意性：** 同じ音声を複数の舌・口の配置で生成できるため、音声だけから正しい発声器官の状態を一意に決められない。
- **励振源の表現：** F0や口の形だけでは声質・息漏れ・声門閉鎖・摩擦音のノイズなどを十分表現できない。
- **物理モデルの精度：** 実際の人間の声道の全特性を少数のパラメータで完全再現するのは難しい。

そこで、私なら次のようなハイブリッド構成を試します。

研究案：Physics-informed Neural Speech Codec

発話の基本構造は低次元の物理パラメータで再現し、**物理モデルでは表現しきれない声質や高周波の詳細だけをNeural Residualで補う** という構成です。

この場合、物理モデルを通過させることによる強い帰納バイアスを得つつ、最終的な音質はNNで補完できます。残差経路が物理モデルを迂回してすべてを表現しないよう、残差のビットレートを制限することが重要でしょう。

また、これらの研究は必ずしも標準的なNeural Codecのように明示的なRate-Distortion Optimizationを行っているわけではありません。**物理的な調音表現を使って1kbps以下の高品質Codecを作れるか** は、まだ十分に研究の余地がありそうです。

最初に読むならSPARC、実装を触るならSPARCとdiffTLMの2つが特におすすめです。発声器官の状態を潜在表現にするだけなのか、実際の物理的な発声過程までDecoder内部で再現するのかによって、研究の方向性がかなり変わってくると思います。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)