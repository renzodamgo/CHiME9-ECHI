The following rules apply to all systems participating in the CHiME-9 Task 2 (ECHI) challenge:

- **Training data**: You may use the training subset of the ECHI dataset and any external datasets explicitly listed in the [Data and pre-trained models](https://www.chimechallenge.org/current/task2/rules#external-data-and-pre-trained-models) section. If you believe a public dataset is missing, you may propose its inclusion before the deadline specified below.
- **Development data**: The development subset of ECHI may be used for evaluation purposes only. It must not be used for training or automatic tuning.
- **Streaming requirement**: Systems must be streaming in nature — that is, they should process inputs sequentially over time with a maximum algorithmic latency of **20 ms**. Systems must not access or rely on global information from a recording (e.g., global normalization, non-streaming speaker identification or diarization).
- **Latency documentation**: The report accompanying system submissions must include a clearly labeled section titled **“Latency”**, detailing any lookahead, chunk-based processing, and other latency-relevant characteristics. This must also include an explicit estimate of the average algorithmic and emission latency. Only systems meeting the 20 ms algorithmic latency constraint will be ranked.
- **Pre-trained models**: Only the models listed in the [Data and pre-trained models](https://www.chimechallenge.org/current/task2/rules#external-data-and-pre-trained-models) section may be used. If a useful model is missing, you may propose it for inclusion before the deadline. If using pre-trained models be careful to ensure that they are compatible with the 20 ms latency constraint.
- **Evaluation integrity**: Each recording in the evaluation set must be processed independently. The system must not be fine-tuned on the evaluation data or use global information across recordings. Within a session, when performing enhancement, systems must treat the Aria and the hearing aid devices independently. (There are no restrictions on how the Aria and hearing aid signals are used during training.)
- **Speaker identity**: The IDs of the four conversational partners are provided for all sessions (train, dev and eval). Systems can freely use the the ‘Rainbow passage’ recordings of all four speakers in the session, i.e. to guide target speaker extraction, or own voice suppression.

> **Note**: We are also interested in systems that might not fully meet the required 20 ms latency constraint and such systems and respective papers are welcome as contributions to the CHiME workshop. Thus, if your system violates any of the above rules (e.g., uses private data), you may still submit it, but it will not be included in the official rankings or considered for the subjective listening test stage.

---

## [](https://www.chimechallenge.org/current/task2/rules#evaluation)Evaluation

Systems being evaluated should take as **input**, only,

- the multichannel audio signal for the device being considered,
- (optionally) the ‘Rainbow passage’ recordings of the device wearer and each of the conversational partners, i.e. to distinguish between voices of the device wearer, the conversational partners, and other speakers in the environment. Note, the ID numbers for the device wearers and conversational partners are provided in the metadata.

Systems should **output** a **single-channel rendition of each of the three conversational partners** speaking to the device wearer (using either the hearing aid [HA] or Aria glasses). (For file naming conventions and format details see [Submission](https://www.chimechallenge.org/current/task2/Submission); for how to evaluate your signals locally see [Baseline and Evaluation Framework](https://www.chimechallenge.org/current/task2/baseline)).

These three output streams will be evaluated using:

- **Objective measures**: To assess the intelligibility and quality of each stream.
- **Listening tests**: To assess listener preference when hearing the combined (summed) signal of all three streams.

For the development data, participants will be provided with:

- Time-stamped segmentations marking the start and end of each speech segment per speaker.
- Corresponding reference audio for each segment (see [Reference Signals](https://www.chimechallenge.org/current/task2/data#reference-signals)).

For the final evaluation dataset, the noisy signals will be provided but the segmentation and reference audio will be withheld.

Entrants will submit their enhanced signals and metrics will be computed by the organisers.

> Final system **rankings** will be based solely on listening test results. It is well-known that objective metrics only partly correlate with listening tests and that this is particularly to be expected for the recordings in the CHiME9-ECHI data with the reference signals constructed as described in [Reference Signals](https://www.chimechallenge.org/current/task2/data#reference-signals). Thus, participants should be aware that the provided objective metrics and the resulting scores are to be considered rather informative. Objective metrics will be reported for all submissions to help guide development but will not affect rankings.

---

### [](https://www.chimechallenge.org/current/task2/rules#objective-measures)Objective Measures

Objective evaluation will include both **reference-free** (i.e., independent metrics a.k.a. non-intrusive) and **reference-based** (intrusive) metrics (i.e., dependent metrics). All metrics are being computed using the [VERSA toolkit](https://github.com/wavlab-speech/versa)). The metrics will be computed per speech segment, per speaker, and averaged across all sessions in the evaluation set. A segment-length weighted average is provided. The evaluation code will provide overall scores, and also per-session and per-participant statistics for each metric being used.

The metrics will be computed and reported separately using two types of signal:

- **individual** - using segments from the separate speaker streams.
- **summed** - the three individual speaker streams are summed (and similarly for the references).

We expect the summed metric to better reflect the quality of the signals being used in the listening test, which will be constructed by summing the individual speaker streams to reconstitute the conversation. For example, systems that separate the conversation from the background but misallocate conversation partners between the participant streams, may do poorly on the ‘individual metrics’ but produce perfectly good signals when summed.

#### [](https://www.chimechallenge.org/current/task2/rules#independent-metrics)INDEPENDENT METRICS

The following independent metrics will be reported.

- Deep Noise Suppression MOS Score of P.835 (DNSMOS) (1)
- Deep Noise Suppression MOS Score of P.808 (DNSMOS) (2)
- Non-intrusive Speech Quality and Naturalness Assessment (NISQA) (3)
- UTokyo-SaruLab System for VoiceMOS Challenge 2022 (UTMOS) (4)
- PESQ in TorchAudio-Squim (6)
- STOI in TorchAudio-Squim (7)
- SI-SDR in TorchAudio-Squim (8)

Numbers in parentheses refer to the table row on the [VERSA GitHub page here](https://github.com/wavlab-speech/versa/blob/main/docs/supported_metrics.md#independent-metrics).

#### [](https://www.chimechallenge.org/current/task2/rules#dependent-metrics)DEPENDENT METRICS

The following dependent metrics will be reported.

- Mel Cepstral Distortion (MCD) (1)
- Signal-to-interference Ratio (SIR) (4)
- Signal-to-artifact Ratio (SAR) (5)
- Signal-to-distortion Ratio (SDR) (6)
- Convolutional scale-invariant signal-to-distortion ratio (CI-SDR) (7)
- Scale-invariant signal-to-noise ratio (SI-SNR) (8)
- Perceptual Evaluation of Speech Quality (PESQ) (9)
- Short-Time Objective Intelligibility (STOI) (10)
- Frequency-Weighted SEGmental SNR (FWSEGSNR) (20)
- Weighted Spectral Slope (WSS) (21)
- Cepstrum Distance Objective Speech Quality Measure (CD) (22)
- Composite Objective Speech Quality (23)
- Coherence and speech intelligibility index (CSII) (24)
- Normalized-covariance measure (NCM) (25)

Numbers in parentheses refer to the table row on the [VERSA GitHub page here](https://github.com/wavlab-speech/versa/blob/main/docs/supported_metrics.md#dependent-metrics).

#### [](https://www.chimechallenge.org/current/task2/rules#non-match-metric)NON MATCH METRIC

In addition, the following metrics listed by VERSA as ‘non match metrics’ will be used:

- MOS in TorchAudio-Squim (2)
- Log Likelihood Ratio (LLR) (11)

Numbers in brackets refer to the table row on the [VERSA GitHub page here](https://github.com/wavlab-speech/versa/blob/main/docs/supported_metrics.md#non-match-metrics).

Non-match metrics are those where the reference signal is not necessarily a matched counterpart of the signal being assessed. They are useful in situations such as ECHI where the precise noise-free ground truth is hard to obtain.

> Note that independent metrics might be erroneous since they might show only weak correlation with assessment by the final listening test, e.g. by being positively impacted by competing or backchannel speech or non-speech audio parts like laughter, and that dependent metrics might be only weakly correlated due to the reference signal constructions. We therefore advise challenge participants to listen to the signals enhanced by their respective systems instead of blindly relying on metric scores alone.

---

The metrics above are computed using the Versa metric config file shown below:

```
- name: nisqa
- name: pseudo_mos
  predictor_types: ["utmos", "dnsmos"]
- name: pysepm
- name: pesq
- name: signal_metric
- name: stoi
- name: squim_no_ref
- name: squim_ref
```

---

> ⚠️ **Latency rule**: Participants may apply a **fixed delay** (up to 20 ms) across all submitted output streams to match reference timing. This single delay must be consistent across all recordings and clearly reported.

---

### [](https://www.chimechallenge.org/current/task2/rules#listening-tests)Listening Tests

Subjective evaluation will be conducted using **MUSHRA-style listening tests**, where human listeners assess segments formed by summing the three processed speaker streams into a single channel.

In case that too many submissions are received to conduct a full listening test for all systems, a pre-listening test with reduced number of signals will be conducted to short-list the most promising systems.

Listeners will be asked to assess

- target signal quality
- background noise quality
- overall signal quality
- intelligibility (either subjectively self-reported or semi-formal (since formal intelligibility assessment by e.g. matrix tests will not be possible due to the nature of the CHiME9-ECHI Dataset))

(More information on the full methodology will be provided later.)

### [](https://www.chimechallenge.org/current/task2/rules#external-data-and-pre-trained-models)External data and pre-trained models

Besides the ECHI dataset published with this challenge, the participants are allowed to use public datasets and pre-trained models listed below. In case you want to propose additional dataset or pre-trained model to be added to these lists, do so by contacting us until August 18th 2025. If you want to use a private dataset or model, you may still submit your system to the challenge, but we will not include it in the final rankings.

Participants may use these publicly available datasets for building the systems:

- [AMI](https://groups.inf.ed.ac.uk/ami/corpus/)
- [LibriSpeech](https://www.openslr.org/12/)
- [TEDLIUM](https://www.openslr.org/51/)
- [MUSAN](https://www.openslr.org/17/)
- [RWCP Sound Scene Database](https://www.openslr.org/13/)
- [REVERB Challenge RIRs.](http://reverb2014.dereverberation.com/tools/reverb_tools_for_Generate_mcTrainData.tgz)
- [Aachen AIR dataset.](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/)
- [BUT Reverb database.](https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database)
- [SLR28 RIR and Noise Database (contains Aachen AIR, MUSAN noise, RWCP sound scene database and REVERB challenge RIRs, plus simulated ones).](https://www.openslr.org/28/)
- [VoxCeleb 1&2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
- [FSD50k](https://zenodo.org/record/4060432)
- WSJ0-2mix, WHAM, [WHAMR](http://wham.whisper.ai/), WSJ
- [SINS](https://github.com/fgnt/sins)
- [LibriCSS acoustic transfer functions (ATF)](https://github.com/jsalt2020-asrdiar/jsalt2020_simulate/tree/master)
- [NOTSOFAR1 simulated CSS dataset](https://github.com/microsoft/NOTSOFAR1-Challenge#2-training-on-the-full-simulated-training-dataset)
- [Ego4D](https://ego4d-data.org/)
- [Project Aria Datasets](https://www.projectaria.com/datasets/)
- [DNS challenge noises](https://github.com/microsoft/DNS-Challenge)

In addition, following pre-trained models may be used:

> **A note on latency**: The models listed below are provided as potential starting points. Their inclusion in this list does not guarantee compliance with the 20 ms algorithmic latency rule and many used ‘as is’ are definitely not compliant. It is the participants responsibility for verifying and documenting that their system meets the challenge requirements.

- Wav2vec:
    - [S3PRL](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html)
        - [wav2vec-large](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec-large)
- Wav2vec 2.0:
    - [Fairseq:](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#wav2vec-20)
        - All models including [Wav2Vec 2.0 Large (LV-60 + CV + SWBD + FSH)](https://dl.fbaipublicfiles.com/fairseq/wav2vec/w2v_large_lv_fsh_swbd_cv_ftsb300_updated.pt) and the multi-lingual [XLSR-53 56k](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt)
    - [Torchaudio:](https://pytorch.org/audio/0.10.0/pipelines.html)
        - [WAV2VEC2_BASE](https://pytorch.org/audio/0.10.0/pipelines.html#wav2vec2-base)
        - [WAV2VEC2_LARGE](https://pytorch.org/audio/0.10.0/pipelines.html#wav2vec2-large)
        - [WAV2VEC2_LARGE_LV60K](https://pytorch.org/audio/0.10.0/pipelines.html#wav2vec2-large-lv60k)
        - [WAV2VEC2_XLSR53](https://pytorch.org/audio/0.10.0/pipelines.html#wav2vec2-xlsr53)
        - [WAV2VEC2_ASR_LARGE_LV60K_960H](https://pytorch.org/audio/0.10.0/pipelines.html#wav2vec2-asr-large-lv60k-960h)
        - [WAV2VEC2_ASR_BASE_960](https://pytorch.org/audio/0.10.0/pipelines.html#wav2vec2-asr-base-960h)
    - [Huggingface:](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
        - [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
        - [facebook/wav2vec2-large-960h](https://huggingface.co/facebook/wav2vec2-large-960h)
        - [facebook/wav2vec2-large-960h-lv60-self](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self)
        - [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)
        - [facebook/wav2vec2-large-lv60](https://huggingface.co/facebook/wav2vec2-large-lv60)
        - [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)
        - [wav2vec2-large lv60 + speaker verification](https://huggingface.co/anton-l/wav2vec2-base-superb-sv)
        - Other models on Huggingface using the same weights as the [Fairseq ones](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#wav2vec-20).
    - [S3PRL](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html)
        - [wav2vec2_base_960](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec2)
        - [wav2vec2_base_960](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec2_base_960)
        - [wav2vec2_large_960](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec2_large_960)
        - [wav2vec2_large_ll60k](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec2_large_ll60k)
        - [wav2vec2_large_lv60_cv_swbd_fsh](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec2_large_lv60_cv_swbd_fsh)
        - [wav2vec2_conformer_relpos](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec2_conformer_relpos)
        - [wav2vec2_conformer_rope](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec2_conformer_rope)
        - [Xlsr_53](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#xlsr_53)
- [HuBERT](https://arxiv.org/abs/2106.07447)
    - [Torchaudio](https://pytorch.org/audio/0.10.0/pipelines.html#wav2vec-2-0-hubert-representation-learning)
        - [HuBERT base](https://pytorch.org/audio/0.10.0/pipelines.html#hubert-base)
        - [HuBERT large](https://pytorch.org/audio/0.10.0/pipelines.html#hubert-large)
        - [HuBERT xlarge](https://pytorch.org/audio/0.10.0/pipelines.html#hubert-xlarge)
        - [HuBERT ASR large](https://pytorch.org/audio/0.10.0/pipelines.html#hubert-asr-large)
        - [HuBERT ASR xlarge](https://pytorch.org/audio/0.10.0/pipelines.html#hubert-asr-xlarge)
    - [Huggingface:](https://huggingface.co/docs/transformers/model_doc/hubert)
        - [hubert-base-ls960](https://huggingface.co/facebook/hubert-base-ls960)
        - [hubert-large-ll60k](https://huggingface.co/facebook/hubert-large-ll60k)
        - [hubert-xlarge-ll60k](https://huggingface.co/facebook/hubert-xlarge-ll60k)
        - [hubert-large-ls960-ft](https://huggingface.co/facebook/hubert-large-ls960-ft)
        - [hubert-xlarge-ls960-ft](https://huggingface.co/facebook/hubert-xlarge-ls960-ft)
    - [S3PRL:](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html)
        - [hubert-base](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#hubert-base)
        - [hubert-large_ll60k](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#hubert-large_ll60k)
- [WavLM](https://arxiv.org/abs/2110.13900)
    - [Huggingface:](https://huggingface.co/docs/transformers/model_doc/wavlm)
        - [wavlm-base](https://www.chimechallenge.org/current/task1/rules#:~:text=Huggingface%3A-,wavlm%2Dbase,-wavlm%2Dbase%2Dsv)
        - [wavlm-base-sv](https://huggingface.co/microsoft/wavlm-base-sv)
        - [wavlm-base-sd](https://huggingface.co/microsoft/wavlm-base-sd)
        - [wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus)
        - [wavlm-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv)
        - [wavlm-base-plus-sd](https://huggingface.co/microsoft/wavlm-base-plus-sd)
        - [wavlm-large](https://huggingface.co/microsoft/wavlm-large)
    - [S3PRL:](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html)
        - [wavlm-base](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wavlm-base)
        - [wavlm-base-plus](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wavlm-base-plus)
        - [wavlm-large](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wavlm-large)
- [Tacotron2](https://github.com/NVIDIA/tacotron2)
    - [Torchaudio:](https://pytorch.org/audio/0.10.0/pipelines.html#tacotron2-text-to-speech)
        - [tacotron2-wavernn-phone-ljspeech](https://pytorch.org/audio/0.10.0/pipelines.html#tacotron2-wavernn-phone-ljspeech)
        - [tacotron2-griffinlim-phone-ljspeech](https://pytorch.org/audio/0.10.0/pipelines.html#tacotron2-griffinlim-phone-ljspeech)
        - [tacotron2-wavernn-char-ljspeech](https://pytorch.org/audio/0.10.0/pipelines.html#tacotron2-wavernn-char-ljspeech)
        - [tacotron2-griffinlim-char-ljspeech](https://pytorch.org/audio/0.10.0/pipelines.html#tacotron2-griffinlim-char-ljspeech)
- [ECAPA-TDNN](https://arxiv.org/abs/2005.07143)
    - [Speechbrain](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
    - [NeMo toolkit](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speaker_recognition/models.html#ecapa-tdnn)
- [X-vector extractor](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf)
    - [VBx repo ResNet101 16kHz](https://github.com/BUTSpeechFIT/VBx/tree/master/VBx/models/ResNet101_16kHz)
    - [Kaldi](https://kaldi-asr.org/models/m7)
    - [Pyannote Audio](https://huggingface.co/pyannote/embedding)
- [Pyannote Segmentation](https://huggingface.co/pyannote/segmentation)
- [Pyannote Diarization (Pyannote Segmentation+ECAPA-TDNN from SpeechBrain)](https://huggingface.co/pyannote/speaker-diarization)
- [NeMo toolkit ASR pre-trained models:](https://github.com/NVIDIA/NeMo)
    - [Citrinet](https://huggingface.co/nvidia/stt_en_citrinet_1024_gamma_0_25)
    - [Conformer-CTC](https://huggingface.co/nvidia/stt_en_conformer_ctc_large)
    - [Conformer-Transducer](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html#conformer-transducer)
    - [FastConformer](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi)
- [NeMo toolkit speaker ID embeddings models:](https://github.com/NVIDIA/NeMo)
    - [TitaNet](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speaker_recognition/models.html#titanet)
    - [SpeakerNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/speakerverification_speakernet)
    - [ECAPA-TDNN](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speaker_recognition/models.html#ecapa-tdnn)
- [NeMo toolkit VAD models:](https://github.com/NVIDIA/NeMo)
    - [MarbleNet VAD](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speech_classification/models.html#marblenet-vad)
- [NeMo toolkit diarization models:](https://github.com/NVIDIA/NeMo)
    - [Multi-Scale Diarization Decoder](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speaker_diarization/models.html#multi-scale-diarization-decoder)
- [Whisper](https://arxiv.org/abs/2212.04356)
    - [Whisper official repo (all versions: small, medium, large v1/v2/v3)](https://github.com/openai/whisper)
- [OWSM: Open Whisper-style Speech Model](https://www.wavlab.org/activities/2024/owsm/)
    - [OWSM v3 E-Branchformer](https://huggingface.co/espnet/owsm_v3.1_ebf)
    - [OWSM v3 with E-Branchformer (base, smaller version)](https://huggingface.co/espnet/owsm_v3.1_ebf_base)
    - [OWSM v3](https://huggingface.co/espnet/owsm_v3)
    - [OWSM v2](https://huggingface.co/espnet/owsm_v2)
    - [OWSM v2 E-Branchformer (note: this is probably the best OWSM for English ASR)](https://huggingface.co/espnet/owsm_v2_ebranchformer)
    - [OWSM v1](https://huggingface.co/espnet/owsm_v1)
- [Icefall Zipformer](https://huggingface.co/yfyeung/icefall-asr-gigaspeech-zipformer-2023-10-17)
- [RWKV Transducer](https://www.modelscope.cn/models/iic/speech_rwkv_transducer_asr-en-16k-gigaspeech-vocab5001-pytorch-online/summary)