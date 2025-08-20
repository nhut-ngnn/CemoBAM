# CemoBAM: Advancing Multimodal Emotion Recognition through Heterogeneous Graph Networks and Cross-Modal Attention Mechanisms
<i>
  Official code repository for the manuscript 
  <b>"CemoBAM: Advancing Multimodal Emotion Recognition through Heterogeneous Graph Networks and Cross-Modal Attention Mechanisms"</b>, 
  accepted to 
  <a href="https://sites.google.com/view/apnoms2025">The 25th Asia-Pacific Network Operations and Management Symposium Conference</a>.
</i>

> Please press ⭐ button and/or cite papers if you feel helpful.

<p align="center">
<img src="https://img.shields.io/github/stars/nhut-ngnn/CemoBAM">
<img src="https://img.shields.io/github/forks/nhut-ngnn/CemoBAM">
<img src="https://img.shields.io/github/watchers/nhut-ngnn/CemoBAM">
</p>

<div align="center">

[![python](https://img.shields.io/badge/-Python_3.8.20-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![pytorch](https://img.shields.io/badge/Torch_2.0.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![cuda](https://img.shields.io/badge/-CUDA_11.8-green?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit-archive)
</div>

<p align="center">
<img src="https://img.shields.io/badge/Last%20updated%20on-18.05.2025-brightgreen?style=for-the-badge">
<img src="https://img.shields.io/badge/Written%20by-Nguyen%20Minh%20Nhut-pink?style=for-the-badge"> 
</p>


<div align="center">

[**Abstract**](#Abstract) •
[**Install**](#install) •
[**Usage**](#usage) •
[**References**](#references) •
[**Citation**](#citation) •
[**Contact**](#Contact)

</div>

## Abstract 
> Multimodal Speech Emotion Recognition (SER) offers significant advantages over unimodal approaches by integrating diverse information streams such as audio and text. However, effectively fusing these heterogeneous modalities remains a significant challenge. We propose CemoBAM, a novel dual-stream architecture that synergistically combines a Cross-modal Heterogeneous Graph Attention Network (CH-GAT) and a Cross-modal Convolutional Block Attention Mechanism (xCBAM). In CemoBAM architecture, the CH-GAT constructs a heterogeneous graph that models intra- and inter-modal relationships, employing multi-head attention to capture fine-grained dependencies across audio and text feature embeddings. The xCBAM enhances feature refinement through a cross-modal transformer with a modified 1D-CBAM, employing bidirectional cross-attention and channel-spatial attention to emphasize emotionally salient features. The CemoBAM architecture surpasses previous state-of-the-art methods by 0.32\% on IEMOCAP and 3.25\% on ESD datasets. Comprehensive ablation studies validate the impact of Top-K graph construction parameters, fusion strategies, and the complementary contributions of both modules. The results highlight CemoBAM's robustness and potential for advancing multimodal SER applications.
>
> Index Terms: Multimodal emotion recognition, Speech emotion recognition, Cross-modal heterogeneous graph attention, Cross-modal convolutional block attention mechanism, Feature fusion.


## Install
### Clone this repository
```
git clone https://github.com/nhut-ngnn/CemoBAM.git
```

### Create Conda Enviroment and Install Requirement
Navigate to the project directory and create a Conda environment:
```
cd CemoBAM
conda create --name cemobam python=3.8
conda activate cemobam
```
### Install Dependencies
```
pip install -r requirements.txt
```
### Dataset 

CemoBAM is evaluated on two widely-used multimodal emotion recognition datasets:

#### IEMOCAP (Interactive Emotional Dyadic Motion Capture)
- **Modality**: Audio + Text  
- **Classes**: `angry`, `happy`, `sad`, `neutral` (4 classes)  
- **Sessions**: 5  
- **Official Website**: [https://sail.usc.edu/iemocap/](https://sail.usc.edu/iemocap/)  
- **Note**: We use **Wav2Vec2.0** for audio and **BERT** for text feature extraction.

#### 🔹 ESD (Emotional Speech Dataset)
- **Modality**: Audio + Text  
- **Languages**: English, Mandarin, and more  
- **Classes**: `neutral`, `angry`, `happy`, `sad`, `surprise` (5 classes)  
- **Official GitHub**: [https://github.com/HLTSingapore/ESD](https://github.com/HLTSingapore/ESD)  

## Usage
### Preprocessed Features
We provide `.pkl` files with BERT and Wav2Vec2.0 embeddings for each dataset:

```
feature/
├── IEMOCAP_BERT_WAV2VEC_train.pkl
├── IEMOCAP_BERT_WAV2VEC_val.pkl
├── IEMOCAP_BERT_WAV2VEC_test.pkl
├── ESD_BERT_WAV2VEC_train.pkl
├── ESD_BERT_WAV2VEC_val.pkl
├── ESD_BERT_WAV2VEC_test.pkl
```

### Train & Evaluate
Run a grid search on different `k_text` and `k_audio` values for graph construction:
```bash
bash selected_topK.sh
```

### Single Run Example
To run CemoBAM with a specific configuration:
```bash
bash run_training.sh
```

### Evaluation
Evaluate saved models using:
```bash
bash run_eval.sh
```
## References
[1] Nhat Truong Pham, SERVER: Multi-modal Speech Emotion Recognition using Transformer-based and Vision-based Embeddings (ICIIT), 2023. Available https://github.com/nhattruongpham/mmser.git.

[2] Mustaqeem Khan, MemoCMT: Cross-Modal Transformer-Based Multimodal Emotion Recognition System (Scientific Reports), 2025. Available https://github.com/tpnam0901/MemoCMT.

[3] Nhut Minh Nguyen, HemoGAT: Heterogeneous multi-modal emotion recognition with cross-modal transformer and graph attention network, 2025. Available https://github.com/nhut-ngnn/HemoGAT.

## Citation
If you use this code or part of it, please cite the following papers:
```
Update soon
```
## Contact
For any information, please contact the main author:

Nhut Minh Nguyen at FPT University, Vietnam

**Email:** [minhnhut.ngnn@gmail.com](mailto:minhnhut.ngnn@gmail.com)<br>
**ORCID:** <link>https://orcid.org/0009-0003-1281-5346</link> <br>
**GitHub:** <link>https://github.com/nhut-ngnn/</link>
