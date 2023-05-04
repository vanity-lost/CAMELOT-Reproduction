# CAMELOT

This repository is a reproduction of the paper "Learning of Cluster-based Feature Importance for Electronic Health Record Time-series", accepted at ICML 2022, in Baltimore, MD, US.

> 📋 Looking for the original paper? Check out [ICML Paper](https://lnkd.in/d3kT-RRe).

> 📋 Want to read the authors' implementation? Check out [camelot-icml](https://github.com/hrna-ox/camelot-icml).

## 🤔 What is this?

The purpose of this project is to conduct research on the paper entitled "[Learning of Cluster-based Feature Importance for Electronic Health Record Time-series](https://lnkd.in/d3kT-RRe)". This paper introduces a novel approach to learning cluster-based feature importance for electronic health record (EHR) time-series data. The authors present the model CAMELOT, which combines time-series K-means with the encoder-decoder network. The efficacy of the model is validated on two real-world EHR datasets, demonstrating its superiority over existing feature selection methods in terms of robustness and interoperability.

To gain a better understanding of the model, we intend to reproduce the model from scratch, replicate the experiments, test our hypotheses, and document our findings.

## ⚡ Quick Start

- Python Dependeceny: [requirements.txt](https://github.com/vanity-lost/Encoder-Decoder-with-Cluster-based-Attention/blob/main/requirements.txt)

  - `pip install -r requirements.txt`

- Data Instructions (MIMIC-IV):

  - Download the [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/1.0/)
  - Download the core directory from [MIMIC-IV](https://physionet.org/content/mimiciv/1.0/)
  - Download the [data preprocessing pipeline](https://github.com/hrna-ox/camelot-icml/tree/main/src/data_processing/MIMIC)
  - Run the scripts by `python run_processing.py` (You need to be careful about the directory structure)

- Run notebooks or train/evaluation scripts

## :book: Notebooks

To review the training/evaluation codes and results conveniently, we prepare three notebooks:

- [Showcase.ipynb](https://github.com/vanity-lost/Encoder-Decoder-with-Cluster-based-Attention/blob/main/code/Showcase.ipynb): contains data preprocessing, data loading, training, and evaluation with plots and stats
- [ablation_paper.ipynb](https://github.com/vanity-lost/Encoder-Decoder-with-Cluster-based-Attention/blob/main/code/ablations_paper.ipynb): contains the ablation studies in the paper
- [ablation_ours.ipynb](https://github.com/vanity-lost/Encoder-Decoder-with-Cluster-based-Attention/blob/main/code/ablations_ours.ipynb): contains the ablation studies proposed by us

## :black_nib: Training

To train the model in the paper, run:

`python train.py`

## :eyeglasses: Evaluation

To evaluate the trained model on MIMIC-IV-ED, run:

`python eval.py`

## :sunny: Pre-trained Models

You can download pretrained models here:

- [Our CAMELOT model](https://github.com/vanity-lost/Encoder-Decoder-with-Cluster-based-Attention/blob/main/code/best_model) trained on MIMIC-IV-ED.

## :page_with_curl: Results

Our model achieves the following performance on :

Ablations in the paper:
| | CAMELOT | Without $loss_{dist}$ | Without $loss_{clus}$ | Without $loss_{dist}, loss_{clus}$ | Without Attention |
|:---:|:---:|:---:|:---:|:---:|:---:|
| AUC | 0.771 (±0.023) | 0.773 (±0.010) | 0.765 (±0.017) | 0.768 (±0.013) | 0.748 (±0.081) |
| F1-score | 0.318 (±0.021) | 0.317 (±0.016) | 0.317 (±0.027) | 0.323 (±0.014) | 0.322 (±0.039) |
| Recall | 0.353 (±0.006) | 0.353 (±0.006) | 0.347 (±0.009) | 0.355 (±0.010) | 0.385 (±0.066) |
| NMI | 0.109 (±0.010) | 0.106 (±0.010) | 0.104 (±0.017) | 0.107 (±0.012) | 0.105 (±0.039) |

Ablations proposed:
| | CAMELOT-GMM | CAMELOT-GRU | CAMELOT-Denoising |
|:---:|:---:|:---:|:---:|
| AUC | 0.768 (± 0.019) | 0.764 (± 0.019) | 0.764 (± 0.009) |
| F1-score | 0.324 (± 0.012) | 0.323 (± 0.010) | 0.320 (± 0.013) |
| Recall | 0.353 (± 0.006) | 0.364 (± 0.022) | 0.362 (± 0.036) |
| NMI | 0.107 (± 0.006) | 0.101 (± 0.010) | 0.103 (± 0.011) |

## 📋 Contribute

> The repository is under MIT license. Contribution are always welcome! Please submit a pull request or contact randolph.zhao.49@gmail.com for more details.
