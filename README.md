# CAMELOT

Reproducing the paper"Learning of Cluster-based Feature Importance for Electronic Health Record Time-series", accepted at ICML 2022, in Baltimore, MD, US.

- Looking for the original paper? Check out [ICML Paper](https://lnkd.in/d3kT-RRe).

- Want to read the authors' implementation? Check out [camelot-icml](https://github.com/hrna-ox/camelot-icml).

## ⚡ Quick Start

- Python Dependeceny: [requirements.txt](https://github.com/vanity-lost/Encoder-Decoder-with-Cluster-based-Attention/blob/main/requirements.txt)

- Data Instructions (MIMIC-IV):
  - Download the [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/1.0/)
  - Download the core directory from [MIMIC-IV](https://physionet.org/content/mimiciv/1.0/)
  - Download the [data preprocessing pipeline](https://github.com/hrna-ox/camelot-icml/tree/main/src/data_processing/MIMIC)
  - Run the scripts by `python run_processing.py` (You need to be careful about the directory structure)

- Run one of the notebooks: Showcase.ipynb, ablation_ours.ipynb, and ablation_paper.ipynb.


## 🤔 What is this?

The purpose of this project is to conduct research on the paper entitled "[Learning of Cluster-based Feature Importance for Electronic Health Record Time-series](https://lnkd.in/d3kT-RRe)". This paper introduces a novel approach to learning cluster-based feature importance for electronic health record (EHR) time-series data. The authors present the model CAMELOT, which combines time-series K-means with the encoder-decoder network. The efficacy of the model is validated on two real-world EHR datasets, demonstrating its superiority over existing feature selection methods in terms of robustness and interoperability.

To gain a better understanding of the model, we intend to reproduce the model from scratch, replicate the experiments, test our hypotheses, and document our findings.

## 📖 File System

- Original Paper.pdf
- README&#46;md
- requirements.txt
- code/
  - CAMELOT&#46;py: the CAMELOT architecture
  - Showcase.ipynb: the descriptive notebook
  - ablation_ours.ipynb: codes and results of our proposed ablations
  - ablation_paper.ipynb: codes and results of ablations in the paper
  - best_model
  - data_utils&#46;py: CustomDataset and data loader
  - evaluation_utils&#46;py: Evaluation metrics and helper functions
  - model_utils&#46;py: model parts and helper functions
  - train_utils&#46;py: training loop and helper functions
  - variants_ours&#46;py: models of our proposed ablatioins
  - variants_paper&#46;py: models of ablatioins in the paper

## 📃 Results

Ablations in the paper:
| | CAMELOT | Without $loss_{dist}$ | Without $loss_{clus}$ | Without $loss_{dist}, loss_{clus}$ | Without Attention |
|:---:|:---:|:---:|:---:|:---:|:---:|
| AUC | 0.771 ($\pm$0.023) | 0.773 ($\pm$0.010) | 0.765 ($\pm$0.017) | 0.768 ($\pm$0.013) | 0.748 ($\pm$0.081) |
| F1-score | 0.318 ($\pm$0.021) | 0.317 ($\pm$0.016) | 0.317 ($\pm$0.027) | 0.323 ($\pm$0.014) | 0.322 ($\pm$0.039) |
| Recall | 0.353 ($\pm$0.006) | 0.353 ($\pm$0.006) | 0.347 ($\pm$0.009) | 0.355 ($\pm$0.010) | 0.385 ($\pm$0.066) |
| NMI | 0.109 ($\pm$0.010) | 0.106 ($\pm$0.010) | 0.104 ($\pm$0.017) | 0.107 ($\pm$0.012) | 0.105 ($\pm$0.039) |

Ablations proposed:
| | CAMELOT-GMM | CAMELOT-GRU | CAMELOT-Denoising |
|:---:|:---:|:---:|:---:|
| AUC | 0.768 ($\pm$ 0.019) | 0.764 ($\pm$ 0.019) | 0.764 ($\pm$ 0.009) |
| F1-score | 0.324 ($\pm$ 0.012) | 0.323 ($\pm$ 0.010) | 0.320 ($\pm$ 0.013) |
| Recall | 0.353 ($\pm$ 0.006) | 0.364 ($\pm$ 0.022) | 0.362 ($\pm$ 0.036) |
| NMI | 0.107 ($\pm$ 0.006) | 0.101 ($\pm$ 0.010) | 0.103 ($\pm$ 0.011) |

