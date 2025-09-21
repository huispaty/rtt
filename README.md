# Towards Minimum Latency Real-Time Piano Transcription
---

This repository contains the code to run inference and evaluation for the causal minimum-latency piano transcription model described in:

> **Exploring System Adaptations for Minimum Latency Real-Time Piano Transcription**
>
> Authors. Patricia Hu, Silvan Peter, Jan Schlüter, Gerhard Widmer
>
> Presented at ISMIR 2025

## Overview
This repository includes inference code and checkpoint to run and evaluate our final model configuration (see Section 4.4 in the paper).

## Requirements
Create a new conda environment with Python 3.9 or higher, activate it and run: `pip install -r requirements.txt`. To run and evaluate the model, run: `python src/inference.py path/to/maestro-v3` where `path/to/maestro-v3` points to the local directory containing the MAESTRO dataset version 3.

# Publication
If you want to find out more, check out our [paper](https://arxiv.org/pdf/2509.07586).

# Acknowledgments
This work is supported by the European Research Council (ERC) under the EU’s Horizon 2020 research & innovation programme, grant agreement No. 10101937 (”Wither Music?”).
