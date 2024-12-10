# Self-guided Decoding to Reduce Hallucinations in Large Language Models
This repository contains the implementation for our work on reducing hallucinations in Large Language Models through self-guided decoding, developed as part of CS263: Natural Language Processing at UCLA (Fall 2024).

## Overview
Our work explores using LLM's internal representations to detect and reduce hallucinations during generation. We implement a classifier-guided decoding technique that steers the model towards more truthful outputs by adjusting generation probabilities based on hallucination likelihood.
Repository Structure

## Main Components

```NLP_Project.ipynb``` : Primary implementation notebook that handles:

- TruthfulQA dataset processing

- Token-level dataset creation for classifier training

- Classifier training and implementation

- Inference with the guided decoding approach

- BLEU score calculation and output generation



```human_eval.ipynb```: Evaluation interface notebook for:

- Human evaluation of model outputs

- Automatic calculation of Truthfulness and Informativeness metrics

- Result analysis and visualization

## Additional Files

- ```test_indices.npy```: Contains test split indices for reproducibility
- ```Modified generation/utils.py```: Contains custom implementation for guided decoding

## Implementation Notes
The implementation requires modifying the transformers library source code, specifically the generation utilities. The modifications enable the guided decoding approach without affecting other model functionalities.

## Evaluation
We evaluate our approach using both automatic metrics (BLEU) and human evaluation (Truthfulness and Informativeness). The evaluation pipeline is fully automated through the provided notebooks.

## Dataset
We use the TruthfulQA dataset, which is specifically designed to evaluate factual consistency in language model outputs. The dataset processing includes creation of token-level annotations for classifier training.

## Dependencies

- numpy

- torch

- transformers

- scikit-learn

- datasets

- tqdm

- sacrebleu



## Usage

- Run NLP_Project.ipynb to train the model and generate outputs

- Use human_eval.ipynb for evaluation of the generated outputs

- Results will be automatically saved and metrics calculated
