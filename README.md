# LightWeight-FineTuning-to-Foundation-Model

## Explanation

This project explores Parameter-Efficient Fine-Tuning (PEFT) using LoRA (Low-Rank Adaptation) on a lightweight transformer model for binary sequence classification. Instead of fine-tuning all model parameters, LoRA introduces additional trainable weights into a few layers of the model.

The pipeline includes:

1. Selecting a compact pretrained model (prajjwal1/bert-tiny)
2. Creating a small text classification dataset
3. Training with LoRA-based fine-tuning using the peft library
4. Saving and reloading the fine-tuned adapter
5. Evaluating performance before and after tuning

This approach is ideal for low-resource environments where full model fine-tuning would be impractical.

This project was completed as part of the Generative AI Fundamentals Nanodegree on Udacity.

## Dependencies

All required libraries are listed in requirements.txt. The core packages include:

* transformers
* datasets
* peft
* evaluate
* torch

## Evaluation

The model is evaluated using accuracy before and after LoRA-based fine-tuning. The lightweight tuning approach drastically reduces training costs while maintaining performance on small datasets.

## Key Concepts 

* Parameter-Efficient Fine-Tuning (PEFT)
* Low-Rank Adaptation (LoRA)
* Binary Sequence Classification
* Hugging Face transformers, datasets, evaluate, and peft libraries