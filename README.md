# LLM-Course-Projects
This repository contains projects completed as part of the Large Language Models (LLMs) course at 
the University of Tehran (Spring 2024). The projects cover a range of topics related to LLMs, 
including word embeddings, fine-tuning, retrieval-augmented generation, reinforcement learning from human feedback (Alignment), and more.

## Table of Contents

1. [Course Assignment 1 (CA1)](#course-assignment-1-ca1)
   - [Word Embeddings and Masked Language Models](#word-embeddings-and-masked-language-models)
   - [Transfer Learning with BERT](#transfer-learning-with-bert)

2. [Course Assignment 2 (CA2)](#course-assignment-2-ca2)
   - [GPT-2 Text Generation](#gpt-2-text-generation)
   - [Soft Prompt Tuning](#soft-prompt-tuning)

3. [Course Assignment 3 (CA3)](#course-assignment-3-ca3)
   - [Chain-of-Thought and Self-Consistency](#chain-of-thought-and-self-consistency)
   - [Parameter-Efficient Fine-Tuning (PEFT)](#parameter-efficient-fine-tuning-peft)
   - [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)

4. [Course Assignment 4 (CA4)](#course-assignment-4-ca4)
   - [Reinforcement Learning from Human Feedback (RLHF)](#reinforcement-learning-from-human-feedback-rlhf)
   - [Quantization and Instruction Tuning](#quantization-and-instruction-tuning)
   - [Text Evaluation Using Language Models](#text-evaluation-using-language-models)

---

## Course Assignment 1 (CA1)

### Word Embeddings and Masked Language Models

In this project, we explore foundational concepts in natural language processing (NLP). 
We start with word embeddings using the `GloVe` model and the `Gensim` library, 
visualizing and interpreting the semantic relationships captured by these embeddings. 
Next, we dive into Masked Language Modeling (MLM) with `BERT`, a crucial technique in contextual language understanding. 
We experiment with masking tokens in sentences and training `BERT` to predict them, 
which is fundamental to its pre-training process.

### Transfer Learning with BERT

This part focuses on fine-tuning the pre-trained `BERT` model for downstream NLP tasks 
such as text classification and question answering. Using the `Transformers` library by Hugging Face, 
we fine-tune `BERT` on specific datasets and evaluate its performance. The goal is to understand 
how transfer learning can be applied to enhance `BERT`'s performance on task-specific data.

---

## Course Assignment 2 (CA2)

### GPT-2 Text Generation

In this assignment, we work with the `GPT-2` model to explore different prompting techniques:

- **Single Sentence Prompting**: We experiment with generating text based on a single sentence prompt, analyzing generation speed, memory usage, and model performance.
  
- **Batch Generation Prompting**: We compare the performance of single prompts with batch generation, using multiple prompts of varying lengths. This allows us to assess how prompt length and batch processing affect `GPT-2`'s output and efficiency.

### Soft Prompt Tuning

Soft prompt tuning involves creating learnable prompts that are optimized for specific tasks. In this project, we use `BERT` as a base model, applying soft prompts to a dataset of Persian sentences. We implement custom layers for prompt embeddings and evaluate the effectiveness of these soft prompts in improving the model's performance on polarity classification tasks.

---

## Course Assignment 3 (CA3)

### Chain-of-Thought and Self-Consistency

This project explores the Chain-of-Thought (CoT) reasoning technique, which enhances the problem-solving capabilities of LLMs. We apply CoT and Self-Consistency methods to the `Phi-2` model for a question-answering task, comparing its performance to traditional approaches.

### Parameter-Efficient Fine-Tuning (PEFT)

In this section, we focus on efficient fine-tuning methods for large models. We use `LoRA` (Low-Rank Adaptation) to fine-tune the `Phi-2` model for a question generation task, demonstrating how PEFT can reduce computational overhead while maintaining or improving model performance.

### Retrieval-Augmented Generation (RAG)

We implement a Retrieval-Augmented Generation (RAG) system using the `Llama-2 7B Chat` model. The project includes creating a retrieval pipeline with both TF-IDF and semantic retrievers, and integrating these with the language model to generate responses based on retrieved documents. We evaluate the effectiveness of this approach in information retrieval tasks.

---

## Course Assignment 4 (CA4)

### Reinforcement Learning from Human Feedback (RLHF)

This project introduces RLHF as a method to align language model outputs with human preferences. We fine-tune a `GPT-2` model on a summarization task, train a reward model, and apply Proximal Policy Optimization (PPO) to optimize the model based on human feedback.

### Quantization and Instruction Tuning

We explore model quantization using the `QLoRA` method to fine-tune the `Mistral 7B` model. This process reduces the model's memory footprint and speeds up inference. Additionally, we perform instruction tuning on the `Mistral 7B Instruct` model, enabling it to follow complex instructions more effectively.

### Text Evaluation Using Language Models

In this final section, we evaluate text generation using `BERTScore`, a metric for comparing the similarity of generated text to reference text. We experiment with both the official `BERTScore` implementation and a custom implementation, using the `DeBERTa` model for evaluation tasks.

---

Please note that *this README was generated with the assistance of ChatGPT (GPT-4 engine), an AI language model developed by OpenAI.*