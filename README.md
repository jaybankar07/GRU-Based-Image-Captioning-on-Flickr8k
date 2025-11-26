# flickr8k-gru-image-captioning
GRU-based image captioning model on the Flickr8k dataset using VGG16 image features and Keras/TensorFlow, including preprocessing, training, and caption generation.

# GRU-Based Image Captioning on Flickr8k

This repository contains an implementation of an **image captioning** system using a **GRU-based decoder** on top of **VGG16** image features, trained on the **Flickr8k** dataset.

The project is implemented in a Jupyter notebook (`LAB06_GRU.ipynb`) and is designed to run in a **Kaggle**-style environment, but can be adapted to local setups by changing the dataset paths.

---

## 🔍 Project Overview

The pipeline includes:

- Loading the **Flickr8k** images and captions
- Preprocessing captions (lowercasing, cleaning, adding `startseq` / `endseq` tokens)
- Building a **tokenizer** and computing vocabulary size & max caption length
- Extracting **image features** using **VGG16** (pretrained on ImageNet)
- Creating training sequences:
  - Image feature vector as input 1
  - Partial caption sequence as input 2
  - Next-word prediction as output (one-hot)
- Training a **GRU-based decoder** that combines image and text features
- Saving:
  - Tokenizer (`tokenizer.pkl`)
  - Image features (`image_features.pkl`)
  - Trained GRU models (`best_gru_captioning_model.h5`, `final_gru_captioning_model.h5`)
- Generating sample captions on validation images

---

⚠️ Note: The Flickr8k dataset is not included in this repository.
Download it separately and place it under data/flickr8k/ (or adjust base_dir in the notebook).

---

## 🧠 Model Architecture (High-Level)

1] Image Encoder
    Pretrained VGG16 (ImageNet weights)
    Remove the top classification layers
    Use the penultimate layer as a 4096-dim feature vector

2] Text Decoder (GRU)
    Input: tokenized caption sequence
    Embedding layer → Dropout → GRU(256)
    Merged with encoded image features via Add
    Dense layers with ReLU + Dropout
    Final Dense layer with softmax over the vocabulary

3] Training Objective
    Predict the next word in the caption, given:
    Image features
    Partial caption up to current time step

---

## 🛠️ Technologies Used

Python
TensorFlow / Keras
VGG16 (feature extractor)
GRU-based decoder
NumPy, Pandas
Matplotlib, Seaborn (for visualization)
PIL / Pillow
Scikit-learn (train/validation split)
Pickle (for saving tokenizer & features)

---

## 🔮 Future Improvements

Use Bidirectional GRU or LSTM decoders

Experiment with attention mechanisms for better captions

Use beam search instead of greedy decoding for generation

Train on larger datasets (e.g., MS COCO) for more diverse captions

Convert the notebook into modular Python scripts (src/)

---

## 👨‍💻 Author

Jay Bankar
(Deep Learning / AI Enthusiast)
