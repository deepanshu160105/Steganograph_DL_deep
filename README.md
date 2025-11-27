# 🕵️ Stego-SRNet: LSB Steganography & SRNet Detection

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> **A Hybrid Machine Learning & Cryptography Project** > Hiding data in images using LSB Steganography and detecting it using a custom SRNet Deep Learning model.

---

## 📑 Table of Contents
- [Overview](#-overview)
- [🔗 Model Download](#-model-download)
- [⚠️ Limitations](#-limitations)
- [🧠 How It Works](#-how-it-works)
- [📂 Dataset](#-dataset)
- [📊 Performance](#-performance)
- [🛠️ Installation](#-installation)
- [💻 Usage](#-usage)
- [🤝 Contributing](#-contributing)

---

## 📜 Overview

**Stego-SRNet** is a project designed to explore the boundaries of digital forensics. It consists of two main components:
1.  **Steganography Engine:** A Python-based tool to embed encrypted text into images using **Least Significant Bit (LSB)** modification.
2.  **Steganalysis Detector:** A Deep Learning model based on the **SRNet (Steganalysis Residual Network)** architecture, built with **TensorFlow**, which detects whether an image contains hidden data.

While standard CNNs focus on image content (objects, colors), this model is designed to suppress content and focus on **noise residuals**, allowing it to spot the statistical anomalies introduced by steganography.

---

## 🔗 Model Download

The trained model weights (`.h5` / `.keras`) are hosted on Kaggle. You can download them to run inference locally.

[![Kaggle](https://img.shields.io/badge/Kaggle-Download_Model_Weights-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/models/adarshdubey1127/steganomodel)

---

## ⚠️ Limitations

Please be aware of the following constraints regarding this specific implementation:

### 1. Resolution Limit (150x150)
Due to **GPU hardware limitations** during the training phase, the SRNet model was trained on and is optimized for images of size **150x150 pixels**. 
* **Inference:** Input images larger than this will be resized to 150x150 before processing.
* **Impact:** Resizing usually destroys steganographic artifacts. Therefore, for accurate detection, the analysis should ideally be performed on 150x150 crops or patches of the original image, rather than resizing the whole image.

### 2. Compression Sensitivity
The model detects LSB artifacts. Converting images to lossy formats (like **JPEG**) acts as a filter that "cleans" these artifacts, causing the model to classify the image as "Clean" (False Negative). Always use **PNG** or **BMP**.

---

## 🧠 How It Works

### The Steganography Part
We take a secret text message, encrypt it (optional), convert it to binary, and replace the least significant bit of the image pixels with these bits. To the human eye, the change is invisible.

### The Detection Part (SRNet)
The **SRNet** architecture is specialized for this task. Unlike ResNet or VGG:
* It uses **unpooled layers** in the beginning to preserve noise details.
* It computes **residual noise maps** to separate the "signal" (image content) from the "noise" (hidden data).

---

## 📂 Dataset

The dataset used to train this model was **custom generated** to ensure a perfect 50/50 balance:

* **Source:** High-quality raw images.
* **Generation Process:** 1.  Selected a "Cover" image.
    2.  Generated a random payload.
    3.  Embedded the payload to create a "Stego" counterpart.
* **Total Samples:** [Insert Total Number, e.g., 10,000] images.
* **Preprocessing:** All images were processed to **150x150** dimensions to match the model architecture.

---

## 📊 Performance

The model was evaluated on a held-out test set containing equal numbers of Cover and Stego images.

| Metric | Result |
| :--- | :--- |
| **Accuracy** | **93.90%** |

*Technique used: Spatial Domain LSB Detection*

---

## 🛠️ Installation

### Prerequisites
* Python 3.8+
* TensorFlow 2.x
* Numpy, Pandas, OpenCV/Pillow

### Setup
1.  **Clone the repo**
    ```bash
    git clone [https://github.com/yourusername/stego-srnet.git](https://github.com/yourusername/stego-srnet.git)
    cd stego-srnet
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---
## 💻 Usage & Workflow

This system is designed to be straightforward to use for research and testing purposes. The workflow is divided into two main phases: Data Hiding and Detection.

### Phase 1: Steganography (Hiding Data)
To test the model or create new training data, you first need to generate "Stego-images."
1.  **Input:** Provide a clean "Cover" image (must be a lossless format like PNG or BMP) and a text string you wish to hide.
2.  **Process:** The system converts your text into binary and embeds it into the image pixels using LSB (Least Significant Bit) modification.
3.  **Output:** The tool saves a new "Stego-image" which looks visually identical to the original but contains the hidden payload.

### Phase 2: Steganalysis (The Detector)
To check if an image contains hidden data, feed it into the SRNet model.
1.  **Load Model:** The system loads the pre-trained weights (`.h5` file) downloaded from the link above.
2.  **Image Preprocessing:** Any input image is automatically resized to **150x150 pixels** to match the neural network's architecture.
3.  **Analysis:** The model scans the image for statistical noise anomalies rather than visual content.
4.  **Verdict:** The system outputs a classification result:
    * **Clean:** No hidden data detected.
    * **Stego:** High probability of hidden data modification.

---

## 🤝 Contributing

Contributions are welcome, especially those that address the current resolution limitations!

**How to Contribute:**
1.  **Fork the Project** on GitHub.
2.  **Create a Feature Branch** (e.g., for a new architecture or better preprocessing).
3.  **Commit your changes** with clear descriptions.
4.  **Push to the Branch** and open a **Pull Request**.

**Areas for Improvement:**
* Optimizing the model to handle higher resolutions (above 150x150) without resizing.
* Adding support for detecting other steganography algorithms (like HUGO or WOW).
* Improving the web/command-line interface for easier usage.

---

## 👤 Author

**[Adarsh Dubey]**

* **Mail:** [adarshiiitkota@gmail.com]
* **LinkedIn:** [https://www.linkedin.com/in/adarsh-dubey-313b1a227/]

---
