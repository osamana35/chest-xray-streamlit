# PneuNet: Pneumonia Classification from Chest X-Rays with CNNs

An interpretable, production-ready deep learning framework designed to accurately classify chest X-ray images into three categories: **Normal**, **Bacterial Pneumonia**, and **Viral Pneumonia**. This project was developed as a graduation project at Yarmouk University.

---

## 🚀 Live Demo
* **Interactive Web Application:** [Click Here to Open the Live App](https://s4pzndg6x7rvfddbfmku8j.streamlit.app/)

---

## 🛠️ Key Technical Highlights & Methodology
Unlike standard black-box classifiers, this repository implements a comprehensive, clinically-focused pipeline:
* **Advanced Preprocessing:** Applied **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to enhance localized contrast in grayscale X-rays, capturing subtle radiographic features.
* **High-Resolution Training:** Configured and optimized training at **600×600 pixels** to maximize feature retention.
* **Class Imbalance Mitigation:** Integrated **Focal Loss** instead of standard cross-entropy to actively counter the training difficulty associated with the bacterial pneumonia class.
* **Explainable AI (XAI):** Embedded **Grad-CAM** visualizations to generate post-hoc heatmaps, highlighting the exact pathological regions the model focuses on for clinical validation.
* **Deployment:** Built a real-time diagnostic interface utilizing **Streamlit**.

---

## 📊 Model Evaluation Summary
We experimented extensively with 6 distinct architectures: ResNet50, MobileNetV2, DenseNet121, EfficientNetB0, NASNetMobile, and VGG16. 

The fine-tuned **VGG16-based model** (with the last 60 layers unfrozen) combined with Focal Loss delivered the most robust results:
* **Validation Accuracy:** Achieved a final accuracy of **87.90%**.
* **Class-wise Performance:** Showed a significant reduction in confusion between viral and bacterial pneumonia compared to literature benchmarks.

---

## 🧑‍💻 My Core Role & Contributions
As the **Project Lead and Core AI Engineer**, I managed the project lifecycle and was responsible for dataset preparation, leading the model benchmarking strategy, hyperparameter tuning, and model explainability:

* **Dataset Diagnosis & Optimization:** During initial testing, I analyzed the evaluation metrics and diagnosed a critical performance drop specifically within viral pneumonia cases. Instead of leaving the dataset as is, I deduced that the model lacked feature diversity in this class. To solve this, I made the strategic decision to merge the COVID-19 image data directly into the viral pneumonia category, which expanded the training space and successfully corrected the accuracy drop.
* **Model Benchmarking & Architecture Evaluation:** Coordinated and led the team's benchmarking strategy across 6 distinct deep learning models (VGG16, ResNet50, MobileNetV2, DenseNet121, EfficientNetB0, and NASNetMobile). Personally focused on implementing and optimizing the core training pipelines, and ran comparative experiments proving that upgrading the input resolution from 224×224 to 600×600 pixels was essential to preserve critical medical details.
* **Hyperparameter Tuning & Regularization:** Designed and implemented custom classification layers on top of the pre-trained networks. I configured Global Average Pooling, L2 weight regularization (lambda = 0.001), and tuned Dropout rates between 0.4 and 0.5 to control and prevent overfitting during transfer learning.
* **Loss Function Innovation:** Analyzed confusion matrices from baseline models and discovered a high rate of misclassification between viral and bacterial pneumonia. To fix this inter-class confusion, I replaced standard cross-entropy with Focal Loss (gamma = 2.0, alpha = 0.25), forcing the networks to dynamically focus on hard-to-classify samples.
* **Explainable AI (Grad-CAM) Integration:** Programmed the Grad-CAM module to extract feature activation maps directly from the final convolutional layers. This visualizes exactly which lung regions the model utilizes for its predictions, ensuring the final VGG16 architecture is verifiable and trustworthy for clinical use.

---

## 📦 Tech Stack
* **Languages:** Python
* **Frameworks:** TensorFlow / Keras, OpenCV, Scikit-Learn
* **Deployment:** Streamlit
