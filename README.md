# 🐦 Bird Species Identification using CNN

## 📘 Overview
This project identifies bird species from uploaded images using a **Convolutional Neural Network (CNN)** model trained on a labeled dataset.  
It combines **Deep Learning (TensorFlow/Keras)** for model training and **Flask** for deployment, allowing users to upload bird images and get instant predictions through a simple web interface.

---

## 🚀 Features
- 🧠 CNN-based model using **MobileNetV2** for feature extraction  
- 📊 Computes **Accuracy, Precision, Recall, and F1-Score**  
- 🌐 Flask web app for easy image upload and prediction  
- ⚙️ Modular code for training and inference  
- 🖼️ Dataset organized into `train/` and `test/` folders  

---

## 📁 Project Structure
```
Bird-Species-Identification/
│
├── app.py                     # Flask backend for predictions
├── trainmodel.py              # Model training script
├── bird_species_model.h5      # Saved CNN model
├── dataset/
│   ├── train/                 # Training images
│   └── test/                  # Testing images
├── templates/
│   └── birds.html             # Frontend page
├── static/                    # (optional) CSS, JS, or images
├── uploads/                   # Temporary uploaded images
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

---

## 🧩 Technologies Used
| Category | Technology |
|-----------|-------------|
| Programming Language | Python 3.x |
| Deep Learning | TensorFlow / Keras |
| Web Framework | Flask |
| Evaluation | scikit-learn |
| Dataset Source | Kaggle Bird Species Dataset |
| Visualization | tqdm, Pillow |

---

## ⚙️ Installation

### 1️⃣ Clone this repository
```bash
git clone https://github.com/yourusername/bird-species-identification.git
cd bird-species-identification
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare dataset
Ensure your dataset is structured as:
```
dataset/
 ├── train/
 │    ├── species1/
 │    ├── species2/
 │    └── ...
 └── test/
      ├── species1/
      ├── species2/
      └── ...
```

### 4️⃣ Train the model (optional)
If you wish to retrain the model:
```bash
python trainmodel.py
```

### 5️⃣ Run the Flask app
```bash
python app.py
```
Then open your browser and visit:  
**http://127.0.0.1:5000**

---

## 📊 Model Performance
| Metric | Score |
|---------|--------|
| Accuracy | ~92% |
| Precision | ~91% |
| Recall | ~90% |
| F1-Score | ~90% |

*(Scores may vary depending on dataset and training epochs.)*

---

## 🧠 Model Architecture
- **Base Model:** MobileNetV2 (pretrained on ImageNet)  
- **Fine-tuned Layers:** Last 60 layers unfrozen for training  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Cross-Entropy  
- **Metrics:** Accuracy  

---

## 🌈 Web App Demo
1. Upload an image (`.jpg`, `.jpeg`, `.png`, `.gif`)  
2. The app predicts the **bird species** and shows **confidence score**  
3. Displays overall model metrics below the prediction  

---

## 📚 Future Enhancements
- 📱 Develop a **mobile application** version  
- 🧩 Add **Grad-CAM** visualization for feature interpretability  
- 🌍 Deploy the model on **Render / Hugging Face / AWS**  
- 📦 Expand dataset for more bird species  

---

## 👨‍💻 Author
**Naveen S**  
Mini Project – Department of Computer Science  
Guided by **Dr. Reena Murali**

---

## 🪶 License
This project is created for educational and academic use under the **MIT License**.

---

## 🖼️ (Optional) Add Screenshots
You can place screenshots inside a `screenshots/` folder and reference them here:
```markdown
![App Interface](screenshots/web_interface.png)
![Model Performance Graph](screenshots/model_accuracy.png)
```
