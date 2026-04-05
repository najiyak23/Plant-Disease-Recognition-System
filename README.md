# 🌿 Plant Disease Recognition System

A **web application** to detect plant diseases from leaf images using **Convolutional Neural Networks (CNNs)** and **TensorFlow**, with a **user-friendly Streamlit interface** and a sidebar **plant chatbot** for guidance.  

---

## 🔗 Dataset

The dataset is from [Kaggle: New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) and consists of **87,000 RGB images** of healthy and diseased crop leaves, categorized into **38 classes**.  


---

## 🛠 Technical Stack

- **Frontend:** Streamlit  
- **Backend / ML:** TensorFlow, Keras  
- **Programming Language:** Python 3.x  
- **Libraries:**  
  - `tensorflow` – model training and inference  
  - `numpy` – array manipulations  
  - `matplotlib` – visualization  
  - `opencv-python` – image processing  

---

## 📌 Features

1. **Plant Disease Detection**  
   - Upload an image of a plant leaf  
   - Detects diseases from 38 classes  
   - Displays prediction with disease name  

2. **Visualizations**  
   - Accuracy & loss graphs  
   - Confusion matrix for validation set  

3. **Sidebar Chatbot**  
   - Rule-based bot for app guidance and plant disease information  
   - Answers questions like “How to use the app?” or “Explain tomato disease”  

4. **User-Friendly Interface**  
   - Multi-page layout: Home, About, Disease Recognition  
   - Quick predictions with visual feedback  

---

## ⚙️ How to Run Locally

1. **Clone the repository:**

   ```bash
    git clone https://github.com/najiyak23/Plant-Disease-Recognition-System.git
    cd Plant-Disease-Recognition-System
   ```

2. **Install dependencies:**
   
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app:**
   ```bash
   streamlit run main.py
   ```
4. **Open in browser:**
   
    - Home – Overview of the app
    - About – Dataset info and details
    - Disease Recognition – Upload leaf images to detect diseases
  

## 🖼 Usage

1. **Upload Image:**
   - Select a plant leaf image on the Disease Recognition page.
2. **Show Image:**
   - Click Show Image to preview it.
3. **Predict Disease:**
   - Click Predict to run the trained CNN model.
     - Disease name will be displayed
     - Snow animation 🎉 for visual feedback
4. **Chatbot Assistance:**
   - Ask questions in the sidebar chatbot for tips or crop-specific disease info.

## Model Details

- **Architecture:** CNN with 5 convolution blocks, dropout, and dense layers
- **Input size:** 128x128 RGB images
- **Output:** 38-class softmax
- **Training:**
  - Optimizer: Adam (learning rate = 0.0001)
  - Loss: Categorical Crossentropy
  - Epochs: 10
- **Performance:**
  - Training Accuracy: ~97.8%
  - Validation Accuracy: ~94.5%

## Evaluation Metrics

- Confusion matrix for 38 classes
- Classification report (precision, recall, f1-score)
  
## Future Improvements
- Integrate real-time camera input for leaf scanning
- Add more advanced chatbot with AI-based responses
- Support more crops and diseases
- Optimize model size for faster predictions

