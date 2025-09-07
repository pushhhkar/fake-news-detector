# 📰 Fake News Detector

An end-to-end Machine Learning project that detects whether a news article is **FAKE** or **REAL**.  
Built with `scikit-learn` and deployed using **Streamlit Cloud**.

## 🚀 Live Demo  
Try it here 👉 [Fake News Detector](https://fake-news-detector-hgcdfrk3hnqvfq6wd8jxhd.streamlit.app/)

---

## 📌 Project Overview
- **Goal**: Build a system that can classify news articles as **FAKE** or **REAL**.  
- **Dataset**: Kaggle "Fake and Real News Dataset" (two files: `Fake.csv` and `True.csv`).  
- **Model**: `PassiveAggressiveClassifier` trained on TF-IDF vectorized text.  
- **Deployment**: Hosted publicly on Streamlit Cloud.  

---

## 🛠️ How It Works
1. **Data Preparation**  
   - Merged `Fake.csv` and `True.csv` into a single dataset.  
   - Cleaned missing values and labeled data (`FAKE` / `REAL`).  

2. **Training**  
   - Used `TfidfVectorizer` for text feature extraction.  
   - Trained `PassiveAggressiveClassifier`.  
   - Achieved **~99% training accuracy**.  

3. **Deployment**  
   - Saved model (`model.pkl`) and vectorizer (`vectorizer.pkl`).  
   - Built a Streamlit app (`app.py`) to make predictions.  

---

## 📂 Files in Repository
- `prepare_data.py` → dataset cleaning & merging script  
- `train.py` → model training script  
- `app.py` → Streamlit web app  
- `model.pkl` & `vectorizer.pkl` → trained model and text processor  
- `requirements.txt` → dependencies for Streamlit Cloud  

---

## 🔎 Limitations
This project is trained only on the **Kaggle dataset**, where:
- “REAL” news mostly comes from Reuters.  
- “FAKE” news comes from specific unreliable sources.  

⚠️ Because of this bias, the model may misclassify **legit news from BBC, CNN, etc.** as FAKE.  
Improving generalization requires:  
- Training on a more diverse dataset (multiple news outlets).  
- Using advanced NLP models (e.g., BERT, DistilBERT).  

---

## 🧭 Future Improvements
- Use **transformer models** (BERT) for better accuracy.  
- Add **explainability** (SHAP/LIME) to show which words influenced predictions.  
- Extend to **multimodal fake news detection** (text + images).  

---

## 💡 How to Run Locally
```bash
# Clone repo
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
