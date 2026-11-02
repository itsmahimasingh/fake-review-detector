# 🔍 Fake Review Detector

A machine learning web app that predicts whether a product review is genuine or fake (computer-generated).

🔗 **Live demo:** [fake-review-detector-mahima.streamlit.app](https://fake-review-detector-mahima.streamlit.app)
📓 **Full analysis on Kaggle:** [Spotting AI-Written Reviews](https://www.kaggle.com/code/mahimasinghb25bs3283/spotting-ai-written-reviews)

## 📌 Problem

Online shoppers rely on reviews to make decisions, but many reviews are fake or auto-generated to boost or damage a product's rating. This project uses Natural Language Processing (NLP) to flag reviews that are likely fake.

## ✨ Features

- Paste any product review and get an instant Genuine / Fake prediction
- Shows the model's confidence score
- Explains which signals pushed the prediction towards fake or genuine

## 🛠️ Tech Stack

- **Language:** Python
- **ML / NLP:** scikit-learn (TF-IDF + Logistic Regression)
- **Web App:** Streamlit
- **Model Storage:** joblib
- **Training:** Google Colab

## 📊 Dataset

Fake Reviews Dataset (Salminen et al.) from Kaggle: 40,432 product reviews, half genuine and half computer-generated, across product categories.

## 🧠 How It Works

1. **Vectorization:** review text is converted into numbers using TF-IDF (single words and word pairs)
2. **Style features:** 8 simple counts are added, such as review length, exclamation marks, use of "I/my/me", vocabulary variety and star rating
3. **Classification:** a Logistic Regression model predicts whether the review is genuine or fake
4. **Deployment:** the trained model is saved as `detector.joblib` and loaded by the Streamlit app

## 📈 Results

| Model | Accuracy | F1 |
|---|---|---|
| Baseline (TF-IDF only) | 89.4% | 0.893 |
| TF-IDF + 8 style features | 90.5% | 0.904 |

## 📁 Project Structure

```
fake-review-detector/
├── .streamlit/
│   └── config.toml      # Streamlit theme settings
├── app.py               # Streamlit web app
├── detector.joblib      # Trained ML model
└── requirements.txt     # Python dependencies
```

## 🚀 Run Locally

```bash
git clone https://github.com/itsmahimasingh/fake-review-detector.git
cd fake-review-detector
pip install -r requirements.txt
streamlit run app.py
```

## 🔮 Future Improvements

- Use transformer models (e.g. BERT) for higher accuracy
- Support bulk checking of reviews from a CSV file
- Test on reviews written by newer AI tools

## 👩‍💻 Author

**Mahima Singh**, BS in Artificial Intelligence & Data Science, IIT Jodhpur
GitHub: [@itsmahimasingh](https://github.com/itsmahimasingh)
