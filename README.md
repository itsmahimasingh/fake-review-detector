Fake Review Detector
A machine learning web app that predicts whether a product review is genuine or fake (computer-generated).
🔗 Live demo: fake-review-detector.streamlit.app <!-- replace with your real link -->
📌 Problem
Online shoppers rely on reviews to make decisions, but many reviews are fake or auto-generated to boost or damage a product's rating. This project uses Natural Language Processing (NLP) to flag reviews that are likely fake.
✨ Features
Paste any product review and get an instant Genuine / Fake prediction
Shows the model's confidence score <!-- delete this line if your app doesn't show it -->
Simple, fast web interface built with Streamlit
🛠️ Tech Stack
Language: Python
ML / NLP: scikit-learn (TF-IDF + [Model name, e.g. Logistic Regression])
Web App: Streamlit
Model Storage: joblib
Training: Google Colab
📊 Dataset
Fake Reviews Dataset from Kaggle, containing genuine reviews and computer-generated fake reviews across product categories. <!-- paste the exact Kaggle link -->
🧠 How It Works
Preprocessing: reviews are cleaned (lowercasing, removing punctuation and stopwords)
Vectorization: text is converted into numbers using TF-IDF
Classification: a trained ML model predicts whether the review is genuine or fake
Deployment: the trained pipeline is saved as detector.joblib and loaded by the Streamlit app
<!-- fill these in from your Colab notebook -->
📁 Project Structure
fake-review-detector/
├── .streamlit/
│   └── config.toml      # Streamlit theme settings
├── app.py               # Streamlit web app
├── detector.joblib      # Trained ML model
└── requirements.txt     # Python dependencies
🚀 Run Locally
bash
git clone https://github.com/itsmahimasingh/fake-review-detector.git
cd fake-review-detector
pip install -r requirements.txt
streamlit run app.py
🔮 Future Improvements
Use transformer models (e.g. BERT) for higher accuracy
Support bulk checking of reviews from a CSV file
Highlight words that most influenced the prediction
👩‍💻 Author
Mahima Singh BS in Artificial Intelligence & Data Science, IIT Jodhpur GitHub: @itsmahimasingh
