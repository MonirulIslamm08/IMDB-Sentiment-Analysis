
# 🎭 IMDB Sentiment Analysis  

## Overview  
This project performs sentiment analysis on IMDB movie reviews using Natural Language Processing (NLP) techniques. It explores various machine learning models, including Logistic Regression, Random Forest, XGBoost, and deep learning with LSTMs.  

## Features  
✅ Data Preprocessing (Tokenization, Lemmatization, Stopword Removal)  
✅ Exploratory Data Analysis (EDA) with Visualizations  
✅ Bag of Words (BoW) and TF-IDF Vectorization  
✅ Word2Vec Embeddings  
✅ Model Training & Evaluation (Logistic Regression, Random Forest, XGBoost)  
✅ Deep Learning with LSTM for Text Classification  
✅ Deployment with Flask API  

## Dataset 📂  
The dataset used is the **IMDB Movie Reviews Dataset**, containing 50,000 labeled reviews (positive/negative).  

## Installation 🚀  
```bash
git clone https://github.com/yourusername/imdb-sentiment-analysis.git  
cd imdb-sentiment-analysis  
pip install -r requirements.txt  
```

## Usage 🛠  
#### Train the Model  
```python
python train.py  
```
#### Run Sentiment Prediction API  
```bash
python app.py  
```
#### Example API Request  
```json
POST /predict  
{
  "text": "The movie was fantastic! Great acting and storyline."
}
```

## Results 📊  
| Model | Accuracy |
|--------|----------|
| Logistic Regression | 87.3% |
| Random Forest | 84.4% |
| XGBoost | 86.9% |
| LSTM | 89.5% |

## Technologies Used 🔧  
- Python 🐍  
- Pandas, NumPy  
- Scikit-learn, XGBoost  
- TensorFlow/Keras (LSTM)  
- Flask (API Deployment)  
- Matplotlib, Seaborn (Visualization)  

## Deployment 🌍  
The model is deployed as a REST API using Flask.  

## Future Improvements 🚀  
🔹 Improve LSTM performance with hyperparameter tuning  
🔹 Deploy using FastAPI or Flask on cloud services  
🔹 Experiment with transformer models like BERT  

## Author 👨‍💻  
📧 Email: *your.email@example.com*  
🔗 LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/your-profile)  
🐙 GitHub: [Your GitHub](https://github.com/yourusername)  
