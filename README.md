SPAM SMS DETECTION USING MACHINE LEARNING
========================================

PROJECT OVERVIEW
----------------
This project classifies SMS messages as either Spam or Ham (Not Spam) using Natural Language Processing (NLP) and Machine Learning.

The objective is to automatically identify unwanted spam messages and improve message filtering systems.

DATASET
-------
SMS Spam Collection Dataset

Dataset Features:
- Text Message
- Label

Classes:
- Ham (Legitimate Message)
- Spam (Unwanted Message)

TECHNOLOGIES USED
-----------------
- Python
- Pandas
- NumPy
- NLTK
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

PROJECT WORKFLOW
----------------
1. Load Dataset
2. Data Cleaning
3. Text Preprocessing
4. Stopword Removal
5. TF-IDF Vectorization
6. Train-Test Split
7. Logistic Regression Training
8. Prediction
9. Model Evaluation
10. Real-Time User Message Testing

TEXT PREPROCESSING
------------------
Performed NLP preprocessing steps:

- Lowercase Conversion
- URL Removal
- Special Character Removal
- Punctuation Removal
- Extra Space Removal
- Stopword Removal

NLP TECHNIQUES
--------------
- Tokenization
- Stopword Removal
- TF-IDF Vectorization

MACHINE LEARNING MODEL
----------------------
Model Used:
- Logistic Regression

FEATURE EXTRACTION
------------------
TF-IDF Vectorizer was used to convert SMS text into numerical feature vectors for machine learning.

MODEL EVALUATION
----------------
Metrics Used:

- Accuracy Score
- Confusion Matrix
- Classification Report
- Precision
- Recall
- F1 Score

RESULTS
-------
Model Accuracy:
~94%

Confusion Matrix:
- Correctly Classified Most Ham Messages
- Correctly Identified Spam Messages
- Low False Positive Rate

KEY FINDINGS
------------
- TF-IDF effectively converts text into machine-readable features.
- Logistic Regression performs strongly on SMS classification.
- The model achieves high accuracy while remaining computationally efficient.
- Spam messages contain identifiable word patterns that can be learned by the model.

REAL-TIME PREDICTION
--------------------
The project includes a custom user input system.

Example:

Input:
"free money win now"

Output:
SPAM

Probability:
0.85

APPLICATIONS
------------
- Email Filtering
- SMS Spam Detection
- Customer Communication Systems
- Cybersecurity Applications
- Fraud Detection Systems

SKILLS DEMONSTRATED
-------------------
- Natural Language Processing (NLP)
- Text
