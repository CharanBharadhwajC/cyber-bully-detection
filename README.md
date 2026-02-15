Cyber Bully Detection System

DistilBERT + Logistic Regression Fallback

🔍 Overview

This project is a real-time cyberbullying detection web application built using:

Fine-tuned DistilBERT (primary model)

TF-IDF + Logistic Regression (fallback model)

Flask web application

SQLite for flagged message logging

The system classifies text into:

hate_speech

offensive

safe

It combines deep learning with rule-based escalation to reduce false negatives.

🧠 Model Architecture
1️⃣ Primary Model

Fine-tuned DistilBERT (Hugging Face Transformers)

Tokenized with DistilBERT tokenizer

Softmax classification head

Confidence-based thresholding

Runs on CPU/GPU

2️⃣ Fallback Model

TF-IDF + Logistic Regression (scikit-learn)

Used if:

DistilBERT fails to load

Or inference errors occur

3️⃣ Advisory Layer (High Precision Rules)

Certain patterns are immediately escalated:

Explicit racial slurs

Direct violent threats

Explicit hate speech indicators

This prevents low-confidence misses.

🖥 Web Application

Built with:

Flask

Gunicorn (for production)

SQLite database

Features:

Real-time chat-style interface

Confidence score display

Admin panel for flagged messages

Statistics dashboard

REST API endpoint (/api/analyze)

📁 Project Structure
cyber_pj/
│
├── src/
│   ├── app.py
│   └── utils.py
│
├── templates/
├── static/
├── models/
│   ├── distilbert/
│   └── model_fallback.joblib
│
├── requirements.txt
├── README.md
└── .gitignore

⚙️ Installation (Local Setup)
1️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run application
python src/app.py


Open:

http://127.0.0.1:7860

🐳 Docker Deployment (Optional)

Build image:

docker build -t cyber-bully-detector .


Run container:

docker run -p 7860:7860 cyber-bully-detector

🌐 Hugging Face Deployment

Live Demo:
👉 [YOUR_HUGGINGFACE_LINK_HERE]

📊 Example Predictions
Input	Prediction
I love my family	SAFE
You idiot	OFFENSIVE
You filthy nigger	HATE_SPEECH
I will kill you	HATE_SPEECH
🛡 Safety Design Philosophy

This system prioritizes:

Minimizing false negatives for hate speech

Conservative rule escalation

Confidence threshold control

Hybrid ML + rule-based detection

🚀 Future Improvements

Better contextual bias detection

Multi-language support

Adversarial robustness testing

Model quantization for faster inference

Confidence calibration tuning

📌 Technologies Used

Python 3.10

PyTorch

Hugging Face Transformers

scikit-learn

Flask

SQLite

Gunicorn

📄 License

This project is for educational and research purposes.