from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# -----------------------------
# NLTK setup for server deployment
# -----------------------------
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download only the required resources
nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Dataset
# -----------------------------
texts = [
    "I love this product", "Great experience", "Amazing service",
    "Terrible item", "Worst purchase", "Bad experience",
    "The product is okay", "Average quality", "It is fine"
]
labels = [
    "positive", "positive", "positive",
    "negative", "negative", "negative",
    "neutral", "neutral", "neutral"
]

# -----------------------------
# Preprocessing
# -----------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(w) for w in words if w.isalpha() and w not in stop_words]
    return " ".join(words)

cleaned_texts = [clean_text(t) for t in texts]

# -----------------------------
# Vectorization & Model Training
# -----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_texts)

model = LogisticRegression(max_iter=200)
model.fit(X, labels)

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    if request.method == "POST":
        text = request.form["text"]
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        sentiment = model.predict(vec)[0]

    return render_template("index.html", sentiment=sentiment)

# -----------------------------
# Run the app
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
