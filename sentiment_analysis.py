# --- Import Libraries ---
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download required NLTK data 
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# --- Step 1: Create a small sample dataset ---
positive = [
    "I love this product, it’s fantastic!",
    "Great experience, I’m very satisfied.",
    "Amazing quality and excellent service."
]

negative = [
    "Terrible item, completely disappointed.",
    "Worst purchase ever, I hate it.",
    "Bad experience, not worth the money."
]

neutral = [
    "The product is okay, nothing special.",
    "It works as expected, average quality.",
    "It’s fine, not too bad or good."
]

texts = positive + negative + neutral
labels = (["positive"] * len(positive)) + (["negative"] * len(negative)) + (["neutral"] * len(neutral))

# --- Step 2: Text Preprocessing ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(w) for w in words if w.isalpha() and w not in stop_words]
    return " ".join(words)

cleaned_texts = [clean_text(t) for t in texts]

# --- Step 3: Split data ---
X_train, X_test, y_train, y_test = train_test_split(cleaned_texts, labels, test_size=0.3, random_state=42)

# --- Step 4: Vectorize and Train Model ---
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# --- Step 5: Evaluate ---
y_pred = model.predict(X_test_vec)
print("\nModel Evaluation:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# --- Step 6: Try a few example predictions ---
def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    return prediction

print("\n===================n")
examples = [
    "I really love this phone!",
    "The service was terrible.",
    "It’s fine, does what it should."
]

for ex in examples:
    print(f"{ex} --> {predict_sentiment(ex)}")
