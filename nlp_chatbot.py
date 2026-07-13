import random
import pickle
import os

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from chatbot import KNOWLEDGE_BASE, regex_match

# ============================================================
# ONE-TIME NLTK DATA DOWNLOAD (safe to run every time - skips if already present)
# ============================================================
for pkg in ['punkt', 'punkt_tab', 'wordnet', 'stopwords', 'averaged_perceptron_tagger']:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {'not', 'no', 'never', 'very', 'too'}


# ============================================================
# PART 1: ADVANCED PREPROCESSING + BUILD TRAINING DATA
# ============================================================
def advanced_preprocess(text):
    """Full NLP preprocessing: tokenise, lemmatise, remove stop words."""
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


X_raw = []
X_proc = []
y = []

for entry in KNOWLEDGE_BASE:
    if entry['tag'] == 'unknown':
        continue
    for pattern in entry['patterns']:
        X_raw.append(pattern)
        X_proc.append(advanced_preprocess(pattern))
        y.append(entry['tag'])

print(f'Training samples: {len(X_proc)}')
print(f'Unique tags     : {len(set(y))}')


# ============================================================
# PART 2: TF-IDF + LOGISTIC REGRESSION TRAINING
# ============================================================
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, analyzer='word')
X_tfidf = tfidf.fit_transform(X_proc)

print(f'Vocabulary size: {len(tfidf.vocabulary_)}')
print(f'TF-IDF matrix  : {X_tfidf.shape}')

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

lr_clf = LogisticRegression(max_iter=500, random_state=42)
lr_clf.fit(X_tr, y_tr)
lr_preds = lr_clf.predict(X_te)

print(f'\nLogistic Regression Accuracy: {accuracy_score(y_te, lr_preds):.4f}')
print('\nClassification Report:')
print(classification_report(y_te, lr_preds, target_names=le.classes_))


# ============================================================
# PART 3: COSINE SIMILARITY INTENT DETECTION
# ============================================================
def predict_intent_cosine(user_input, threshold=0.25):
    """Predict intent using TF-IDF cosine similarity."""
    processed = advanced_preprocess(user_input)
    user_vector = tfidf.transform([processed])

    similarities = cosine_similarity(user_vector, X_tfidf)[0]
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    best_tag = y[best_idx]

    print(f'  [Debug] Best match: "{X_raw[best_idx]}" | Score: {best_score:.3f} | Tag: {best_tag}')

    if best_score < threshold:
        best_tag = 'unknown'

    for entry in KNOWLEDGE_BASE:
        if entry['tag'] == best_tag:
            return best_tag, random.choice(entry['responses']), best_score

    for entry in KNOWLEDGE_BASE:
        if entry['tag'] == 'unknown':
            return 'unknown', random.choice(entry['responses']), 0.0


def run_intent_tests():
    test_inputs = [
        'hi there how are you',
        'what is the scholarship amount',
        'I need information about accommodation',
        'what jobs will I get after graduating',
        'I want to study computer science',
    ]
    print('=== NLP Intent Detection Test ===')
    for inp in test_inputs:
        tag, resp, score = predict_intent_cosine(inp)
        print(f'Input: "{inp}"')
        print(f'Detected Intent: {tag} (similarity: {score:.3f})')
        print(f'Response: {resp[:60]}...')
        print('-' * 55)


# ============================================================
# PART 4: FULL NLP CHAT LOOP
# ============================================================
def nlp_chat():
    print('=' * 60)
    print(' CollegeBot NLP Edition - AI-Powered Assistant')
    print(' Main Flow Services and Technologies Pvt. Ltd.')
    print('=' * 60)
    print('Ask me anything about the college! Type "bye" to exit.\n')

    history = []
    turn = 0

    while True:
        user_input = input('You: ').strip()
        if not user_input:
            continue

        turn += 1
        history.append(user_input)

        tag, response, score = predict_intent_cosine(user_input)

        if score < 0.15:
            tag2, response = regex_match(user_input)
            print(f'  [Switched to rule-based: tag={tag2}]')
            tag = tag2
        else:
            print(f'  [NLP detected intent: {tag}, confidence: {score:.2f}]')

        print(f'Bot: {response}\n')

        if tag == 'farewell':
            print(f'Session complete. Total turns: {turn}')
            break


# ============================================================
# PART 5: SAVE / LOAD MODEL
# ============================================================
MODEL_PATH = 'chatbot_tfidf.pkl'
ENCODER_PATH = 'chatbot_encoder.pkl'
LABELS_PATH = 'chatbot_labels.pkl'


def save_model():
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(tfidf, f)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(lr_clf, f)
    with open(LABELS_PATH, 'wb') as f:
        pickle.dump((X_tfidf, y, X_raw), f)
    print('Model saved!')


def load_model():
    global tfidf, lr_clf, X_tfidf, y, X_raw
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            tfidf = pickle.load(f)
        with open(ENCODER_PATH, 'rb') as f:
            lr_clf = pickle.load(f)
        with open(LABELS_PATH, 'rb') as f:
            X_tfidf, y, X_raw = pickle.load(f)
        print('Model loaded from disk!')
        return True
    return False


# ============================================================
# PART 6: EVALUATION - CONFUSION MATRIX
# ============================================================
def plot_confusion_matrix():
    y_pred_labels = le.inverse_transform(lr_preds)
    y_true_labels = le.inverse_transform(y_te)

    unique_tags = sorted(set(y))
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=unique_tags)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_tags, yticklabels=unique_tags,
                linewidths=0.5, annot_kws={'size': 11})
    plt.title('Chatbot Intent Classification - Confusion Matrix',
              fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('True Intent', fontsize=11)
    plt.xlabel('Predicted Intent', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('chatbot_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print('Confusion matrix saved as chatbot_confusion_matrix.png')
    plt.show()

    print(f'\nFinal Accuracy: {accuracy_score(y_te, lr_preds):.4f}')


# ============================================================
# PART 7: ENTRY POINT
# ============================================================
if __name__ == '__main__':
    run_intent_tests()
    save_model()
    plot_confusion_matrix()
    nlp_chat()