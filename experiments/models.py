from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from symanto_dec import DeClassifier
from sentence_transformers import SentenceTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

DEC_MODEL_NAME_OR_PATH = "sentence-transformers/paraphrase-mpnet-base-v2"
ENCODER = SentenceTransformer(DEC_MODEL_NAME_OR_PATH)


def fs_dec_tuning(X_tr, y_tr, X_te, y_te):
    # Train the model
    model = DeClassifier(DEC_MODEL_NAME_OR_PATH)
    model.encoder = ENCODER
    model.fit(X_tr, y_tr, strategy="lt")
    # Predict
    preds = model.predict(X_te)
    # Evaluate
    f1 = f1_score(y_true=y_te, y_pred=preds, average="macro")
    return preds, f1


def fs_lr_tuning(X_tr, y_tr, X_te, y_te):
    pipeline = make_pipeline(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2)),
        LogisticRegression(),
    )
    # Train the model
    pipeline.fit(X_tr, y_tr)
    # Predict
    preds = pipeline.predict(X_te)
    # Evaluate
    f1 = f1_score(y_true=y_te, y_pred=preds, average="macro")
    return preds, f1
