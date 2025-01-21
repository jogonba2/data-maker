from collections import Counter
import numpy as np
from scipy.spatial.distance import jensenshannon
import multiprocessing as mp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import bm25s


def compute_jsd(synthetic_texts, real_texts):
    # Concat the texts
    synthetic = " ".join(synthetic_texts)
    real = " ".join(real_texts)

    # Tokenize both texts
    tokens1 = [
        token
        for token in synthetic.lower().strip().split()
        if token not in ENGLISH_STOP_WORDS
    ]
    tokens2 = [
        token
        for token in real.lower().strip().split()
        if token not in ENGLISH_STOP_WORDS
    ]

    # Count token frequencies in both texts
    count1 = Counter(tokens1)
    count2 = Counter(tokens2)

    # Get total number of tokens in both texts
    total1 = len(tokens1)
    total2 = len(tokens2)

    # Convert token counts to probability distributions
    prob1 = {word: count / total1 for word, count in count1.items()}
    prob2 = {word: count / total2 for word, count in count2.items()}

    # Find all unique words in both texts
    all_words = set(prob1.keys()).union(set(prob2.keys()))

    # Create probability vectors for each text, ensuring all words are included
    prob_vector1 = np.array([prob1.get(word, 0) for word in all_words])
    prob_vector2 = np.array([prob2.get(word, 0) for word in all_words])

    # Compute the Jensen-Shannon Divergence using scipy
    jsd = jensenshannon(prob_vector1, prob_vector2)

    return jsd


def bm25(queries, corpus, labels):
    # https://huggingface.co/blog/xhluca/bm25s
    # https://arxiv.org/pdf/2407.03618
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus))
    results, scores = retriever.retrieve(bm25s.tokenize(queries), k=1)
    doc_ids = results[:, 0]
    scores = scores[:, 0]
    retrieved_labels = [labels[doc_id] for doc_id in doc_ids]
    retrieved_docs = [corpus[doc_id] for doc_id in doc_ids]
    return retrieved_docs, retrieved_labels, scores
