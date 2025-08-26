import argparse
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

def extract_meta_features(df):
    meta = pd.DataFrame()
    meta["num_urls"] = df["text"].apply(lambda x: len(re.findall(r"http[s]?://", str(x))))
    meta["num_exclamation"] = df["text"].apply(lambda x: str(x).count("!"))
    meta["num_currency"] = df["text"].apply(lambda x: str(x).count("$"))
    meta["text_length"] = df["text"].apply(lambda x: len(str(x)))
    meta["upper_pct"] = df["text"].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1))
    meta["has_ip_url"] = df["text"].apply(lambda x: 1 if re.search(r"http[s]?://\d+\.\d+\.\d+\.\d+", str(x)) else 0)
    meta["spf_fail"] = (df["spf"] == "fail").astype(int)
    meta["dkim_fail"] = (df["dkim"] == "fail").astype(int)
    meta["reply_to_mismatch"] = (df["from"] != df["reply_to"]).astype(int)
    meta["return_path_mismatch"] = (df["from"] != df["return_path"]).astype(int)
    return meta

def fitness(chromosome, X, y, alpha=0.01):
    if np.sum(chromosome) == 0:
        return 0
    X_sel = X[:, chromosome == 1]
    clf = LinearSVC(max_iter=2000)
    scores = cross_val_score(clf, X_sel, y, cv=3, scoring="f1")
    return scores.mean() - alpha * (np.sum(chromosome) / len(chromosome))

def run_ga(X, y, pop_size=20, gens=10, alpha=0.01, mutation_rate=0.1):
    n_features = X.shape[1]
    population = np.random.randint(0, 2, size=(pop_size, n_features))
    best_scores = []

    for gen in range(gens):
        fitnesses = np.array([fitness(ind, X, y, alpha) for ind in population])
        best_idx = np.argmax(fitnesses)
        best_scores.append(fitnesses[best_idx])
        print(f"Gen {gen+1}: Best F1 = {best_scores[-1]:.4f}")

        probs = fitnesses - fitnesses.min() + 1e-6
        probs = probs / probs.sum()
        new_pop = []
        for _ in range(pop_size // 2):
            parents = population[np.random.choice(pop_size, 2, p=probs)]
            point = np.random.randint(1, n_features - 1)
            child1 = np.concatenate([parents[0][:point], parents[1][point:]])
            child2 = np.concatenate([parents[1][:point], parents[0][point:]])
            for child in [child1, child2]:
                for i in range(n_features):
                    if np.random.rand() < mutation_rate:
                        child[i] = 1 - child[i]
            new_pop.extend([child1, child2])
        population = np.array(new_pop)

    fitnesses = np.array([fitness(ind, X, y, alpha) for ind in population])
    best_idx = np.argmax(fitnesses)
    return population[best_idx], best_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="sample_emails.csv")
    parser.add_argument("--max_text_features", type=int, default=500)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--gens", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.01)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    y = df["label"].values

    tfidf = TfidfVectorizer(max_features=args.max_text_features, ngram_range=(1,2))
    X_text = tfidf.fit_transform(df["text"]).toarray()
    X_meta = extract_meta_features(df).values
    X = np.hstack([X_text, StandardScaler().fit_transform(X_meta)])

    best_chrom, scores = run_ga(X, y, pop_size=args.pop, gens=args.gens, alpha=args.alpha)

    plt.plot(scores)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("GA Fitness Curve")
    plt.show()

    print("Selected features:", np.where(best_chrom == 1)[0])
