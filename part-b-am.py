#Part B:

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


np.random.seed(42)
n = 2000

df = pd.DataFrame({
    'annual_income':    np.random.randint(30_000, 150_000, n),
    'credit_score':     np.random.randint(300, 850, n),
    'loan_amount':      np.random.randint(5_000, 100_000, n),
    'employment_years': np.random.randint(0, 40, n),
    'debt_to_income':   np.round(np.random.uniform(0.05, 0.60, n), 2),
    'num_credit_cards': np.random.randint(0, 13, n),
})

def decide(row):
    score = 0
    if row['credit_score'] > 700:    score += 3
    elif row['credit_score'] > 650:  score += 1
    if row['debt_to_income'] < 0.35: score += 2
    if row['employment_years'] > 5:  score += 1
    if row['annual_income'] > 80_000: score += 1
    score += np.random.choice([-1, 0, 0, 0, 1])
    return 1 if score >= 4 else 0

df['approved'] = df.apply(decide, axis=1)
X = df.drop('approved', axis=1)
y = df['approved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# (a)

print("=" * 60)
print("(a) How does splitting differ?")
print("=" * 60)
print("""
Random Forest:
  - At each node, it picks the BEST split from a random subset of features.
  - It evaluates every possible threshold for those features and picks the one
    that gives the highest information gain (or lowest Gini impurity).

Extra Trees (Extremely Randomized Trees):
  - At each node, it picks RANDOM thresholds for each feature in the random subset.
  - Instead of searching for the best split, it just picks a random cut-point
    for each feature and then chooses the best among those random splits.
  - This makes each individual tree weaker but MORE diverse.
  - More diversity = better ensemble = less overfitting.

Key difference: RF finds the BEST split, ExtraTrees uses RANDOM splits.
""")


# (b)

print("=" * 60)
print("(b) Speed Comparison")
print("=" * 60)

rf  = RandomForestClassifier(n_estimators=200, random_state=42)
et  = ExtraTreesClassifier(n_estimators=200, random_state=42)


start = time.time()
rf.fit(X_train, y_train)
rf_train_time = time.time() - start

start = time.time()
rf.predict(X_test)
rf_pred_time = time.time() - start


start = time.time()
et.fit(X_train, y_train)
et_train_time = time.time() - start

start = time.time()
et.predict(X_test)
et_pred_time = time.time() - start

print(f"  Random Forest  — Train: {rf_train_time:.4f}s  |  Predict: {rf_pred_time:.4f}s")
print(f"  Extra Trees    — Train: {et_train_time:.4f}s  |  Predict: {et_pred_time:.4f}s")
print(f"\n  ExtraTrees is typically FASTER because it skips the 'find best threshold'")
print(f"  step and just picks random thresholds. Less computation per split.\n")


# (c)

print("=" * 60)
print("(c) Performance Comparison on Loan Dataset")
print("=" * 60)

models = {
    'Random Forest': rf,
    'Extra Trees':   et,
}

results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results.append({
        'Model':       name,
        'Accuracy':    accuracy_score(y_test, y_pred),
        'F1-Score':    f1_score(y_test, y_pred),
        'ROC-AUC':     roc_auc_score(y_test, y_prob),
        'CV Mean Acc': cv_scores.mean(),
        'CV Std':      cv_scores.std(),
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))





- **RF** finds the best threshold for each feature at every split.
- **ExtraTrees** picks random thresholds, then chooses the best among them.
- This makes ExtraTrees faster but individual trees are weaker.


- ExtraTrees trains faster because skipping the threshold search saves computation.
- Amazon and Netflix use ExtraTrees in real-time pipelines where speed matters.


- On our loan dataset, both models performed comparably.
- ExtraTrees sometimes slightly beats RF because the added randomness reduces
  overfitting (more diversity in the ensemble).


- When you need faster training with similar accuracy.
- When your dataset is large and you want to reduce computation.
- When overfitting is a concern (more randomness = less overfitting).



