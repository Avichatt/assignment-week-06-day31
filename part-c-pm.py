 Part C:

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# Q1: Conceptual



1. COMPUTE COST:
   - 1000 trees takes 10x more time and memory to train than 100 trees.
   - If your dataset is large, this can mean hours vs minutes.
   - In a production environment, this means higher cloud costs.

2. PREDICTION LATENCY:
   - At prediction time, 1000 trees is 10x slower than 100 trees.
   - For a web app that needs to respond in <100ms, this matters A LOT.
   - For batch predictions (running overnight), it matters less.

3. MARGINAL IMPROVEMENT:
   - If accuracy is truly identical, the extra 900 trees add ZERO value.
   - In Random Forest, accuracy typically plateaus after a certain number
     of trees. Adding more doesn't hurt accuracy, but doesn't help either.
   - However, 1000 trees gives slightly smoother probability estimates
     and more stable feature importances (less variance between runs).

4. PRODUCTION DEPLOYMENT:
   - Model size: 1000 trees = ~10x larger model file to store and load.
   - Memory: the model takes more RAM when loaded for serving.
   - Scaling: if you serve millions of requests, the extra cost adds up.

MY RECOMMENDATION:
   Use 100 trees. Since accuracy is the same, the extra 900 trees waste
   resources. The only reason to keep 1000 would be if you need very stable
   probability estimates or feature importances for reporting. In production,
   you want the smallest model that meets your accuracy requirements.

   A good practice: plot accuracy vs n_estimators to find the "elbow" where
   accuracy plateaus, then use that number + a small buffer.



# Q2: 


np.random.seed(42)
n = 1000
X_demo = np.random.randn(n, 5)
y_demo = (X_demo[:, 0] + X_demo[:, 1] * 2 + np.random.randn(n) * 0.5 > 1).astype(int)


def compare_models(X, y, models_dict):
   
    Trains each model with 5-fold CV and returns a DataFrame with
    mean and std of accuracy, F1, and training time for each model.

    Parameters:
        X : features (array-like)
        y : target (array-like)
        models_dict : dict of {model_name: model_object}

    Returns:
        pd.DataFrame with columns: Model, Accuracy_Mean, Accuracy_Std,
                                    F1_Mean, F1_Std, Train_Time_Sec
   
    results = []

    for name, model in models_dict.items():
        # Time the training (full CV)
        start = time.time()
        acc_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        f1_scores  = cross_val_score(model, X, y, cv=5, scoring='f1')
        train_time = time.time() - start

        results.append({
            'Model':         name,
            'Accuracy_Mean': round(acc_scores.mean(), 4),
            'Accuracy_Std':  round(acc_scores.std(), 4),
            'F1_Mean':       round(f1_scores.mean(), 4),
            'F1_Std':        round(f1_scores.std(), 4),
            'Train_Time_Sec': round(train_time, 3),
        })

    return pd.DataFrame(results)



models = {
    'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest (100)': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
}

comparison_df = compare_models(X_demo, y_demo, models)
print(comparison_df.to_string(index=False))


# Q3: 


    rf1 = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)
    rf2 = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)
    

   The problem is NO random_state is set. Here's what's going on:

   1. RandomForest has TWO sources of randomness:
      a) Bootstrap sampling — each tree gets a random subset of rows
      b) Feature sampling — at each split, a random subset of features is considered

   2. When you don't set random_state, Python uses a different random seed
      each time. So rf1 and rf2 build completely different sets of trees.

   3. With only n_estimators=10 (very few trees), the importance estimates
      are UNSTABLE. Each tree sees different data and features, so the
      average importance across just 10 trees varies a lot between runs.

THE FIX (two things):
   1. Set random_state for reproducibility:
      rf = RandomForestClassifier(n_estimators=10, random_state=42)

   2. Use MORE trees to stabilize importance estimates:
      rf = RandomForestClassifier(n_estimators=200, random_state=42)
      



X_tr, X_te, y_tr, y_te = train_test_split(X_demo, y_demo, test_size=0.2, random_state=42)

print("Demonstrating the bug:")
rf1 = RandomForestClassifier(n_estimators=10).fit(X_tr, y_tr)
rf2 = RandomForestClassifier(n_estimators=10).fit(X_tr, y_tr)
print(f"  rf1 importances: {np.round(rf1.feature_importances_, 3)}")
print(f"  rf2 importances: {np.round(rf2.feature_importances_, 3)}")
print(f"  Different? {not np.allclose(rf1.feature_importances_, rf2.feature_importances_)}")

print("\nAfter fix (random_state + more trees):")
rf3 = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_tr, y_tr)
rf4 = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_tr, y_tr)
print(f"  rf3 importances: {np.round(rf3.feature_importances_, 3)}")
print(f"  rf4 importances: {np.round(rf4.feature_importances_, 3)}")
print(f"  Different? {not np.allclose(rf3.feature_importances_, rf4.feature_importances_)}")
