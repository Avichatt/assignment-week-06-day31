#Part C:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Q1: 

BIAS = how far off our model's average prediction is from the correct value.
VARIANCE = how much our model's predictions change when we train on different data.

Decision Tree (high variance, low bias):
  - A deep Decision Tree memorizes the training data perfectly (low bias).
  - But if we change the training data slightly, the tree structure changes
    completely — it's very sensitive. That's high variance.
  - This is why a single DT tends to OVERFIT.

Random Forest (lower variance, similar bias):
  - RF trains many different trees on random subsets of data (bagging).
  - Each individual tree still has high variance, BUT when we average all
    their predictions, the variance cancels out!
  - The bias stays roughly the same, but variance drops significantly.

How bagging reduces variance (conceptual diagram):

    Single Decision Tree:
    ┌─────────────────────────────────────┐
    │  Training Set → One Tree → Prediction│
    │  HIGH variance (unstable)            │
    └─────────────────────────────────────┘

    Random Forest (Bagging):
    ┌─────────────────────────────────────────────┐
    │  Bootstrap Sample 1 → Tree 1 → Pred 1       │
    │  Bootstrap Sample 2 → Tree 2 → Pred 2       │
    │  Bootstrap Sample 3 → Tree 3 → Pred 3       │
    │                                             │
    │  Bootstrap Sample N → Tree N → Pred N       │
    │                                             │
    │  Final = AVERAGE(Pred 1..N)                 │
    │  LOW variance (averaging cancels noise)     │
    └─────────────────────────────────────────────┘

    Variance of average = (original variance) / N
    → More trees = lower variance!



# Q2:

np.random.seed(42)
n = 1000
X_data = np.random.randn(n, 5)
y_data = (X_data[:, 0] + X_data[:, 1] * 2 + np.random.randn(n) * 0.5 > 1).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)


def plot_overfitting_curve(X, y, max_depths):
    
    Trains Decision Trees at each max_depth and plots train vs test accuracy.
    Identifies the optimal depth where test accuracy is highest.

    Parameters:
        X : features (array-like)
        y : target (array-like)
        max_depths : list of int, the depths to try
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    train_accs = []
    test_accs  = []

    for depth in max_depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_tr, y_tr)
        train_accs.append(accuracy_score(y_tr, dt.predict(X_tr)))
        test_accs.append(accuracy_score(y_te, dt.predict(X_te)))

    
    best_idx   = np.argmax(test_accs)
    best_depth = max_depths[best_idx]
    best_acc   = test_accs[best_idx]


    plt.figure(figsize=(8, 5))
    plt.plot(max_depths, train_accs, 'o-', label='Train Accuracy', color='steelblue')
    plt.plot(max_depths, test_accs,  's-', label='Test Accuracy',  color='coral')
    plt.axvline(best_depth, color='green', linestyle='--', alpha=0.7,
                label=f'Optimal depth = {best_depth}')
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')
    plt.title('Overfitting Curve: Train vs Test Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('overfitting_curve.png', dpi=100)
    plt.show()

    print(f"\nOptimal max_depth = {best_depth} (Test Accuracy = {best_acc:.4f})")
    
    return best_depth



depths = list(range(1, 21))
optimal = plot_overfitting_curve(X_data, y_data, depths)



# Q3: 

Given code:
    rf = RandomForestClassifier(n_estimators=500, max_depth=3, random_state=42)
    rf.fit(X_train, y_train)
    print(f'Train: {rf.score(X_train, y_train):.2f}')  
    print(f'Test: {rf.score(X_test, y_test):.2f}')      

Question: Is this a problem? Why or why not?

Answer: NO, this is NOT a problem. Here's why:

1. max_depth=3 makes each tree very SHALLOW.
   - Shallow trees can't memorize the training data (low variance).
   - So the model is NOT overfitting.

2. When train ≈ test accuracy, it means the model generalizes well.
   - Overfitting would show: train >> test (e.g., train=0.99, test=0.80).
   - Here both are 0.95, so the model learned real patterns, not noise.

3. n_estimators=500 is fine — more trees don't cause overfitting in RF.
   - RF is unique: adding more trees does NOT overfit.
   - It just reduces variance (or plateaus).

4. The only concern might be: is 0.95 the best we can do?
   - We could try deeper trees or tune other hyperparameters.
   - But the model is stable and generalizing well — that's good!

Summary: Identical train and test accuracy with a shallow RF means the model
is well-regularized and generalizing properly. This is actually IDEAL behavior.

