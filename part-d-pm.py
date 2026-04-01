 Part D: 

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: 


AI Response:
───────────────────────────────────────────────────────────
Imagine you're a teacher grading student essays. You have 100 essays, but
instead of reading all 100, each grader only reads a RANDOM 63 essays
(about 2/3). The remaining ~37 essays they NEVER saw.


OOB error works:
- Each tree in a Random Forest is trained on ~63% of the data (bootstrap sample).
- The other ~37% it never saw — these are "Out-of-Bag" samples.
- We test each tree ONLY on its OOB samples (data it never trained on).
- The average error across all these OOB predictions = OOB error.

It's like getting a free test score without needing a separate test set!
───────────────────────────────────────────────────────────



# Step 2:

Aspects:
  1. The ~63% figure is correct — each bootstrap sample contains about
     63.2% of the original data (1 - 1/e ≈ 0.632).
  2. The key idea is captured: test on data the model never saw.
  3. The "free test score" idea is correct — OOB doesn't need a separate
     hold-out set.
  4. The analogy (graders + essays) is easy to understand for non-tech people.

SLIGHT OVERSIMPLIFICATION:
  1. The analogy doesn't mention that OOB predictions are AGGREGATED across
     multiple trees. For each data point, we collect predictions from all
     trees that didn't see it, then take a majority vote. This is important
     because one tree's OOB prediction would be noisy.
  2. The analogy implies graders are independent, but doesn't explain that
     each tree also uses a random subset of FEATURES (not just data).


# Step 3:

Prompt used: 

AI Response:

OOB error and test error can differ significantly in these cases:

1. Small dataset — With few samples, the OOB subset for each tree is tiny,
   making OOB estimates noisy and unreliable.

2. Data distribution shift — If your test set comes from a different
   distribution (e.g., different time period, different demographics),
   OOB error won't catch this because OOB samples come from the SAME
   distribution as the training data.

3. Class imbalance — With severe imbalance, OOB estimates per tree may
   not have enough minority class samples to evaluate properly.

4. Very few trees — With few trees (e.g., 10), each sample may only be
   OOB for 3-4 trees, making the majority vote unreliable.


My explanation:
  Points 1, 2, and 4 are correct and well-explained.
  Point 3 is valid — stratification isn't guaranteed in bootstrap.
  Missing point: OOB error is computed on ~37% of data per tree,
     while test error uses a fixed hold-out set. The OOB estimate uses
     ALL training data (each point tested by trees that didn't see it),
     so it's actually closer to leave-one-out cross-validation.
  Also missing: temporal data. If data has time ordering (e.g., stock
     prices), bootstrap sampling breaks the time sequence, so OOB error
     would be overly optimistic compared to a proper time-split test.


# Step 4


np.random.seed(42)
n = 1500
X = np.random.randn(n, 5)
y = (X[:, 0] + X[:, 1] * 2 + np.random.randn(n) * 0.3 > 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RF with oob_score=True
rf = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
rf.fit(X_train, y_train)

oob_acc  = rf.oob_score_
test_acc = accuracy_score(y_test, rf.predict(X_test))

print(f"  OOB Accuracy  : {oob_acc:.4f}")
print(f"  Test Accuracy : {test_acc:.4f}")
print(f"  Difference    : {abs(oob_acc - test_acc):.4f}")
print()
print("  As expected, OOB and test accuracy are very close when the data")
print("  comes from the same distribution. OOB is a reliable free estimate!")
