"""
Day 32 | AM Session | Part A: 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree, _tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance


# Step 1 

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
    if row['credit_score'] > 700:
        score += 3
    elif row['credit_score'] > 650:
        score += 1
    if row['debt_to_income'] < 0.35:
        score += 2
    if row['employment_years'] > 5:
        score += 1
    if row['annual_income'] > 80_000:
        score += 1
    
    score += np.random.choice([-1, 0, 0, 0, 1])
    return 1 if score >= 4 else 0

df['approved'] = df.apply(decide, axis=1)

print("Dataset shape:", df.shape)
print("\nClass balance:\n", df['approved'].value_counts())
print("\nFirst 5 rows:\n", df.head())

X = df.drop('approved', axis=1)
y = df['approved']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 2 

dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

print("\n========== Decision Tree Text Rules ==========")
print(export_text(dt, feature_names=list(X.columns), max_depth=3))

# Extract human-readable rules from the tree
def extract_top_rules(tree, feature_names, top_n=3):
    """Walk the tree and return the top-N leaf rules sorted by sample count."""
    t = tree.tree_
    fname = [feature_names[i] if i != _tree.TREE_UNDEFINED else "?" for i in t.feature]
    rules = []

    def _walk(node, path):
        if t.feature[node] == _tree.TREE_UNDEFINED:          # leaf
            counts = t.value[node][0]
            label  = int(np.argmax(counts))
            conf   = counts[label] / counts.sum()
            rules.append((list(path), label, conf, int(counts.sum())))
        else:
            th = t.threshold[node]
            _walk(t.children_left[node],  path + [f"{fname[node]} <= {th:.2f}"])
            _walk(t.children_right[node], path + [f"{fname[node]} > {th:.2f}"])

    _walk(0, [])
    rules.sort(key=lambda r: r[3], reverse=True)
    return rules[:top_n]

top3 = extract_top_rules(dt, list(X.columns))
print("\n========== Top 3 Decision Rules ==========")
for i, (path, lbl, conf, samp) in enumerate(top3, 1):
    action = "APPROVE" if lbl == 1 else "REJECT"
    print(f"  Rule {i}: IF {' AND '.join(path)} -> {action}  "
          f"({conf*100:.0f}% confidence, {samp} samples)")


# Step 3 

param_dist = {
    'n_estimators':    [50, 100, 200, 300],
    'max_depth':       [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'max_features':    ['sqrt', 'log2'],
    'min_samples_leaf': [1, 2, 4],
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=15, cv=5, scoring='roc_auc',
    random_state=42, n_jobs=-1
)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

print("\n========== Best RF Params ==========")
print(rf_search.best_params_)


# Step 4 

def report(model, Xt, yt, name):
    pred  = model.predict(Xt)
    prob  = model.predict_proba(Xt)[:, 1]
    acc   = accuracy_score(yt, pred)
    f1    = f1_score(yt, pred)
    auc   = roc_auc_score(yt, prob)
    print(f"\n--- {name} ---")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    return acc, f1, auc

dt_m = report(dt, X_test, y_test, "Decision Tree (depth=4)")
rf_m = report(best_rf, X_test, y_test, "Random Forest (tuned)")

print("\nInterpretability:")
print("  Decision Tree : HIGH — every split can be shown to regulators.")
print("  Random Forest : LOW  — hundreds of trees, hard to explain one decision.")


# Step 5 

perm = permutation_importance(best_rf, X_test, y_test,
                              n_repeats=10, random_state=42)

imp_df = pd.DataFrame({
    'Feature':               X.columns,
    'Default (MDI)':         best_rf.feature_importances_,
    'Permutation (mean)':    perm.importances_mean,
}).sort_values('Permutation (mean)', ascending=False)

print("\n========== Feature Importance Comparison ==========")
print(imp_df.to_string(index=False))


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
imp_df_sorted = imp_df.sort_values('Default (MDI)', ascending=True)
axes[0].barh(imp_df_sorted['Feature'], imp_df_sorted['Default (MDI)'], color='steelblue')
axes[0].set_title('Default (MDI) Importance')
imp_df_sorted2 = imp_df.sort_values('Permutation (mean)', ascending=True)
axes[1].barh(imp_df_sorted2['Feature'], imp_df_sorted2['Permutation (mean)'], color='coral')
axes[1].set_title('Permutation Importance')
plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=100)
plt.show()
print("Saved: feature_importance_comparison.png")


# Step 6

print("\n========== Recommendation ==========")
print("""
The bank should deploy the Random Forest as its primary scoring engine because it
delivers superior Accuracy, F1-Score, and ROC-AUC, which means fewer bad loans slip
through. However, since regulators require clear explanations for every decision,
we should keep the Decision Tree (max_depth=4) as an interpretability layer: its
top rules (e.g. "IF credit_score > 700 AND debt_to_income < 0.35 → APPROVE") can
serve as human-readable justifications. In practice, the RF scores the applicant
and the DT rules are shown as the 'reason code' — giving us both accuracy AND
transparency.
""")
