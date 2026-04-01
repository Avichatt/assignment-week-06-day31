 Part A: 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text, _tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score, confusion_matrix)


# Step 1 

np.random.seed(42)
n = 3000

df = pd.DataFrame({
    'claim_amount':         np.random.randint(500, 50000, n),
    'policy_age_months':    np.random.randint(1, 120, n),
    'num_previous_claims':  np.random.randint(0, 10, n),
    'age_of_claimant':      np.random.randint(18, 80, n),
    'police_report_filed':  np.random.choice([0, 1], n, p=[0.6, 0.4]),
    'witness_present':      np.random.choice([0, 1], n, p=[0.5, 0.5]),
    'claim_day_of_week':    np.random.randint(0, 7, n),   
    'time_to_report_days':  np.random.randint(0, 60, n),
})


def is_fraud(row):
    score = 0
    if row['claim_amount'] > 30000:          score += 2
    if row['num_previous_claims'] > 5:       score += 2
    if row['police_report_filed'] == 0:      score += 1
    if row['time_to_report_days'] > 30:      score += 2
    if row['witness_present'] == 0:          score += 1
    if row['policy_age_months'] < 6:         score += 1
    
    score += np.random.choice([-1, 0, 0, 0, 0, 1])
    return 1 if score >= 5 else 0

df['is_fraud'] = df.apply(is_fraud, axis=1)
print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
print(df.head())

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 2

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)


print(export_text(dt, feature_names=list(X.columns), max_depth=3))

def extract_fraud_rules(tree, feature_names, target_label=1, top_n=3):
   
    t = tree.tree_
    fname = [feature_names[i] if i != _tree.TREE_UNDEFINED else "?" for i in t.feature]
    rules = []

    def _walk(node, path):
        if t.feature[node] == _tree.TREE_UNDEFINED:
            counts = t.value[node][0]
            label  = int(np.argmax(counts))
            if label == target_label:
                conf = counts[label] / counts.sum()
                rules.append((list(path), conf, int(counts.sum())))
        else:
            th = t.threshold[node]
            _walk(t.children_left[node],  path + [f"{fname[node]} <= {th:.2f}"])
            _walk(t.children_right[node], path + [f"{fname[node]} > {th:.2f}"])

    _walk(0, [])
    rules.sort(key=lambda r: r[1] * r[2], reverse=True)
    return rules[:top_n]

fraud_rules = extract_fraud_rules(dt, list(X.columns))
print("\n========== Top 3 Fraud Indicator Rules ==========")
for i, (path, conf, samp) in enumerate(fraud_rules, 1):
    print(f"  Rule {i}: IF {' AND '.join(path)}")
    print(f"           → FRAUD ({conf*100:.0f}% confidence, {samp} samples)\n")


# Step 3 

param_dist = {
    'n_estimators':     [100, 200, 300],
    'max_depth':        [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'max_features':     ['sqrt', 'log2'],
    'min_samples_leaf': [1, 2, 4],
    'class_weight':     ['balanced', None],
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20, cv=5,
    scoring='recall', 
    random_state=42, n_jobs=-1
)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
print(f"\nBest RF Params: {rf_search.best_params_}")


# Step 4 

def full_report(model, Xt, yt):
    pred = model.predict(Xt)
    prob = model.predict_proba(Xt)[:, 1]
    return {
        'Accuracy':  accuracy_score(yt, pred),
        'Precision': precision_score(yt, pred),
        'Recall':    recall_score(yt, pred),
        'F1-Score':  f1_score(yt, pred),
        'ROC-AUC':   roc_auc_score(yt, prob),
    }

dt_metrics = full_report(dt, X_test, y_test)
rf_metrics = full_report(best_rf, X_test, y_test)

metrics_df = pd.DataFrame({
    'Metric': list(dt_metrics.keys()),
    'Decision Tree (depth=5)': [f"{v:.4f}" for v in dt_metrics.values()],
    'Random Forest (tuned)':   [f"{v:.4f}" for v in rf_metrics.values()],
})

print(metrics_df.to_string(index=False))


# Step 5

print("\n========== Cost-Sensitive Evaluation ==========")
print("Cost assumption: Missing a fraud (FN) costs 10x a false alarm (FP)")

FP_COST = 1   
FN_COST = 10 

def cost_analysis(model, Xt, yt, name):
    pred = model.predict(Xt)
    tn, fp, fn, tp = confusion_matrix(yt, pred).ravel()
    total_cost = fp * FP_COST + fn * FN_COST
    print(f"\n  {name}:")
    print(f"    TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"    Cost = ({fp} × ${FP_COST}) + ({fn} × ${FN_COST}) = ${total_cost}")
    return total_cost

dt_cost = cost_analysis(dt, X_test, y_test, "Decision Tree")
rf_cost = cost_analysis(best_rf, X_test, y_test, "Random Forest")

winner = "Random Forest" if rf_cost < dt_cost else "Decision Tree"
print(f"\n  Lower-cost model: {winner}")


# Step 6 



Paragraph 1 — Model Selection:
The Random Forest model should be deployed as the primary fraud scoring engine.
It achieved a higher Recall ({rf_metrics['Recall']:.2f}) compared to the Decision Tree
({dt_metrics['Recall']:.2f}), which is critical because every missed fraud case (FN)
costs the company 10x more than a false alarm (FP). The RF also delivered a stronger
ROC-AUC ({rf_metrics['ROC-AUC']:.2f}), meaning it better separates fraud from
legitimate claims across all threshold settings. The cost analysis confirms the RF
saves the company ${dt_cost - rf_cost} per test batch compared to the DT.

Paragraph 2 — Addressing Regulatory Requirements:
Since regulators require model explanations, we recommend a two-model approach. The
Random Forest scores every incoming claim automatically. For claims flagged as fraud,
the Decision Tree's extracted rules (e.g., "IF time_to_report > 30 days AND no police
report AND claim > $30k = FRAUD") are presented as the human-readable explanation to
the regulator. This gives us the best of both worlds: the RF's accuracy for catching
fraud and the DT's transparency for compliance. The DT rules act as a simplified
'proxy explanation' of the RF's decision.

