 #Part B: 



# Gradient Boosting vs Random Forest (Conceptual)



Bagging (used by Random Forest) trains many independent trees in PARALLEL on
random subsets of the data, then averages their predictions — each tree has no
knowledge of the others. Boosting (used by Gradient Boosting) trains trees
SEQUENTIALLY: the first tree makes predictions, then the second tree focuses
specifically on the MISTAKES the first tree made, then the third tree fixes the
second tree's mistakes, and so on. Each new tree is built to correct the errors
of all the previous trees combined. Because of this, bagging reduces VARIANCE
(by averaging many noisy models), while boosting reduces BIAS (by iteratively
fixing errors). Boosting can achieve higher accuracy but is more prone to
overfitting if not tuned carefully, and it's slower to train since trees must
be built one after another instead of in parallel.



Best resource I found for understanding boosting:

Title : "Gradient Boosting from Scratch"
Link  : https://www.youtube.com/watch?v=3CC4N4z3GJc  (StatQuest with Josh Starmer)

Why this is good:
  - Josh explains boosting step-by-step with visual examples
  - He shows exactly how each tree corrects the previous tree's errors
  - The video is beginner-friendly and only ~15 minutes long
  - He also has a follow-up video on XGBoost which we'll need later

Alternative text resource:
  - "A Gentle Introduction to Gradient Boosting" by Machine Learning Mastery
    https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/



┌─────────────────┬──────────────────────┬──────────────────────┐
│  Aspect         │  Bagging (RF)        │  Boosting (GBM)      │
├─────────────────┼──────────────────────┼──────────────────────┤
│  Training       │  Parallel            │  Sequential          │
│  Goal           │  Reduce variance     │  Reduce bias         │
│  Trees          │  Independent         │  Each corrects prev  │
│  Overfitting    │  Resistant           │  Can overfit easily  │
│  Speed          │  Faster (parallel)   │  Slower (sequential) │
│  Typical use    │  Random Forest       │  XGBoost, LightGBM   │
└─────────────────┴──────────────────────┴──────────────────────┘



