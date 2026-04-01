
 #Part D:

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Step 1: 

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#1a1a2e')


ax.text(8, 9.5, 'Decision Tree  vs  Random Forest  vs  Logistic Regression',
        fontsize=18, fontweight='bold', ha='center', color='white',
        fontfamily='sans-serif')
ax.text(8, 9.0, 'A Simple Guide for Choosing the Right Model',
        fontsize=11, ha='center', color='#aaaaaa', style='italic')


col_x     = [2.5, 8, 13.5]
col_w     = 4.2
col_h     = 7.0
colors    = ['#16213e', '#0f3460', '#1a1a40']
borders   = ['#e94560', '#0096c7', '#f77f00']
titles    = ['Decision Tree', 'Random Forest', 'Logistic Regression']

[

    "Splits data using yes/no questions (like a flowchart)",
    "Many decision trees vote together (teamwork!)",
    "Draws a straight line to separate classes (simple math)",
]

[
    "• Need to explain decisions clearly, Small-medium dat, Quick prototyping",
    "• Need HIGH accuracy, Large datasets, Don't need to explain, every decision",
    "• Simple yes/no problems, Need probability scores, Linear relationships, Very fast predictions",
]

pros = [
    " Easy to understand, No scaling needed, Handles non-linear",
    " Very accurate,  Handles missing data,  Resistant to overfitting",
    " Super fast,  Easy to interpret, Works well with few features",
]

cons = [
    "Overfits easily,  Unstable data changes =  different tree)",
    "Slow to train,  Hard to explain,  Needs more memory",
    "Can't handle,  complex patterns, Needs feature scaling",
]


interp_scores = [9, 4, 8]
accuracy_scores = [5, 9, 6]
speed_scores = [7, 4, 9]

for i in range(3):
    cx = col_x[i]
  
    card = mpatches.FancyBboxPatch(
        (cx - col_w/2, 0.5), col_w, col_h,
        boxstyle="round,pad=0.15", facecolor=colors[i],
        edgecolor=borders[i], linewidth=2, alpha=0.9
    )
    ax.add_patch(card)

  
    y_pos = 7.0
    ax.text(cx, y_pos, titles[i], fontsize=14, fontweight='bold',
            ha='center', color=borders[i])

 
    y_pos -= 0.7
    ax.text(cx, y_pos, descriptions[i], fontsize=8, ha='center',
            color='#cccccc', linespacing=1.4, va='top')

  
    y_pos -= 1.5
    ax.text(cx, y_pos, "When to use:", fontsize=9, fontweight='bold',
            ha='center', color='white')
    y_pos -= 0.2
    ax.text(cx - 1.5, y_pos, when_to_use[i], fontsize=7.5,
            color='#bbbbbb', va='top', linespacing=1.3)

  
    y_pos -= 1.6
    ax.text(cx - 1.5, y_pos, pros[i], fontsize=7.5,
            color='#88d888', va='top', linespacing=1.3)

    
    y_pos -= 1.4
    ax.text(cx - 1.5, y_pos, cons[i], fontsize=7.5,
            color='#e88888', va='top', linespacing=1.3)

   
    bar_y     = 0.8
    bar_h     = 0.12
    bar_gap   = 0.18
    bar_max_w = 3.0
    bar_labels = ['Interpret.', 'Accuracy', 'Speed']
    bar_vals   = [interp_scores[i], accuracy_scores[i], speed_scores[i]]
    bar_cols   = ['#00bbf9', '#f15bb5', '#fee440']

    for j in range(3):
        by = bar_y + j * bar_gap
        w  = (bar_vals[j] / 10) * bar_max_w
        # Background bar
        ax.barh(by, bar_max_w, height=0.1, left=cx - bar_max_w/2,
                color='#333333', align='center')
        # Value bar
        ax.barh(by, w, height=0.1, left=cx - bar_max_w/2,
                color=bar_cols[j], align='center', alpha=0.8)
        # Label
        ax.text(cx - bar_max_w/2 - 0.1, by, bar_labels[j],
                fontsize=6, ha='right', va='center', color='#aaaaaa')

plt.tight_layout()
plt.savefig('model_comparison_infographic.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.show()
print("Saved: model_comparison_infographic.png")

\
# Step 2: 


  YES — the key differences are correctly represented:
  - DT is the most interpretable but prone to overfitting.
  - RF is the most accurate but harder to explain.
  - Logistic Regression is the fastest and simplest but limited to linear patterns.


  SLIGHTLY — here's what's missing or simplified:
  1. Random Forest CAN provide some interpretability via feature importance,
     it's not completely a black box.
  2. Logistic Regression can handle non-linear patterns if we manually add
     polynomial features (but the model itself is still linear).
  3. Decision Trees don't always overfit — with proper pruning (max_depth,
     min_samples_leaf), they can be quite stable.
  4. The accuracy scores are rough estimates — in practice, it depends heavily
     on the specific dataset.

Improvements I made:
  - Added the mini bar charts for Interpretability/Accuracy/Speed to make
    the comparison more data-driven rather than just text.
  - Used color coding (green for pros, red for cons) for quick scanning.
  - Kept the language non-technical ("yes/no questions", "teamwork", "straight line")
    so a business stakeholder could understand it.

