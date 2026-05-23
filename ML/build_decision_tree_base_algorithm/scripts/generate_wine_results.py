import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import os

os.makedirs('data', exist_ok=True)

wine = datasets.load_wine()
X = wine.data
y = wine.target

models = {
    'LogisticRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=5000, random_state=317))
    ]),
    'SVC_RBF': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', random_state=42))
    ]),
    'DecisionTree': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', DecisionTreeClassifier(random_state=42))
    ])
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

cv_res = {}
res = {}
reports = {}
conf_mats = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_res[name] = scores.tolist()

    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    res[name] = {'train': float(train_score), 'test': float(test_score)}

    y_pred = model.predict(X_test)
    reports[name] = classification_report(y_test, y_pred, output_dict=True)
    conf_mats[name] = confusion_matrix(y_test, y_pred).tolist()

    print(f"{name}: CV mean={scores.mean():.4f}, CV std={scores.std():.4f}")
    print(f"  Train Acc: {train_score:.4f} | Test Acc: {test_score:.4f}\n")

# Save numeric results
results = {
    'cv_res': cv_res,
    'res': res,
    'reports': reports,
    'confusion_matrices': conf_mats
}
with open('data/wine_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Plot CV means and test scores
model_names = list(res.keys())
cv_means = [np.mean(cv_res[m]) for m in model_names]
cv_stds = [np.std(cv_res[m]) for m in model_names]
test_scores = [res[m]['test'] for m in model_names]

x_pos = np.arange(len(model_names))
fig, ax = plt.subplots(figsize=(8,5))
bars_cv = ax.bar(x_pos - 0.2, cv_means, 0.4, label='CV Mean Accuracy', color='#2196F3', yerr=cv_stds, capsize=5)
bars_test = ax.bar(x_pos + 0.2, test_scores, 0.4, label='Test Accuracy', color='#FF5722')
ax.set_ylabel('Accuracy')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation=20, ha='right')
ax.legend()
ax.set_ylim(0.6, 1.02)
for bar in bars_cv:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005, f'{height:.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars_test:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005, f'{height:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('data/model_comparison.png', bbox_inches='tight')
plt.close()

# Plot confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(12,4))
for ax, name in zip(axes, model_names):
    cm = np.array(conf_mats[name])
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha='center', va='center', color='black')
fig.colorbar(im, ax=axes.ravel().tolist())
plt.tight_layout()
plt.savefig('data/confusion_matrices.png', bbox_inches='tight')
plt.close()

print('Results saved to data/wine_results.json and images in data/')
