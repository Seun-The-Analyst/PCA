# PCA

# Breast Cancer Dataset PCA & Logistic Regression Project

This project demonstrates how to use Principal Component Analysis (PCA) and Logistic Regression for classification on the breast cancer dataset from scikit-learn. The workflow includes data loading, dimensionality reduction, visualization, and predictive modeling.

## Project Steps

1. **Install Required Libraries**
   - scikit-learn
   - pandas
   - matplotlib
   - numpy

   ```
   %pip install scikit-learn pandas matplotlib numpy
   ```

2. **Load the Dataset**
   - The breast cancer dataset is loaded using `sklearn.datasets.load_breast_cancer`.
   - Data is converted to a pandas DataFrame for easier manipulation.

3. **Data Exploration**
   - Display the first few rows and all columns of the dataset.
   - Check the structure and contents of the data.

4. **Principal Component Analysis (PCA)**
   - Standardize the features using `StandardScaler`.
   - Apply PCA to reduce the dataset to two principal components.
   - Visualize the data in the new PCA space, colored by class.

5. **Logistic Regression for Prediction**
   - Split the PCA-transformed data into training and testing sets.
   - Train a logistic regression model on the training data.
   - Evaluate the model using accuracy and a classification report.
   - Visualize the decision boundary of the logistic regression in the PCA space.

## Example Code Snippets

**Load and Prepare Data:**
```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
```

**PCA and Visualization:**
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X_scaled = StandardScaler().fit_transform(df.drop('target', axis=1))
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['target'] = df['target']

plt.figure(figsize=(8,6))
for target, color in zip([0, 1], ['red', 'blue']):
    plt.scatter(
        pca_df.loc[pca_df['target'] == target, 'PC1'],
        pca_df.loc[pca_df['target'] == target, 'PC2'],
        c=color,
        label=f"Class {target}",
        alpha=0.5
    )
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Breast Cancer Dataset')
plt.legend()
plt.show()
```

**Logistic Regression and Decision Boundary:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

X = pca_df[['PC1', 'PC2']]
y = pca_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(8,6))
for target, color in zip([0, 1], ['red', 'blue']):
    plt.scatter(
        X_test.loc[y_test == target, 'PC1'],
        X_test.loc[y_test == target, 'PC2'],
        c=color,
        label=f"Class {target}",
        alpha=0.5
    )
x_min, x_max = X_test['PC1'].min() - 1, X_test['PC1'].max() + 1
y_min, y_max = X_test['PC2'].min() - 1, X_test['PC2'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Logistic Regression Decision Boundary (PCA Features)')
plt.legend()
plt.show()
```

## Notes

- The project uses only the first two principal components for visualization and classification.
- The workflow can be extended to use more components or other classifiers.
- All code is written in Python and designed for use in Jupyter Notebooks.

---
**Author:**  
Nexford University - BAN 6420  
Intro R & Python  
