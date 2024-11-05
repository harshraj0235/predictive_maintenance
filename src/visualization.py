import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

def plot_feature_importance(model, X_train, y_train):
    result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)

    sorted_idx = np.argsort(result.importances_mean)[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(X_train.columns[sorted_idx], result.importances_mean[sorted_idx], align='center', color='lightblue')
    plt.xlabel("Mean Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance")
    plt.yticks(sorted_idx, X_train.columns[sorted_idx])
    plt.xlim([0, 0.15])
    plt.show()