from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from loading_and_preprocessing import x_train, y_train, x_val, y_val
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from loading_and_preprocessing import gen

from utils import iter_grid, make_mlp, manual_cv_accuracy, pretty_print_buffer, stratified_kfold_indices,stratified_subsample

rf_grid = {
    "n_estimators": [200, 300, 400, 500, 800],
    "max_depth": [None, 8, 16, 32],
    "max_features": [2, 3, None],
    "min_samples_leaf": [1, 3, 10],
}
best_rf, best_rf_params,best_val_acc = None, None, -np.inf


for params in iter_grid(rf_grid):
    model = RandomForestClassifier(random_state=42,n_jobs=-1, **params)
    model.fit(x_train, y_train)
    acc = accuracy_score(y_val, model.predict(x_val))
    if acc > best_val_acc:
        best_val_acc = acc
        best_rf_params = params
        best_rf = model
        
print("Best RF params:", best_rf_params, "Val acc:", best_val_acc)

rf_cv = manual_cv_accuracy(RandomForestClassifier(random_state=42, n_jobs=-1, **best_rf_params),x_train, y_train, k=10) # type: ignore
print(rf_cv)


pretty_print_buffer()

hidden_grid = [4, 8, 16, 32, 64, 128]

best_mlp, best_h, best_val_acc_mlp = None, None, -np.inf
for h in hidden_grid:
    pipe = make_mlp(h)
    pipe.fit(x_train, y_train)
    acc = accuracy_score(y_val, pipe.predict(x_val))
    if acc > best_val_acc_mlp:
        best_val_acc_mlp = acc
        best_h = h
        best_mlp = pipe
print("Best MLP hidden:", best_h, "Val acc:", best_val_acc_mlp)


use_rf = (best_val_acc >= best_val_acc_mlp)
print("Chosen model:", "RF" if use_rf else "MLP")

Ns = np.arange(10, 1001, 20)  # 10, 30, 50, ... 990, 1010; adjust to exactly 10,20,...,1000 if needed
Ns = np.arange(10, 1001, 10)  # exactly 10,20,...,1000

R = 5
train_err_mean, test_err_mean = [], []

for N in Ns:
    tr_errs, te_errs = [], []
    for r in range(R):
        x_sub, y_sub = stratified_subsample(x_train, y_train, int(N), gen)
        if use_rf:
            model = RandomForestClassifier(random_state=42 + r, n_jobs=-1, **best_rf_params) # type: ignore
            model.fit(x_sub, y_sub)
            tr_acc = accuracy_score(y_sub, model.predict(x_sub))
            te_acc = accuracy_score(y_val, model.predict(x_val))
        else:
            pipe = make_mlp(best_h) # type: ignore
            pipe.fit(x_sub, y_sub)
            tr_acc = accuracy_score(y_sub, pipe.predict(x_sub))
            te_acc = accuracy_score(y_val, pipe.predict(x_val))

        tr_errs.append(1.0 - tr_acc)
        te_errs.append(1.0 - te_acc)

    train_err_mean.append(float(np.mean(tr_errs)))
    test_err_mean.append(float(np.mean(te_errs)))

learning_curve = pd.DataFrame({"N": Ns,
                               "train_error": train_err_mean,
                               "val_error": test_err_mean})
print(learning_curve.head())

pretty_print_buffer()

import matplotlib.pyplot as plt

plt.figure()
plt.plot(learning_curve["N"], learning_curve["train_error"], label="Train error")
plt.plot(learning_curve["N"], learning_curve["val_error"], label="Validation (test) error")
plt.xlabel("Training size N")
plt.ylabel("Error (1 - accuracy)")
plt.title(("RF" if use_rf else "MLP") + " learning curve")
plt.legend()
plt.tight_layout()
plt.show()