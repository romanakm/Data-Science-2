import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data_test = pd.read_csv("/Users/andreapolakovicova/DS_2_HW_1/data-science-2-2024-hw1/2024_DS2_HW1_data_test.csv")
data_train = pd.read_csv("/Users/andreapolakovicova/DS_2_HW_1/data-science-2-2024-hw1/2024_DS2_HW1_data_train.csv")

print(data_train.describe())
print(data_train.head())
print(data_train.shape)

print(data_train.no_of_previous_cancellations.value_counts(dropna=False))
print(data_train.no_of_previous_bookings_not_canceled.value_counts(dropna=False))
print(data_train.booking_status.value_counts(dropna=False))

# Drop rows with missing target values
data_train.dropna(subset=["booking_status"], inplace=True)

TARGET = data_train["booking_status"]
X = data_train.drop(columns=["Booking_ID","booking_status"])

cols_pred = list(data_train.columns[1:-1])

cols_pred_num = [col for col in cols_pred if data_train[col].dtype != 'O']
# define list of categorical predictors
cols_pred_cat = [col for col in cols_pred if data_train[col].dtype == 'O']

print('Numerical predictors:')
print('---------------------')
print(data_train[cols_pred_num].dtypes)
print()
print('Categorical predictors:')
print('-----------------------')
print(data_train[cols_pred_cat].dtypes)
print(data_train.type_of_meal_plan.value_counts())
print(data_train.room_type_reserved.value_counts())
print(data_train.market_segment_type.value_counts())

# DATA SPLIT

data_train_set, data_test_set = train_test_split(data_train, test_size=0.2, random_state=27, stratify=TARGET) # stratification = booking cancellation
data_train_set, data_valid_set = train_test_split(data_train_set, test_size=0.25, random_state=27, stratify=data_train_set['booking_status'])  # 0.25 x 0.8 = 0.2

print(data_train_set)
print(data_test_set)
print(data_valid_set)

# Convert categorical variables to one-hot encoding
X = pd.get_dummies(X, columns=cols_pred_cat)

# Data split
X_train, X_test, y_train, y_test = train_test_split(X, TARGET, test_size=0.2, random_state=27, stratify=TARGET)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=27, stratify=y_train)  # 0.25 x 0.8 = 0.2

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

# XGBoost parameters
params = {
    'max_depth': 4,
    'objective': 'binary:logistic',
    'eval_metric': ['auc'],
}

# Train the model and plot AUC
evals_result = {}
bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dtest, 'test'), (dvalid, 'valid')],
    evals_result=evals_result,
    early_stopping_rounds=10,
    verbose_eval=False
)

# Make predictions
y_pred_prob = bst.predict(dtest)
y_pred = np.round(y_pred_prob)

# Convert predicted probabilities to binary predictions
y_pred = [1 if pred > 0.5 else 0 for pred in y_pred]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print classification report
print(classification_report(y_test, y_pred))

# Print confusion matrix
print(confusion_matrix(y_test, y_pred))

# Plot AUC
metric = 'auc'

fig, ax = plt.subplots(figsize=(6,5))

total_iteration_count = len(evals_result['train'][metric])

for sample, vals in evals_result.items():
    ax.plot(
        range(1, total_iteration_count + 1),
        vals[metric],
        label=sample
    )

best_score = bst.best_score
best_iteration = bst.best_iteration + 1

ax.plot([1, total_iteration_count], [best_score, best_score], color='black', ls='--', lw=1)
ax.scatter([best_iteration], [best_score], color='black')
ax.annotate(
    '{:d}; {:0.3f}'.format(best_iteration, best_score),
    xy=(best_iteration, best_score),
    xytext=(best_iteration, best_score+0.005),
)
ax.set_xlabel('iteration', color='gray')
ax.set_ylabel(metric, color='gray')
ax.legend(loc='best')
ax.set_title(f'Model training - {metric} curves')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_color('gray')
ax.tick_params(axis='y', colors='gray')
ax.tick_params(axis='x', colors='gray')

# Predictor importance based on 'total_gain'
importance_type = 'total_gain'
predictor_strength = sorted([(k, v) for k, v in bst.get_score(importance_type=importance_type).items()], key=lambda x: x[1], reverse=True)
predictor_strength = pd.DataFrame(predictor_strength, columns=['predictor', 'strength'])

# Plot predictor importance
fig = plt.figure(figsize=(6,5))
ax = plt.subplot(1,1,1)

n_strongest = 20
ax.barh(range(n_strongest, 0, -1), predictor_strength['strength'].iloc[0:20])
ax.set_yticks(range(n_strongest, 0, -1))
ax.set_yticklabels(predictor_strength['predictor'].iloc[0:20])
ax.set_xlabel(importance_type)
ax.set_title('Predictor importance based on total_gain')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_color('gray')
ax.tick_params(axis='y', colors='gray')
ax.tick_params(axis='x', colors='gray')

plt.show()
