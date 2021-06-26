import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

from src.svm.helpers import data_helpers
from src.svm.helpers.test_hyperparams import test_hyperparams_linear, test_hyperparams_RBF, test_hyperparams_poly

DEBUG = False
TRAINING_SIZE = 800
TEST_SIZE=15000

# Load data from disk
print('[Main] Info: Load data from disk')
df = pd.read_csv('../../../resources/bank-full.csv')

# Cleanup data
print('[Main] Info: Cleaning NaN rows')
if DEBUG:
    print('[Main] Debug: Null values per column:')
    print(df.isnull().sum())
    print(f'[Main] Debug: Shape before dropping columns with na values: {df.shape}')

data = df.dropna(axis=1)
if DEBUG:
    print(f'[Main] Debug: Shape after dropping columns with na values: {df.shape}')

print('[Main] Info: Convert class attributes from strings to integers')
if DEBUG:
    print(f'[Main] Debug: Before - Dataframe columns and dtypes: \n{df.dtypes}')

X_int_str_class_dict = data_helpers.convert_class_to_int(df, 'Target')
if DEBUG:
    print(f'[Main] Debug: After - Dataframe columns and dtypes: \n{df.dtypes}')

# split X/y frames
print('[Main] Info: Split data to train / test sets')
X, y = df.drop('Target', axis=1).values, df['Target'].values

# Normalize X to [0,1]
X = data_helpers.normalize(X, DEBUG=False)

# split Train / test sets
X_train, X_test = train_test_split(X, random_state=99, train_size=TRAINING_SIZE)
y_train, y_test = train_test_split(y, random_state=99, train_size=TRAINING_SIZE)

if len(y_test) > TEST_SIZE:
    X_test, y_test = X_test[:TEST_SIZE], y_test[:TEST_SIZE]

if DEBUG:
    print("[Main] Debug: Training dataset shape (X, y): ", X_train.shape, y_train.shape)
    print("[Main] Debug: Testing dataset shape: (X, y)", X_test.shape, y_test.shape)


filename = "../../../output/bank_hyperparams_rbf.csv"
test_hyperparams_RBF(X_train, y_train, X_test, y_test, filename=filename)

filename = "../../../output/bank_hyperparams_poly.csv"
test_hyperparams_poly(X_train, y_train, X_test, y_test, filename=filename)

filename = "../../../output/bank_hyperparams_linear.csv"
test_hyperparams_linear(X_train, y_train, X_test, y_test, filename=filename)


print('[Main] Info: ========================Begin testing - SKLearn linear ===================================')
print('[Main] Info: Building SVM model')
# model = svm.SVC(kernel='linear', gamma='auto', C=5)
model = svm.LinearSVC(C=5)
model.fit(X_train, y_train)

print('[Main] Info: Test model on test data')
test_prediction = model.predict(X_test)
print('[Main] Info: Finished prediction!  Test results:')
print(f' - Total test samples: {len(test_prediction)}')
print(f' - Correct: {(test_prediction == y_test).sum() / len(test_prediction)}')
print(f' - Incorrect: {(test_prediction != y_test).sum() / len(test_prediction)}')
print(f' - Model support vectors: {model.support_vectors_.shape}')


