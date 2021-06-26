import pandas as pd
import numpy as np


def convert_class_to_int(df: pd.DataFrame, target_column: str, DEBUG=False) -> dict:
    """Converts DataFrame Target object column from string to integer in {-1, 1}, updates the dataframe in-place.
    Gets:   df - dataframe
            target_column - Target column name
    returns a dictionary of target values: {int value: string} """

    int_str_class_dict = {}
    str_int_class_dict = {}

    for col in df.columns:
        if df[col].dtype == object:
            if col == target_column:
                if df[col].dtype == object and len(pd.unique(df[col])) == 2:
                    col_str_value_list = df[col].unique()
                    col_int_value_list = np.array([-1, 1])
                else:
                    print('[convert_class_to_int] ERROR - Target column does not fix, convertion failed!')
                    return None

            else:
                col_str_value_list = df[col].unique()
                col_int_value_list = np.arange(len(col_str_value_list))

            int_str_class_dict[col] = dict(zip(col_int_value_list, col_str_value_list))
            str_int_class_dict[col] = dict(zip(col_str_value_list, col_int_value_list))

            df[col] = df[col].apply(lambda str: str_int_class_dict[col].get(str))
            df[col] = df[col].astype('int32')

            if DEBUG:
                print(f'[convert_class_to_int] Info: Column {col} converted, Target = {col == target_column}')

    return int_str_class_dict


def assess_accuracy(predictions: np.array, y: np.array):
    results = np.zeros(8)  # Col 0 - class, columns: 1 - total, 2 - successes, 3- false negatives,
    # 4 - false positives, 5 - %success, 6- % false neg, 7 - % false pos

    for i in range(len(y)):
        results[0] = 1
        results[1] += 1
        if predictions[i] == y[i]:
            results[2] += 1
        else:
            if y[i] == 1:
                results[4] += 1
            else:
                results[3] += 1

        results[5] = 100 * results[2] / results[1]
        results[6] = 100 * results[3] / results[1]
        results[7] = 100 * results[4] / results[1]

    results_df = pd.DataFrame([results],
                              columns=['Class', 'Total', 'Successes', 'False neg.', 'False pos.', '% Success',
                                       '% False neg', '% False pos'])
    return results_df


def normalize(X: np.array, DEBUG=False) -> np.array:
    if DEBUG:
        print('[data_helpers.normalize()]: entering method')

    X = X.astype('float32')
    for i in range(X.shape[1]):
        if DEBUG:
            print(f'[data_helpers.normalize()]: Before: Column {i} range: [{X[:, i].min()}, {X[:, i].max()}]')

        if X[:, i].max() - X[:, i].min() != 0:
            X[:, i] = (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min())
        else:
            X[:, i] = 0
        if DEBUG:
            print(f'[data_helpers.normalize()]: After: Column {i} range: [{X[:, i].min()}, {X[:, i].max()}]')

    return X
