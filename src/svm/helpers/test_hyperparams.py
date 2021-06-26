import pandas as pd
import numpy as np

from src.svm.classifiers.KernelSvmClassifier import KernelSvmClassifier
from src.svm.helpers.data_helpers import assess_accuracy


def test_hyperparams_RBF(X_train, y_train, X_test, y_test, filename=None, DEBUG=False):

    overall_results = pd.DataFrame(columns=['C', 'sigma', 'Total', 'Successes', 'False neg.', 'False pos.', '% Success',
                                            '% False neg', '% False pos'])
    for C in [100, 10, 1, 0.1]:
        for sigma in [0.0001, 0.001, 0.01, 0.1, 0.5, 0.75, 1]:      # Gamma = 1 / (2 * sigma**2)
            def RBF(x1, x2):
                """"RBF kernel """
                diff = x1 - x2
                return np.exp(-1 * (diff @ diff) / (2 * (sigma**2)))

            print()
            print('[Main] Info: ========================Begin testing - RBF Kernel ===================================')
            print(f'[Main] Info: Building SVM model with C={C}, sigma={sigma}')
            model = KernelSvmClassifier(C=C, kernel=RBF)
            model.fit(X_train, y_train, DEBUG=DEBUG)

            print('[Main] Info: Test model on test data')
            test_prediction = model.predict(X_test, DEBUG=DEBUG)

            print('[Main] Info: Finished prediction!  Test results:')
            run_results_df = assess_accuracy(test_prediction, y_test)

            run_row = run_results_df.iloc[0,:].to_dict()
            run_row['sigma'] = sigma
            run_row['C'] = C
            overall_results = overall_results.append(run_row, ignore_index=True)
            print(overall_results)
            if filename is not None:
                overall_results.to_csv(filename)

    return


def test_hyperparams_poly(X_train, y_train, X_test, y_test, filename=None, DEBUG=False):

    overall_results = pd.DataFrame(columns=['C', 'Polynom rank', 'Total', 'Successes', 'False neg.', 'False pos.', '% Success',
                                            '% False neg', '% False pos'])
    for C in [100, 10, 1, 0.1, 0.01]:
        for poly_rank in [1, 3, 5, 10, 15]:
            def poly(x1, x2):
                """"Poly kernel"""
                return (x1 @ x2 + 1) ** poly_rank

            print()
            print('[Main] Info: ========================Begin testing - RBF Kernel ===================================')
            print(f'[Main] Info: Building SVM model with C={C}, Polynom rank = {poly_rank}')
            model = KernelSvmClassifier(C=C, kernel=poly)
            model.fit(X_train, y_train, DEBUG=DEBUG)
            print('[Main] Info: Test model on test data')
            test_prediction = model.predict(X_test, DEBUG=DEBUG)
            print('[Main] Info: Finished prediction!  Test results:')

            run_results_df = assess_accuracy(test_prediction, y_test)
            # print(run_results_df)
            run_row = run_results_df.iloc[0,:].to_dict()
            run_row['Polynom rank'] = poly_rank
            run_row['C'] = C
            overall_results = overall_results.append(run_row, ignore_index=True)

            print(overall_results)
            if filename is not None:
                overall_results.to_csv(filename)
    print('Finished all cycles!')
    return



def test_hyperparams_linear(X_train, y_train, X_test, y_test, filename=None, DEBUG=False):

    overall_results = pd.DataFrame(columns=['C', 'Polynom rank', 'Total', 'Successes', 'False neg.', 'False pos.', '% Success',
                                            '% False neg', '% False pos'])
    for C in [100, 10, 1, 0.1, 0.01]:
        def linear_kernel(x1, x2):
            """"Linear kernel"""
            return (x1 @ x2)

        print()
        print('[Main] Info: ========================Begin testing - Linear Kernel ===================================')
        print(f'[Main] Info: Building SVM model with C={C}')
        model = KernelSvmClassifier(C=C, kernel=linear_kernel)
        model.fit(X_train, y_train, DEBUG=DEBUG)
        print('[Main] Info: Test model on test data')
        test_prediction = model.predict(X_test, DEBUG=DEBUG)
        print('[Main] Info: Finished prediction!  Test results:')

        run_results_df = assess_accuracy(test_prediction, y_test)
        # print(run_results_df)
        run_row = run_results_df.iloc[0,:].to_dict()
        run_row['C'] = C
        overall_results = overall_results.append(run_row, ignore_index=True)

        print(overall_results)
        if filename is not None:
            overall_results.to_csv(filename)
    print('Finished all cycles!')
    return