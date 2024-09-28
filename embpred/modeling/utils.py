import numpy as np
import csv
import os
from loguru import logger

 # Calculate average metrics
def report_kfolds_results(model_dir, accs, aucs, macros, losses, accum_conf_mat, kfolds):
    avg_acc = np.mean(accs)
    std_acc = np.std(accs)
    avg_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    avg_macro = np.mean(macros)
    std_macro = np.std(macros)
    avg_loss = np.mean(losses)
    std_loss = np.std(losses)

    # Format the results string for printing/logging (if necessary)
    result_str = f'(K Fold Final Result)| avg_acc={(avg_acc * 100):.2f} +- {(std_acc * 100):.2f}, ' \
                f'avg_auc={(avg_auc * 100):.2f} +- {(std_auc):.2f}, ' \
                f'avg_macro={(avg_macro * 100):.2f} +- {(std_macro):.2f}, ' \
                f'avg_loss={(avg_loss):.2f} +- {(std_loss):.2f}\n'

    # Print or log the result string if needed
    print(result_str)

    assert(os.path.exists(model_dir))

    results_file = os.path.join(model_dir, 'kfold_results.csv')

    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['Metric', 'Mean', 'Standard Deviation'])
        # Write each metric separately with mean and standard deviation in different columns
        writer.writerow(['Accuracy (%)', f'{avg_acc * 100:.2f}', f'{std_acc * 100:.2f}'])
        writer.writerow(['AUC (%)', f'{avg_auc * 100:.2f}', f'{std_auc:.2f}'])
        writer.writerow(['Macro Avg (%)', f'{avg_macro * 100:.2f}', f'{std_macro:.2f}'])
        writer.writerow(['Loss', f'{avg_loss:.2f}', f'{std_loss:.2f}'])

    # Save averaged confusion matrix to CSV
    conf_matrix_file = os.path.join(model_dir, 'average_confusion_matrix.csv')
    avg_conf_matrix = accum_conf_mat / kfolds
    np.savetxt(conf_matrix_file, avg_conf_matrix, delimiter=",", fmt='%d')

    logger.info(f'Results saved to {results_file}')
    logger.info(f'Confusion matrix saved to {conf_matrix_file}')
                