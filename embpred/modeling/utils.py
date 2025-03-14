import numpy as np
import csv
import os
from loguru import logger

def recover_original_filename(augmented_filename: str) -> str:
    """
    Recover the original image filename from an augmented filename.
    
    The augmented filename is assumed to have the format:
        <base_name>-aug<augmentation_number><extension>
    For example, "image1-aug1.jpg" becomes "image1.jpg".
    
    If the filename does not follow this pattern, it is returned unchanged.
    
    Parameters:
        augmented_filename (str): The augmented image file name.
    
    Returns:
        str: The original image file name.
    """
    base_name, ext = os.path.splitext(os.path.basename(augmented_filename))
    if "-aug" in base_name:
        original_base = base_name.split("-aug")[0]
        return original_base + ext
    return os.path.basename(augmented_filename)

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

def report_test_set_results(model_dir, test_micro, test_aucs, test_macro, avg_loss, conf_mat):
    # go back one directory for model_dir
    model_dir = os.path.dirname(model_dir)
    results_file = os.path.join(model_dir, 'test_results.csv')

    result_str = f'(Test Set Final Result)| test_micro={test_micro:.2f}, ' \
                f'test_auc={test_aucs:.2f}, ' \
                f'test_macro={test_macro:.2f}, ' \
                f'avg_loss={avg_loss:.2f}\n'
    print(result_str)

    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['Metric', 'Value'])
        # Write each metric separately with mean and standard deviation in different columns
        writer.writerow(['Micro Avg (%)', f'{test_micro:.2f}'])
        writer.writerow(['AUC (%)', f'{test_aucs:.2f}'])
        writer.writerow(['Macro Avg (%)', f'{test_macro:.2f}'])
        writer.writerow(['Loss', f'{avg_loss:.2f}'])
    
    # Save test set confusion matrix to CSV
    conf_matrix_file = os.path.join(model_dir, 'test_confusion_matrix.csv')
    np.savetxt(conf_matrix_file, conf_mat, delimiter=",", fmt='%d')
    
    logger.info(f'Results saved to {results_file}')
    logger.info(f'Test set confusion matrix saved to {conf_matrix_file}')
