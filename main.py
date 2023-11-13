import os
import pathlib

from datasets import load_dataset
from src.utils import load_abstractive_models, csv_to_txt
from src.experiments import produce_summaries, evaluate


def run_experiments():
    
    # Initialize a list with the selected abstractive models.
    model_paths = [
        'IMISLab/GreekT5-mt5-small-greeksum', 
        'IMISLab/GreekT5-umt5-small-greeksum', 
        'IMISLab/GreekT5-umt5-base-greeksum'
    ]
    
    # Load the testing dataset (.csv).
    csv_path = 'datasets/greeksum_test.csv'
    test_dataset = load_dataset('csv', data_files = csv_path, split = 'all') 

    # Create a directory for the reference summaries and save each one in a separate .txt file. 
    reference_path = 'datasets/greeksum-test-reference-summaries'
    pathlib.Path(reference_path).mkdir(parents = True, exist_ok = True)
    csv_to_txt(test_dataset, reference_path)

    # Load all required models.
    models = load_abstractive_models(
        language_model_paths = model_paths, 
        device = 'cpu'
    )

    # Set the selected summarization methods.
    model_names = ['textrank', 'lead', *map(lambda x: x.split('/')[1], model_paths)]

    # Set the output path for the produced summaries.
    output_path = 'produced'

    # # Produce summaries for each method.
    produce_summaries(test_dataset, output_path, model_names, models)

    # Evaluate the selected methods against the reference summaries.
    for method in model_names:
        print(f'\n{method}:')
        print(evaluate(os.path.join(output_path, method), reference_path,  dataset_length = 10000, slice_size = 1000))
        
    return


if __name__ == '__main__': run_experiments()
