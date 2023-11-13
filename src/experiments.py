import os
import pathlib

from typing import TypeVar, Dict, List, Tuple
from statistics import mean
from tqdm import tqdm
from evaluate import load
from src.models import textrank_gr, lead_gr, abstractive_model_inference

# Generic type class for model and dataset objects.
Model = TypeVar('Model')
HFDataset = TypeVar('HFDataset')


def produce_summaries(dataset: HFDataset, save_dir: str, model_names: List[str], 
                      abstractive_models: Dict[str, Tuple[Model, Model]]):
    """
    This function produces summaries for each dataset item 
    and saves these summaries as separate .txt files in a separate directory for each model.

    Parameters
    -----------
    dataset: Dataset to produce summaries from (HFDataset).
    save_dir: Save directory for all model produced summaries (str).
    model_names: List of model names (List[str]).
    abstractive_models: Dictionary of summarization models (Dict[str, Tuple[Model, Model]]).

    Returns
    --------
    None.
    """

    # Iterate each dataset item (article).
    for i, item in tqdm(enumerate(dataset)):

        # Produce a summary for each model.
        for model_name in model_names:

            if model_name == 'textrank':
                summary = textrank_gr(item['article'], 1)
            elif model_name == 'lead':
                summary = lead_gr(item['article'], 1)
            else:
                summary = abstractive_model_inference(
                    item['article'], *abstractive_models[model_name]
                )

            # Save the summary in a separate file for each dataset entry.
            output_path = os.path.join(save_dir, model_name)
            pathlib.Path(output_path).mkdir(parents = True, exist_ok = True)
            
            with open(os.path.join(output_path, f'{i}.txt'),
                      'w', encoding = 'utf-8-sig', errors = 'ignore') as f:
                f.write(summary)
    return


def evaluate(produced_path: str, reference_path: str,
             dataset_length: int, slice_size: int) -> Dict[str, int]:
    """
    This function compares the produced summaries of each method
    against the reference summaries. The evaluation metrics are ROUGE and BERTScore. 
    For each score we calculate the macro(mean) F1 score.
 
    Parameters
    -----------
    produced_path: Path of the machine produced summaries (str).
    reference_path: Path of the human written summaries (str).
    dataset_length: The length of the dataset (int).
    slice_size: The size of each dataset slice passed to the BERTScore scorer, 
    needs to be a whole multiple of dataset_length (int).
   
    Returns
    --------
    metrics: A dictionary which contains the values of each evaluation metric (Dict[str, int]).
    """
    # Initialize the metric scores.
    metric_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'bertscore': 0.0}
    
    # Initialize the evaluation metrics.

    bertscore = load('bertscore') 
    rouge = load('rouge')

    # Initialize the prediction and reference lists.
    predictions, references = [], []

    # Read the summary text files and pass them into the lists.
    for i in tqdm(range(dataset_length), desc = 'Loading files for evaluation...'):
        with open(os.path.join(produced_path, f'{i}.txt'), 'r', encoding = 'utf-8-sig', errors = 'ignore') as pred, \
            open(os.path.join(reference_path, f'{i}.txt'), 'r', encoding = 'utf-8-sig', errors = 'ignore') as ref:
            predictions.append(pred.read())
            references.append(ref.read())

    # Split the dataset into slices and calculate the BERTScore for each one.
    for i in tqdm(range(dataset_length // slice_size), desc = f'Computing BERTScore metric scores...'):
        metric_scores['bertscore'] += mean(bertscore.compute(
            predictions = predictions[i * slice_size:i * slice_size + slice_size],
            references = references[i * slice_size:i * slice_size + slice_size],
            lang = 'el', device = 'cpu')['f1']
        )

    if (mod := dataset_length % slice_size):
        print('Computing BERTScore metric scores for the remainder documents...')
        metric_scores['bertscore'] += mean(bertscore.compute(
            predictions = predictions[(i + 1) * slice_size:(i + 1) * slice_size + mod],
            references = references[(i + 1) * slice_size:(i + 1) * slice_size + mod],
            lang = 'el', device = 'cpu')['f1']
        )
        i += 1
    
    # Calculate the macro BERTScore.
    metric_scores['bertscore'] = metric_scores['bertscore'] / (i + 1)

    # Compute the ROUGE metric score for each ROUGE metric.
    for i in tqdm(range(dataset_length), desc = 'Computing ROUGE metric scores...'):   
        
        # For each non-empty pair of summaries compute the ROUGE metric scores.
        if predictions[i] and not predictions[i].isspace():
            scores = rouge.compute(predictions = [predictions[i]], references = [references[i]], tokenizer = lambda x: x.split())
            metric_scores['rouge1'] += scores['rouge1']
            metric_scores['rouge2'] += scores['rouge2']
            metric_scores['rougeL'] += scores['rougeL']
            
 
    # Calculate the macro ROUGE metric scores.
    # Empty documents are taken into account by dividing with the entire dataset length. 
    metric_scores['rouge1'] = metric_scores['rouge1'] / dataset_length
    metric_scores['rouge2'] = metric_scores['rouge2'] / dataset_length
    metric_scores['rougeL'] = metric_scores['rougeL'] / dataset_length

    return metric_scores
