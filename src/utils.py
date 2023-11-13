import os
import functools

from typing import Callable, Tuple, Any, TypeVar, List, Dict
from timeit import default_timer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Generic type class for model and dataset objects.
Model = TypeVar('Model')
HFDataset = TypeVar('HFDataset')


def timer(func: Callable) -> Any:
    """
    Function for printing the elapsed system time in seconds, 
    if only the debug flag is set to True.
        
    Parameters
    ----------
    func: function to run using the decorator syntax. (Callable)

    Returns
    -------
    wrapper_func: the result of the wrapper function (Any)
    """
    if not (debug := True):
        return func
    @functools.wraps(func)
    def wrapper_func(*args, **kwargs):
        start_time = default_timer()
        result = func(*args, **kwargs)
        end_time = default_timer()
        print(f'\n{func.__name__}: {end_time - start_time} secs', end = '\n\n')
        return result
    return wrapper_func


@timer
def load_abstractive_models(
        language_model_paths: List[str],
        device: str = 'cpu',
    ) -> Dict[str, Tuple[Model, Model]]:
    """
    Utility function which loads the selected abstractive models.
    
    Parameters
    ------------
    language_model_paths: List of huggingface model paths (List[str]).
    device: device to load and run models ['cpu', 'cuda:0'] (str).

    Returns
    --------
    <object>: All model objects (Dict[str, Tuple[Model, Model]]).
    """
    models = dict()

    for model_path in language_model_paths:
        
        # Load the tokenizer either from the HuggingFace model hub or locally.
        tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length = 1024, truncation = True, padding = 'max_length')

        # Load the pre-trained language model either from the HuggingFace model hub or locally.
        language_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        # Send the model to the pre-specified device (cpu / gpu).
        language_model = language_model.to(device)

        # Assign the tokenizer and its language model to the corresponding model name key.
        models[model_path.split('/')[1]] = (tokenizer, language_model)

    return models


def csv_to_txt(dataset: HFDataset, save_dir: str):
    """
    Utility function which saves each csv entry as a seperate csv file.

    Parameters
    -----------
    dataset: csv dataset (HFDataset).
    save_dir: Output file directory (str).
   
    Returns
    --------
    None.
    """
    
    for i, item in enumerate(dataset): 
        with open(os.path.join(save_dir, f'{i}.txt'), 'w', encoding = 'utf-8-sig', errors = 'ignore') as f:
            f.write(item['abstract'])
    return
