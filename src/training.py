import torch

from typing import List, TypeVar

from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)

# Generic type class for model and dataset objects.
Model = TypeVar('Model')
HFDataset = TypeVar('HFDataset')


def train_model(train_dataset: HFDataset, validation_dataset: HFDataset, tokenizer: Model, model: Model, output_dir: str) -> None:
    """
    Parameters
    -----------
    train_dataset: HuggingFace training dataset (HFDataset).
    validation_dataset: HuggingFace validation dataset (HFDataset).
    tokenizer: huggingface tokenizer model (Model).
    model: huggingface language model (Model).
    device: device to load and run model ['cpu', 'cuda:0'] (str).
    output_dir: directory to save model checkpoints (str).

    Returns
    --------
    None.
    """
    
    def preprocess_function(sample, padding = 'max_length', max_input_length = 1024, max_output_length = 128, truncation = True):
        
        # add prefix to the input for t5
        inputs = ['summarize: ' + item for item in sample['article']]

        # tokenize inputs
        model_inputs = tokenizer(inputs, max_length = max_input_length, padding = padding, truncation = truncation)

        # Tokenize outputs with the `text_target` keyword argument
        labels = tokenizer(text_target = sample['abstract'], max_length = max_output_length, padding = padding, truncation = truncation)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == 'max_length':
            labels['input_ids'] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels['input_ids']
            ]

        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    # Apply the tokenization function to the datasets. 
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched = True, remove_columns = ['article', 'abstract'])

    tokenized_validation_dataset = validation_dataset.map(preprocess_function, batched = True, remove_columns = ['article', 'abstract'])

    # As said in the documentation: "It is more efficient to dynamically pad 
    # the sentences to the longest length in a batch during collation, 
    # instead of padding the whole dataset to the maximum length."
    data_collator = DataCollatorForSeq2Seq(
        tokenizer = tokenizer,
        model = model,
        label_pad_token_id = -100,
        pad_to_multiple_of = 8
    )

    # Define the evaluation metrics during fine-tuning.
    def compute_metrics(pred):
        pass

    # Create a Huggingface Seq2SeqTrainer object. 
    # When an existing model checkpoint is passed,
    # we are essentialy doing fine-tuning.
    # If the execution is stopped, 
    # the latest checkpoint and its training state are automatically loaded.
    trainer = Seq2SeqTrainer(
        model = model,
        args = Seq2SeqTrainingArguments(
                output_dir = output_dir,
                evaluation_strategy = 'no',
                learning_rate = 3e-04,
                resume_from_checkpoint = False,
                per_device_train_batch_size = 8,
                per_device_eval_batch_size = 1,
                fp16 = False, # True overflows on t5 models.
                ignore_data_skip = False,
                overwrite_output_dir = False,
                log_level = 'info',
                logging_steps = 1000,
                save_strategy = 'epoch',
                save_total_limit = 4,
                num_train_epochs = 10
        ),
        train_dataset = tokenized_train_dataset,
        eval_dataset = tokenized_validation_dataset,
        data_collator = data_collator,
        compute_metrics = compute_metrics
    )

    torch.cuda.empty_cache()

    # Start the training loop.
    trainer.train(resume_from_checkpoint = False)

    return
