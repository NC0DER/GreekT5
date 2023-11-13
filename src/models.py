from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
from typing import TypeVar
from transformers import pipeline

# Generic type class for model objects.
Model = TypeVar('Model')

def lead_gr(text: str, n: int) -> str:
    """
    This function implements the LEAD-N method described in many summarization papers.
    This method extracts the first n sentences from the inputted text.
    In its current form, this function is built for tokenizing greek texts, 
    but it can easily be adapted to other languages by changing the tokenizer.

    Parameters
    -----------
    text: text to be summarized (str).
    n: number of leading sentences  to be extracted (int).
   
    Returns
    summary: An extractive summary with the first n sentences (str).
    --------
  
    """
    # Initialize the Greek sentence tokenizer.
    greek_sentence_tokenizer = Tokenizer('greek')._get_sentence_tokenizer('greek')

    # Split the greek text into sentences. 
    sentences = greek_sentence_tokenizer.tokenize(text)

    # The first n sentences are the summary.
    summary = ' '.join(sentences[:n])

    return summary


def textrank_gr(text: str, n: int) -> str:
    """
    This function uses the textrank implementation of sumy to rank sentences 
    and return the top-n in a summary. In its current form, this function 
    is built for tokenizing greek texts, but it can easily be adapted to 
    other languages by changing the tokenizer.

    Parameters
    -----------
    text: text to be summarized (str).
    n: number of sentences to be extracted (int).
   
    Returns
    summary: An extractive summary of n sentences (str).
    --------
    """
    # Initialize the plain text parser and summarizer of sumy.
    parser = PlaintextParser.from_string(text, Tokenizer('greek'))
    summarizer = TextRankSummarizer()
    
    # Rank sentences using TextRank and extract the top-n.
    summary = ' '.join([
        str(sentence) 
        for sentence in summarizer(parser.document, sentences_count = n)
    ])

    return summary


def abstractive_model_inference(text: str, tokenizer: Model, model: Model) -> str:

    """
    This function summarizes a text using the selected Huggingface model.

    Parameters
    -----------
    text: text to be summarized (str).
    tokenizer: the tokenizer model (Model).
    model: the Huggingface model (Model).
   
    Returns
    summary: An abstractive summary (str).
    --------
    """
    # Initialize the summarization pipeline.  
    summarizer = pipeline('summarization', model = model, tokenizer = tokenizer, device = 'cpu', max_new_tokens = 128, truncation = True)
    summary = summarizer('summarize: ' + text)

    return summary[0]['summary_text']
