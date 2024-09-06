[![Python-Versions](https://img.shields.io/badge/python-3.9_|_3.10-blue.svg)]()
[![Open in HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Open_in_HuggingFace-orange)](https://huggingface.co/IMISLab/)
[![Software-License](https://img.shields.io/badge/License-Apache--2.0-green)](https://github.com/NC0DER/LMRank/blob/main/LICENSE)

# GreekT5
This repository hosts code for the paper:
* [Giarelis, N., Mastrokostas, C., & Karacapilidis, N. (2024). GreekT5: A Series of Greek Sequence-to-Sequence Models for News Summarization.](https://link.springer.com/chapter/10.1007/978-3-031-63215-0_5)


## About
This repository holds the evaluation code for a series of Greek News Summarization Sequence-to-Sequence models built using Huggingface Transformers.  
The models were trained on GreekSum as of our research.  
The proposed models were trained and evaluated on the same dataset against GreekBART.  


## Installation
```
pip install requirements.txt
```

## Example Code
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

model_name = 'IMISLab/GreekT5-umt5-base-greeksum'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name) 

summarizer = pipeline(
    'summarization',
    device = 'cpu',
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 128,
    truncation = True
)
    
text = 'Να πάρει ""ξεκάθαρη"" θέση σε σχέση με τον κίνδυνο μετάδοσης του κορονοϊού από τη Θεία Κοινωνία καλεί την κυβέρνηση και τον Πρωθυπουργό με ανακοίνωσή του τη Δευτέρα ο ΣΥΡΙΖΑ. ""Την ώρα που κλείνουν προληπτικά και ορθώς σχολεία, πανεπιστήμια, γήπεδα και λαμβάνονται ειδικά μέτρα ακόμη και για την ορκωμοσία της νέας Προέδρου της Δημοκρατίας, η Ιερά Σύνοδος της Εκκλησίας της Ελλάδος επιμένει ότι το μυστήριο της Θείας Κοινωνίας δεν εγκυμονεί κινδύνους μετάδοσης του κορονοϊού, καλώντας όμως τις ευπαθείς ομάδες να μείνουν σπίτι τους"", αναφέρει η αξιωματική αντιπολίτευση και συνεχίζει: ""Ωστόσο το πρόβλημα δεν είναι τι λέει η Ιερά Σύνοδος, αλλά τι λέει η Πολιτεία και συγκεκριμένα ο ΕΟΔΥ και το Υπουργείο Υγείας, που έχουν και την αποκλειστική κοινωνική ευθύνη για τη μη εξάπλωση του ιού και την προστασία των πολιτών"". ""Σε άλλες ευρωπαϊκές χώρες με εξίσου μεγάλο σεβασμό στη Χριστιανική πίστη και στο θρησκευτικό συναίσθημα, τα μυστήρια της Εκκλησίας είτε αναστέλλονται είτε τροποποιούν το τελετουργικό τους. Μόνο στη χώρα μας έχουμε το θλιβερό προνόμιο μιας πολιτείας που δεν τολμά να πει το αυτονόητο"", προσθέτει, τονίζοντας ότι ""η κυβέρνηση λοιπόν και το Υπουργείο Υγείας οφείλουν να πάρουν δημόσια μια ξεκάθαρη θέση και να μην θυσιάζουν τη δημόσια Υγεία στο βωμό του πολιτικού κόστους"". ""Συμφωνούν ότι η Θεία Κοινωνία δεν εγκυμονεί κινδύνους μετάδοσης του κορονοϊού; Δεν είναι θέμα ευσέβειας αλλά κοινωνικής ευθύνης. Και με τη Δημόσια υγεία δεν μπορούμε να παίζουμε"", καταλήγει η ανακοίνωση του γραφείου Τύπου του ΣΥΡΙΖΑ. *ΠΩΣ ΜΕΤΑΔΙΔΕΤΑΙ. Χρήσιμος οδηγός για να προστατευθείτε από τον κορονοϊό *ΤΑ ΝΟΣΟΚΟΜΕΙΑ ΑΝΑΦΟΡΑΣ. Ποια θα υποδέχονται τα κρούσματα κορονοϊού στην Ελλάδα. *ΤΑΞΙΔΙΑ. Κορονοϊός και αεροδρόμια: Τι να προσέξετε. *Η ΕΠΙΔΗΜΙΑ ΣΤΟΝ ΠΛΑΝΗΤΗ. Δείτε LIVE χάρτη με την εξέλιξη του κορονοϊού.'
output = summarizer('summarize: ' + text)
print(output[0]['summary_text'])
```

## Citation
The model has been officially released with the article:  
[GreekT5: A Series of Greek Sequence-to-Sequence Models for News Summarization](https://link.springer.com/chapter/10.1007/978-3-031-63215-0_5).  
If you use the code or model, please cite the following:

```bibtex
@inproceedings{giarelis2024greekt5,
  title={GreekT5: Sequence-to-Sequence Models for Greek News Summarization},
  author={Giarelis, Nikolaos and Mastrokostas, Charalampos and Karacapilidis, Nikos},
  booktitle={IFIP International Conference on Artificial Intelligence Applications and Innovations},
  pages={60--73},
  year={2024},
  organization={Springer}
}
```

## Contributors
* Nikolaos Giarelis (giarelis@ceid.upatras.gr)
* Charalampos Mastrokostas (cmastrokostas@ac.upatras.gr)
* Nikos Karacapilidis (karacap@upatras.gr)
