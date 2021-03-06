# Quantifying-Intimacy-in-Language

Official Github Repo for EMNLP 2020 paper [Quantifying Intimacy in Language](https://arxiv.org/abs/2011.03020) by [Jiaxin Pei](https://jiaxin-pei.github.io/) and [David Jurgens](https://jurgens.people.si.umich.edu/).

## Data
### Annotated question intimacy data:
`data/annotated_question_intimacy_data` 

## Code
### Python pacakge for intimacy prediction
If `pip` is installed, question-intimacy could be installed directly via [pip](https://pypi.org/project/question-intimacy/):
```
pip3 install question-intimacy
```
    

### Pre-trained model
Our model is also available on [Hugging Face Transformers](https://huggingface.co/pedropei/question-intimacy)
```
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#load tokenizer and model, both will be automatically downloaded for the first usage
tokenizer = AutoTokenizer.from_pretrained("pedropei/question-intimacy")
model = AutoModelForSequenceClassification.from_pretrained("pedropei/question-intimacy")
```

### Code to train the intimacy regressor

Forthcoming


## Contact
Jiaxin Pei (pedropei@umich.edu)
