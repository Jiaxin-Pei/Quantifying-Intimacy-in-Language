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
To fine-tune the roberta-base model over our intimacy dataset
```
python3 train_intimacy_model.py --mode=train \
--model_name=roberta-base \
--pre_trained_model_name_or_path=roberta-base \
--train_path=data/annotated_question_intimacy_data/final_train.txt \
--val_path=data/annotated_question_intimacy_data/final_val.txt \
--test_path=data/annotated_question_intimacy_data/final_test.txt \
--model_saving_path=outputs 
```
The best model will be saved at `outputs/`

after training, to get the score on our annotated test and out-domain set,
```
python3 train_intimacy_model.py --mode=internal-test \
--model_name=roberta-base \
--pre_trained_model_name_or_path=outputs \
--train_path=data/annotated_question_intimacy_data/final_train.txt \
--val_path=data/annotated_question_intimacy_data/final_val.txt \
--test_path=data/annotated_question_intimacy_data/final_test.txt \
--predict_data_path=data/annotated_question_intimacy_data/final_external.txt 
```

to run the fine-tuned model over your own data, prepare a file with a list of input text like `data/inference.txt` and run the following command
```
python3 train_intimacy_model.py --mode=inference \
--model_name=roberta-base \
--pre_trained_model_name_or_path=outputs \
--predict_data_path=data/inference.txt \
--test_saving_path=ooo.txt
```
if you want to do language modeling fine-tuning for the roberta-base model, please checkout the code from [Hugging Face Transformers](https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py)

to train the fine-tuned roberta model over our intimacy dataset, put the model under `saved_model` and run the following command:
```
python3 train_intimacy_model.py --mode=train \
--model_name=roberta-ft \
--pre_trained_model_name_or_path=saved_model \
--train_path=data/annotated_question_intimacy_data/final_train.txt \
--val_path=data/annotated_question_intimacy_data/final_val.txt \
--test_path=data/annotated_question_intimacy_data/final_test.txt \
--model_saving_path=outputs 
```
Please email Jiaxin Pei (pedropei@umich.edu) to request the roberta-base model fine-tuned over 3M questions.


## Contact
Jiaxin Pei (pedropei@umich.edu)
