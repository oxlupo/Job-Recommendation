import csv
from simpletransformers.ner import NERModel, NERArgs
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
from nltk import pos_tag, RegexpParser, tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# model_args = NERArgs()
#
# custom_labels = ["O", "B-SKILL", "I-SKILL", "B-DIPLOMA-MAJOR", "I-DIPLOMA-MAJOR", "B-EXPERIENCE", "I-EXPERIENCE", "B-ORG", "I-ORG", "B-PERSON", "I-PERSON"]
# model_args.labels_list = custom_labels
#
# model = NERModel(
#     "roberta",
#     "roberta-base",
#     args=model_args,
#     use_cuda=False
# )

# model.train_model()
linkdin_skills = open('linkedin_skills.txt', "r", encoding="utf-8")
skills = linkdin_skills.read()
skills_list = skills.split("\n")

def split_by_sentences(text):
    """use NLTK Tokenize for split text file with sentences"""
    sentences_list = tokenize.sent_tokenize(text)
    return sentences_list

def write_to_csv_sentences(sentences_list, index):
    """ a function for save as a specific format for train the simpletransformers model """
    data_frame = pd.DataFrame({f"sentences {index}": f"sentences{index}", "data": sentences_list})
    print(data_frame)

def split_by_token(sentences):
    return word_tokenize(sentences)

cample = open("About/camharvey.txt", "r", encoding="utf-8")
text_cample = cample.read()
sentences_list = split_by_sentences(text=text_cample)

def split_sentences_token(sentences_list):
    """sentences_list is arg :arg
       a Data-frame with number of each sentences and token :returns
    """
    if isinstance(sentences_list, list):
        main_dataframe = pd.DataFrame({

        }, columns=["sentences", "tokens"])
        for index, sentences in enumerate(sentences_list, start=1):
            token = split_by_token(sentences)
            data_frame = pd.DataFrame({
                f"sentences": f"sentences {index}",
                f"tokens": token
            })

            main_dataframe = main_dataframe.append(data_frame, ignore_index=True)
    else:
        raise Exception("You must give the function a list of sentences")

    return main_dataframe

token_dataframe = split_sentences_token(sentences_list)

compression_opts = dict(method="zip",
                        archive_name='sentences_token.csv',
                        )
token_dataframe.to_csv("token.zip", index=False, compression=compression_opts)
