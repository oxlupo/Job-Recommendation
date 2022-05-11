import pandas as pd
import logging
import re
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
from nltk import pos_tag, RegexpParser, tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

def split_by_sentences(text):
    """use NLTK Tokenize for split text file with sentences"""
    sentences_list = tokenize.sent_tokenize(text)
    return sentences_list


def split_by_token(sentences):
    return word_tokenize(sentences)

def make_pattern(skill):
    pass

def split_sentences_token(sentences_list):
    """sentences_list is arg :arg
       a Data-frame with number of each sentence and token :returns
    """
    if isinstance(sentences_list, list):
        main_dataframe = pd.DataFrame({

        }, columns=["sentences_id", "words"])
        for index, sentence in enumerate(sentences_list):

            token = split_by_token(sentence)
            data_frame = pd.DataFrame({
                f"sentences_id": f"{index}",
                f"words": token
            })
            main_dataframe = main_dataframe.append(data_frame, ignore_index=True)
    else:
        raise Exception("You must give the function a list of sentences")

    return main_dataframe
def get_similar_word(sentences, skills):
    """use diff-lib to get most similar word to skill"""
    pattern = r'\bMachine Learning\b'
    for skill in skills:
        pass
    if isinstance(sentences_list, list):
        for sentences in sentences_list:
            try:
                match = re.findall(pattern=pattern, string=sentences)
            except Exception:
                print(Exception)
    elif isinstance(sentences, str):
        try:
            match = re.findall(pattern=pattern, string=sentences)
        except Exception:
            print(Exception)
        return match


def extract_token(sentences_list):
    """a dataframe with 2 columns and save to zip file in local path :returns"""
    token_dataframe = split_sentences_token(sentences_list)

    compression_opts = dict(method="zip",
                            archive_name='sentences_token.csv',
                            )
    return token_dataframe.to_csv("token.zip", index=False, compression=compression_opts)


with open('dataset/linkdin-skills/linkedin_skills.txt', "r", encoding="utf-8") as linkdin_skills:

    skills = linkdin_skills.read()
    skills_list = skills.split("\n")

with open("dataset/About/zhiyunren.txt", "r", encoding="utf-8") as text:
    text = text.read()
    sentences_list = split_by_sentences(text=text)
    skill = get_similar_word(sentences=sentences_list, skills=skills_list)
    extract_token(sentences_list)


