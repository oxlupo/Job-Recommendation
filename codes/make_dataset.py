import mistune
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


def find_labels(text, skills):
    """find labels of each word"""
    text_token = word_tokenize(text)

    data_frame = pd.DataFrame({"sentences_id": " ",
                               "token": text_token,
                               "labels": " ",
                               })

    token_index = [[index, tok] for index, tok in enumerate(list(data_frame["token"]))]
    for skill in skills:
        skill_token = skill.split(" ")
        if len(skill_token) > 1:
            for index, token in enumerate(token_index):
                check_token = text_token[index: len(skill_token) + index]
                check_str = " ".join(check_token)
                interval = [index, len(skill_token)+index]
                if skill == check_str:
                    first = skill_token[0]
                    labels = list(map(lambda x: [x, "B-SKILL"] if x == first else [x, "I-SKILL"], skill_token))
                    for lab in labels:
                        if labels[0] == lab:
                            lab.append(interval[0])
                        else:
                            for inter in range(interval[0] + 1, interval[1]):
                                lab.append(inter)

                    for label in labels:
                        data_frame.at[label[2], "labels"] = label[1]

        else:
            for index in token_index:
                if skill == index[1]:
                    data_frame.at[index[0], "labels"] = "B-SKILL"
        data_frame["labels"] = data_frame["labels"].replace(r'^\s*$', "O", regex=True)
    return data_frame

def split_by_token(sentences):
    return word_tokenize(sentences)


def split_sentences_token(sentences_list, labels):
    """sentences_list is arg :arg
       a Data-frame with number of each sentence and token :returns
    """
    if isinstance(sentences_list, list):
        main_dataframe = pd.DataFrame({

        }, columns=["sentences_id", "words", "labels"])
        for index, sentence in enumerate(sentences_list):

            token = split_by_token(sentence)
            data_frame = pd.DataFrame({
                f"sentences_id": f"{index}",
                f"words": token,
                "labels": "s"
            })
            main_dataframe = main_dataframe.append(data_frame, ignore_index=True)
    else:
        raise Exception("You must give the function a list of sentences")

    return main_dataframe
def get_similar_word(sentence, skills):
    """use diff-lib to get most similar word to skill"""
    if isinstance(sentence, list):
        final_list = []
        for sentence in sentence:
            for skill in skills:
                try:
                    pattern = rf"\b{skill}\b"
                    match = re.findall(pattern=pattern, string=sentence)
                    if not match == []:
                        for sk in match:
                            final_list.append(sk)
                except Exception:
                    continue
    elif isinstance(sentence, str):
        final_list = []
        for skill in skills:
            try:
                pattern = rf"\b{skill}\b"
                match = re.findall(pattern=pattern, string=sentence)
                if not match == []:
                    for sk in match:
                        final_list.append(sk)
            except Exception:
                continue
    else:
        raise "acceptable only list and str type"
    return list(set(final_list))


def extract_token(sentences_list, labels):
    """a dataframe with 2 columns and save to zip file in local path :returns"""
    token_dataframe = split_sentences_token(sentences_list, labels)

    compression_opts = dict(method="zip",
                            archive_name='sentences_token.csv',
                            )
    return token_dataframe.to_csv("token.zip", index=False, compression=compression_opts)


with open('dataset/linkdin-skills/linkedin_skills.txt', "r", encoding="utf-8") as linkdin_skills:

    skills = linkdin_skills.read()
    skills_list = skills.split("\n")
    skills_list = list(map(lambda x: x.lower(), skills_list))
with open("dataset/About/zhiyunren.txt", "r", encoding="utf-8") as text:
    text = text.read().lower()
    sentences_list = split_by_sentences(text=text)
    founded_skill = get_similar_word(sentence=sentences_list, skills=skills_list)
    labels = find_labels(text, founded_skill)
    print(labels)
    extract_token(sentences_list, founded_skill)
