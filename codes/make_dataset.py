import time
import pandas as pd
import logging
import re
import numpy as np
from nltk import tokenize
from nltk.tokenize import word_tokenize
from pathlib import Path
from termcolor import colored
from tqdm.auto import tqdm, trange
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def find_accuracy(dataframe, skills_list):
    """a percentage between [0,100]:return"""
    # labels format is ["O", "B-SKILL" , "O-SKILL"]
    labels = list(dataframe["labels"])
    final_list = []
    for index, label in enumerate(labels):
        if label == "B-SKILL":
            skill_list = []
            string_skill = ""

            word = dataframe.at[index, "words"]
            skill_list.append(word)
            count = 1
            while True:

                if labels[index + count] == "I-SKILL":

                    i_skill = dataframe.at[(index+count), 'words']
                    count += 1
                    if not isinstance(i_skill, float):
                        skill_list.append(i_skill)
                else:
                    break

            string_skill = " ".join(skill_list)
            final_list.append(string_skill)
    labels_count = len(list(set(final_list)))
    skills_count = len(skills_list)
    print(colored(f"the number of labels is >>>>>>> {skills_count}", "green"))
    print(colored(f"the number of labels was founded is >>>>>>> {labels_count}", "green"))
    print(colored(f">>>>>>> [{skills_count-labels_count}] skill was not found ", "red"))
    not_in_list = [x for x in skills_list if not x in final_list]

    return not_in_list, list(set(final_list))




def split_by_sentences(text):
    """use NLTK Tokenize for split text file with sentences"""
    sentences_list = tokenize.sent_tokenize(text)
    return sentences_list


def find_labels(sentences, skills, index_sentence):
    """find labels of each word"""
    # TODO-01: make step for loop in sentences

    main_dataframe = pd.DataFrame({})
    if isinstance(sentences, list):
        for index, sentence in enumerate(sentences):
            text_token = word_tokenize(sentence)
            data_frame = pd.DataFrame({"sentence_id": index,
                                       "words": text_token,
                                       "labels": "E",
                                       })
            token_index = [[index, tok] for index, tok in enumerate(list(data_frame["words"]))]

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
                    for word in token_index:
                        if skill == word[1] and data_frame.at[word[0], "labels"] == "E":
                            data_frame.at[word[0], "labels"] = "B-SKILL"

            # data_frame["labels"] = data_frame["labels"].replace(r'^\s*$', "O", regex=True)
            data_frame["labels"] = data_frame["labels"].replace(r'E', "O", regex=True)
            if len(set(data_frame['labels'])) == 1:
                continue
            main_dataframe = main_dataframe.append(data_frame)
    if isinstance(sentences, str):
        text_token = word_tokenize(sentences)
        data_frame = pd.DataFrame({"sentence_id": index_sentence,
                                       "words": text_token,
                                       "labels": "E",
                                   })
        token_index = [[index, tok] for index, tok in enumerate(list(data_frame["words"]))]
        for skill in skills:
            skill_token = skill.split(" ")
            if len(skill_token) > 1:
                for index, token in enumerate(token_index):
                    check_token = text_token[index: len(skill_token) + index]
                    check_str = " ".join(check_token)
                    interval = [index, len(skill_token) + index]
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
                for word in token_index:
                    if skill == word[1] and data_frame.at[word[0], "labels"] == "E":
                        data_frame.at[word[0], "labels"] = "B-SKILL"
        data_frame["labels"] = data_frame["labels"].replace(r'E', "O", regex=True)

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

def get_similar_word(sentences, skills):
    """use diff-lib to get most similar word to skill"""
    if isinstance(sentences, list):
        final_list = []
        for sentence in tqdm(sentences, total=len(sentences)):
            for skill in skills:
                try:
                    pattern = rf"\b{skill}\b"
                    match = re.findall(pattern=pattern, string=sentence)
                    if not match == []:
                        for sk in match:
                            if not sk in final_list:
                                final_list.append(sk)
                                print(colored(f"skill was founded >>>>> [{sk}]", 'green'))
                except Exception as e:
                    continue
    elif isinstance(sentences, str):
        final_list = []
        for skill in skills:
            try:
                pattern = rf"\b{skill}\b"
                match = re.findall(pattern=pattern, string=sentences)
                if not match == []:
                    for sk in match:
                        if not sk in final_list:
                            final_list.append(sk)
                            print(colored(f"skill was founded >>>>> [{sk}]", 'green'))
            except Exception:
                continue
    else:
        raise "acceptable only list and str type"
    return final_list

def fill_sentences_id(dataframe, text):
    sentences_list = tokenize.sent_tokenize(text)
    sentences_id = []
    sen = []
    for index, token in enumerate(sentences_list):
        token_list = word_tokenize(token)
        for tok in token_list:
            sentences_id.append(index)
            sen.append(tok)
    series = pd.Series(sentences_id)

    try:
        dataframe["sentence_id"] = series
    except Exception as e:
        print(e)
    return dataframe


def extract_token(dataframe):
    """a dataframe with 2 columns and save to zip file in local path :returns"""

    filepath = Path('dataset/ner/ner_skill.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    return dataframe.to_csv(filepath, index=False)


with open('dataset/linkedin-skills/linkedin_skills.txt', "r", encoding="utf-8") as linkdin_skills:

    skills_list = linkdin_skills.read()
    skills_list = skills_list.split("\n")
    skills_list = list(map(lambda x: x.lower(), skills_list))


def make_dataset(text_name):
    """you can call this function for make your dataset for train your model
    and return a csv file in folder of dataset/ner/ner_skill.csv:return
    """
    with open(f"dataset/About/{text_name}", "r", encoding="utf-8") as text:
        text = text.read().lower()
        sentences_list = split_by_sentences(text=text)
        start_found = time.process_time()
        main_dataframe = pd.DataFrame({})

        for index, sentence in enumerate(tqdm(sentences_list, total=len(sentences_list))):
            founded_skill = get_similar_word(sentences=sentence, skills=skills_list)
            if not founded_skill == []:
                labels = find_labels(sentence, founded_skill, index_sentence=index)
                main_dataframe = main_dataframe.append(labels)
        print(colored(f"Finding step {time.process_time() - start_found} was took", "yellow"))
        extract_token(main_dataframe)


def skill_replace(data, ner):
    not_in_list, final_list = find_accuracy(dataframe=ner, skills_list=skills_list)
    sentences_list = split_by_sentences(text=data)
    iter_not_in_list = iter(not_in_list)
    generate_sentence = []
    for sentence in sentences_list:
        skill_check = []
        for skill in final_list:
            pattern = rf"\b{skill}\b"
            match = re.findall(pattern=pattern, string=sentence)
            if not match == []:
                for sk in match:
                    for it in range(len(match)):
                        print(sentence)
                        if not sk in skill_check:
                            skill_check.append(sk)
                            sentence = sentence.replace(sk, next(iter_not_in_list))

                        generate_sentence.append(sentence)
                        print(colored(sentence, "green"))
    return sk


