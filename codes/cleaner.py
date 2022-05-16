from nltk.tokenize import word_tokenize
from nltk import tokenize
import pandas as pd
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


ner = pd.read_csv('ner_skill.csv')
ner = ner.drop(columns="id")

data = open("dataset/About/summary.txt")
data = data.read()
fill_sentences_id(text=data, dataframe=ner)
print(ner.head(130))
ner["words"] = ner["words"].fillna("missing")
for i in range(len(ner)):
    if ner.at[i, "words"] == "missing":
        ner.drop(index=i)
print(ner.head(130))
ner = ner.drop(columns="ss")
