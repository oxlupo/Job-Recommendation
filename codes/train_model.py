import json
from simpletransformers.ner import NERModel, NERArgs
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def read_csv_make_dataset(csv_path):
    """file_path is a directory and read csv with pandas library :arg"""
    dataset = pd.read_csv(csv_path)
    dataset["sentence_id"] = LabelEncoder().fit_transform(dataset["sentence_id"])
    dataset["labels"] = dataset["labels"].str.upper()

    X = dataset[["sentence_id", "words"]]
    Y = dataset["labels"]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    # building up train data and test data
    train_data = pd.DataFrame({"sentence_id": x_train["sentence_id"], "words": x_train["words"], "labels": y_train})
    test_data = pd.DataFrame({"sentence_id": x_test["sentence_id"], "words": x_test["words"], "labels": y_test})

    return train_data, test_data


def arg_config(config):
    """ model configuration argument for training the model
     args was return finally:returns
     """
    parameters = config["hyper-parameters"]
    args = NERArgs()
    args.num_train_epochs = parameters["num_train_epochs"]
    args.learning_rate = parameters["learning_rate"]
    args.overwrite_output_dir = True
    args.train_batch_size = parameters["train_batch_size"]
    args.eval_batch_size = parameters["eval_batch_size"]
    return args


def train_model(args, dataset, train_data, test_data):
    """ train model with simpletransformers"""

    label = dataset["labels"].unique().tolist()
    model = NERModel('bert', 'bert-base-cased', labels=label, args=args, use_cuda=False)

    return model.train_model(train_data, eval_data=test_data, acc=accuracy_score)


def evaluate_model(model, test_data):
    """ evaluate model and return result and training loss"""

    result, model_outputs, preds_list = model.eval_model(test_data)

    return result


def predication_model(model, sentences):
    """a dictionary with extract NER(Name Entity Extraction) :returns"""

    if isinstance(sentences, str):

        prediction, model_output = model.predict([sentences])
        return prediction

    elif isinstance(sentences, list):

        result = []
        for sentence in sentences:

            prediction, model_output = model.predict(sentence)
            result.append(prediction)
        return result
    else:
        raise Exception("TYPE-ERROR: only list and str format was acceptable")


with open("config/config.json") as config:
    config = json.load(config)
    args = arg_config(config)



