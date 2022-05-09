import json
from simpletransformers.ner import NERModel, NERArgs
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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

with open("config.json") as config:
    config = json.load(config)
    arg = arg_config(config)
