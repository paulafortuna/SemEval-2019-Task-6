
from scripts_pipeline.PathsManagement import PathsManagement as Paths

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def read_original_data(self, dataset_name):
    """
    Reads a dataset from file. The method uses a bit copy paste of code, instead of encapsulation,
    This is because reading a dataset is usually very dependant of the specific case,
    :param dataset_name: Name of the dataset to be loaded. The dataset should contain train and test in separated files.
    """
    if (dataset_name == "hateval_en"):
        # read from file and extract the correct columns
        dataset_hs = pd.read_csv(Paths.hateval_en_text_data_train, delimiter='\t', encoding='utf-8')
        self.x_train = dataset_hs["text"].str.lower()
        self.y_train = dataset_hs[["HS"]].values
        self.x_train_id = dataset_hs[["id"]].values[:, 0]

        # read from file and extract the correct columns
        dataset_hs = pd.read_csv(Paths.hateval_en_text_data_test, delimiter='\t', encoding='utf-8')
        self.x_val = dataset_hs["text"].str.lower()
        self.y_val = dataset_hs[["HS"]].values
        self.x_val_id = dataset_hs[["id"]].values[:, 0]

        # makes a dictionary of the classes
        self.labels_index = {"0": 0}
        self.labels_index["1"] = 1
    elif (dataset_name == "pt_hierarchy"):
        dataset_hs = pd.read_csv(Paths.pt_hierarchy_dataset, delimiter='\t', encoding='utf-8')
        text = dataset_hs["text"].str.lower()
        y_values = dataset_hs[["hatespeech_comb"]].values

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(text, y_values, test_size=0.1,
                                                                              random_state=42, stratify=y_values)

        self.x_train_id = self.get_array_ids(self.x_train.size)
        self.x_val_id = self.get_array_ids(self.x_val.size)

        # makes a dictionary of the classes
        self.labels_index = {"0": 0}
        self.labels_index["1"] = 1
    elif (dataset_name == "offenseval"):
        # read from file and extract the correct columns
        def label_offensive(row):
            if row['subtask_a'] == 'OFF':
                return 1
            elif row['subtask_a'] != 'OFF':
                return 0

        dataset_hs = pd.read_csv(Paths.offenseval_en_text_data_train, delimiter='\t', encoding='utf-8')
        self.x_train = dataset_hs["tweet"].str.lower()
        self.y_train = dataset_hs.apply(lambda row: label_offensive(row), axis=1)
        self.x_train_id = dataset_hs[["id"]].values[:, 0]

        # read from file and extract the correct columns
        dataset_hs = pd.read_csv(Paths.offenseval_en_text_data_test, delimiter='\t', encoding='utf-8')
        self.x_val = dataset_hs["tweet"].str.lower()
        self.y_val = dataset_hs.apply(lambda row: label_offensive(row), axis=1)
        self.x_val_id = self.get_array_ids(self.x_val.size)

        # makes a dictionary of the classes
        self.labels_index = {"0": 0}
        self.labels_index["1"] = 1
    elif (dataset_name == "hateval_es"):
        # read from file and extract the correct columns
        dataset_hs = pd.read_csv(Paths.hateval_es_text_data_train, delimiter='\t', encoding='utf-8')
        self.x_train = dataset_hs["text"].str.lower()
        self.y_train = dataset_hs[["HS"]].values
        self.x_train_id = dataset_hs[["id"]].values[:, 0]

        # read from file and extract the correct columns
        dataset_hs = pd.read_csv(Paths.hateval_es_text_data_test, delimiter='\t', encoding='utf-8')
        self.x_val = dataset_hs["text"].str.lower()
        self.y_val = dataset_hs[["HS"]].values
        self.x_val_id = dataset_hs[["id"]].values[:, 0]

        # makes a dictionary of the classes
        self.labels_index = {"0": 0}
        self.labels_index["1"] = 1
    elif (dataset_name == "zeerak"):
        dataset_hs = pd.read_csv(Paths.test_dataset, delimiter=',', encoding='utf-8')
        text = dataset_hs["text"].str.lower()

        y_values = dataset_hs[["Class"]].values
        lb = preprocessing.LabelBinarizer()
        y_values = lb.fit_transform(y_values)
        y_values = y_values[:16907, 1:2] + y_values[:16907, 2:3]

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(text, y_values, test_size=0.1,
                                                                              random_state=42, stratify=y_values)

        self.x_train_id = self.get_array_ids(self.x_train.size)
        self.x_val_id = self.get_array_ids(self.x_val.size)

        # makes a dictionary of the classes
        self.labels_index = {"0": 0}
        self.labels_index["1"] = 1

    elif (dataset_name == "news_comments_en_dataset"):
        dataset_hs = pd.read_csv(Paths.news_comments_en_dataset, delimiter=',', encoding='utf-8')
        text = dataset_hs["text_full_reply"].str.lower()

        y_values = dataset_hs[["is_hate_speech"]].values[:, 0]

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(text, y_values, test_size=0.3,
                                                                              random_state=42, stratify=y_values)

        # makes a dictionary of the classes
        self.labels_index = {"0": 0}
        self.labels_index["1"] = 1

        self.x_train_id = self.get_array_ids(self.x_train.size)
        self.x_val_id = self.get_array_ids(self.x_val.size)

    elif (dataset_name == "news_hate_generation"):
        dataset_hs = pd.read_csv(Paths.news_hate_generation, delimiter=',', encoding='utf-8')
        text = dataset_hs["text_news"].str.lower()

        y_values = dataset_hs[["is_hate_generator"]].values[:, 0]

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(text, y_values, test_size=0.3,
                                                                              random_state=42, stratify=y_values)

        # makes a dictionary of the classes
        self.labels_index = {"0": 0}
        self.labels_index["1"] = 1

        self.x_train_id = self.get_array_ids(self.x_train.size)
        self.x_val_id = self.get_array_ids(self.x_val.size)

    elif (dataset_name == "test"):
        dataset_hs = pd.read_csv(Paths.test_dataset, delimiter=',', encoding='utf-8')
        text = dataset_hs["text"].str.lower()

        y_values = dataset_hs[["Class"]].values
        lb = preprocessing.LabelBinarizer()
        y_values = lb.fit_transform(y_values)
        y_values = y_values[:16907, 1:2] + y_values[:16907, 2:3]

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(text, y_values, test_size=0.1,
                                                                              random_state=42, stratify=y_values)

        self.x_train_id = self.get_array_ids(self.x_train.size)
        self.x_val_id = self.get_array_ids(self.x_val.size)

        self.x_train = self.x_train[0:1000]
        self.y_train = self.y_train[0:1000]
        self.x_val = self.x_val[:100]
        self.y_val = self.y_val[:100]
        self.x_train_id = self.x_train_id[0:1000]
        self.x_val_id = self.x_val_id[0:100]

        # makes a dictionary of the classes
        self.labels_index = {"0": 0}
        self.labels_index["1"] = 1
    elif (dataset_name == "hateval_en_my_division"):
        # read from file and extract the correct columns
        dataset_hs_train = pd.read_csv(Paths.hateval_en_text_data_train, delimiter='\t', encoding='utf-8')
        dataset_hs_test = pd.read_csv(Paths.hateval_en_text_data_test, delimiter='\t', encoding='utf-8')

        # glue both
        dataset_hs = pd.concat([dataset_hs_train, dataset_hs_test], axis=0)

        # extract the correct columns

        text = dataset_hs["text"].str.lower()
        y_values = dataset_hs[["HS"]].values[:, 0]

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(text, y_values, test_size=0.1,
                                                                              random_state=42, stratify=y_values)

        # makes a dictionary of the classes
        self.labels_index = {"0": 0}
        self.labels_index["1"] = 1

        self.x_train_id = self.get_array_ids(self.x_train.size)
        self.x_val_id = self.get_array_ids(self.x_val.size)

    elif (dataset_name == "stormfront"):

        # read from file and build the correct columns
        colnames = ['text']
        dataset_hate = pd.read_csv('/home/dtic/HatEval/original_datasets/white_supremacist_foruns/hate.txt',
                                   delimiter='\t', names=colnames, header=None, encoding='utf-8')
        dataset_hate['class'] = 1
        dataset_noHate = pd.read_csv('/home/dtic/HatEval/original_datasets/white_supremacist_foruns/noHate.txt',
                                     delimiter='\t', names=colnames, header=None, encoding='utf-8')
        dataset_noHate['class'] = 0

        # glue both
        dataset_hs = pd.concat([dataset_hate, dataset_noHate], axis=0)

        # randomize dataframe
        dataset_hs = dataset_hs.sample(frac=1)

        # get text
        text = dataset_hs["text"].str.lower()
        text = text.str.lower()

        y_values = dataset_hs[["class"]].values[:, 0]

        # divide into training and testing
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(text, y_values, test_size=0.1,
                                                                              random_state=42, stratify=y_values)

        # makes a dictionary of the classes
        self.labels_index = {"0": 0}
        self.labels_index["1"] = 1

        self.x_train_id = self.get_array_ids(self.x_train.size)
        self.x_val_id = self.get_array_ids(self.x_val.size)


    else:
        print("The specified dataset does not exist.")