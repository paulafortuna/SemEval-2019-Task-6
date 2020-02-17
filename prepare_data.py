

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

generic_path = "/home/dtic/SemEval-2019-Task-6/training_short/"

zeerak_folder = "zeerak/"
zeerak_dataset = "dataset_hate_speech_racism_sexism_en_zeerak.csv"
ami_folder = "AMI_misoginy_en_it/"
ami_dataset = "en_training.tsv"
davidson_folder = "davidson/"
davidson_dataset = "dataset_hate_speech_en_davidson.csv"
toxicity_folder = "toxicity/"
toxicity_dataset = "toxicity_en.csv"
stormfront_sentence_folder = "stormfront_sentence/"
stormfront_dataset_sentence = "stormfront_data_annotated_by_sentence.csv"
stormfront_post_folder = "stormfront_post/"
stormfront_dataset_post = "stormfront_data_annotated_by_post_binary.csv"


def short_df(df):
    return df.head(n=50)


def create_dataset(folder, dataset_name, sep):
    reading_path = generic_path + folder + dataset_name
    df = pd.read_csv(reading_path, sep = sep)
    train, test = train_test_split(df, test_size=0.3)
    short_df(train).to_csv(generic_path + folder + 'train.tsv', sep = '\t')
    short_df(test).to_csv(generic_path + folder + 'test.tsv', sep = '\t')


#create_dataset(zeerak_folder, zeerak_dataset, ',')
#create_dataset(ami_folder, ami_dataset, '\t')
create_dataset(davidson_folder, davidson_dataset, ',')
#create_dataset(toxicity_folder, toxicity_dataset, ',')
#create_dataset(stormfront_sentence_folder, stormfront_dataset_sentence, ',')
#create_dataset(stormfront_post_folder, stormfront_dataset_post, ',')

trac_train_path = "/home/dtic/SemEval-2019-Task-6/training_short/TRAC_aggressive_en/agr_en_train.csv"
trac_test_path = "/home/dtic/SemEval-2019-Task-6/training_short/TRAC_aggressive_en/agr_en_dev.csv"
folder = "/home/dtic/SemEval-2019-Task-6/training_short/TRAC_aggressive_en/"

df = pd.read_csv(trac_train_path, sep = ",")
short_df(df).to_csv(folder + 'train.tsv', sep = '\t')
df = pd.read_csv(trac_test_path, sep = ",")
short_df(df).to_csv(folder + 'test.tsv', sep = '\t')

