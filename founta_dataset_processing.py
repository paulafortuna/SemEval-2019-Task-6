
import pandas as pd
from sklearn.model_selection import train_test_split

#####################
# remove retweets
#####################

data_path = "/Users/paulafortuna/PycharmProjects/SemEval-2019-Task-6/founta_dataset/hatespeech_text_label_vote_RESTRICTED_100K.csv"
retweet_path = "/Users/paulafortuna/PycharmProjects/SemEval-2019-Task-6/founta_dataset/retweets.csv"

# read both files to pandas
data = pd.read_csv(data_path, header=None, sep = "\t")
retweet = pd.read_csv(retweet_path, header=None, sep = "\t")

print(data.shape)

# for every instance on the list of retweets, keep the first, erase in the other dataset the lines with retweets
elements_to_erase = []
for index, row in retweet.iterrows():
    list_splitted = row[0].split(',')
    list_splitted.pop(0)
    elements_to_erase = elements_to_erase + list_splitted

elements_to_erase = list(map(lambda x: int(x) - 1, elements_to_erase))
data.drop(data.index[elements_to_erase], inplace=True)

# check the size reduces in around 8k
print(data.shape)

# after this divide in train and test
train, test = train_test_split(data, test_size=0.3)
train.to_csv("/Users/paulafortuna/PycharmProjects/SemEval-2019-Task-6/founta_dataset/" + 'train.tsv', sep = '\t')
test.to_csv("/Users/paulafortuna/PycharmProjects/SemEval-2019-Task-6/founta_dataset/" + 'test.tsv', sep = '\t')