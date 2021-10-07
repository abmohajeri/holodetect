from detection.features import *
from utils import *
from channel import *
from evaluation import *
from utils.helpers import read_dataset
from detection.holodetect import HoloDetector

# ======================
# Environment Variables
# ======================
output_path = "result"
data_path = "data/datasets/holodetect/hospital"
# data_path = "data/datasets/other/movies"
# data_path = "data/datasets/other/flights"
# data_path = "data/datasets/holodetect/soccer"
config_path = "data/config"
method = "holodetect"
key2model = {
    "holodetect": HoloDetector
}
hparams = load_config(config_path)
detector = key2model[method](getattr(hparams, method))
dataset = read_dataset(data_path) # data_path should contain raw & cleaned directories
training_data = read_dataset(data_path, [0, 100])

# =============
# Noisy Channel
# =============
# data = [['60612', '6061x2']]
# charNoisyChannel = CharNoisyChannel()
# transformations = []
# transformations.extend(charNoisyChannel.learn_transformation(clr, err) for clr, err in data)
# print(charNoisyChannel.fit(data))

# =================
# Data Augmentation
# =================
# column = 'ProviderNumber'
# training_cleaned_values = [
#     RowBasedValue(value, row_dict, column)
#     for value, row_dict in zip(training_data['clean'][column].values.tolist(), training_data['clean'].to_dict("records"))
# ]
# training_values = [
#     RowBasedValue(value, row_dict, column)
#     for value, row_dict in zip(training_data['raw'][column].values.tolist(), training_data['raw'].to_dict("records"))
# ]
# raw_values = [
#     RowBasedValue(value, row_dict, column)
#     for value, row_dict in zip(dataset['raw'][column].values.tolist(), dataset['raw'].to_dict("records"))
# ]
# ec_str_pairs = list(zip(training_cleaned_values, training_values))
# data, labels = NCGenerator().fit_transform(ec_str_pairs, raw_values)
# assert (len(data) == len(labels))
# print(len(data))
# print([x.value for x in data])
# print(labels)

# =========
# Detection
# =========
# detection = detector.detect(dataset, training_data)
# detection.to_csv(output_path + "/prediction.csv", index=False)

# ==========
# Evaluation
# ==========
evaluator = Evaluator()
evaluator.evaluate(detector, dataset, training_data, output_path)

# Evaluate single csv file
# prob_df = pd.read_csv('result/prediction.csv')
# report = evaluator.evaluate_csv(prob_df, dataset, output_path)
# print(report)
