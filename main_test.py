# ========================
# Test All Part of Package
# ========================

import os
# from detection.features import *
from channel import CharNoisyChannel
from utils import *
# from channel import *
from evaluation import *
from utils.helpers import read_dataset
from detection.holodetect import HoloDetector
from dotenv import load_dotenv

# ======================
# Environment Variables
# ======================
load_dotenv()
data_path = os.getenv('data_path')
output_path = os.getenv('output_path')
config_path = os.getenv('config_path')

hparams = Namespace(**yaml.full_load(open(config_path)))
detector = HoloDetector(hparams)
dataset = read_dataset(data_path)  # data_path should contain raw & cleaned directories
training_data = read_dataset(data_path, [0, 9])

constraint = True  # Change This to True if Constraints Provided
constraints = Parser().load_denial_constraints(training_data['raw'].columns, data_path + '/constraints.txt') if constraint else None
dataset['constraints'] = constraints
training_data['constraints'] = constraints

# =============
# Noisy Channel
# =============
data = [['60612', '6061x2']]
charNoisyChannel = CharNoisyChannel()
transformations = []
transformations.extend(charNoisyChannel.learn_transformation(clr, err) for clr, err in data)
print(charNoisyChannel.fit(data))
print(transformations)

# =================
# Data Augmentation
# =================
# column = 'ProviderNumber' # Hospital Dataset
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

# ========================
# Denial Constraint Parser
# ========================
# errors = ViolationDetector(training_data['raw'], constraints).detect_noisy_cells()
# print(errors)

# =========
# Detection
# =========
# detection = detector.detect(dataset, training_data)
# detection.to_csv(output_path + "/prediction.csv", index=False)

# ==========
# Evaluation
# ==========
# evaluator = Evaluator()
# evaluator.evaluate(detector, dataset, training_data, output_path)

# Evaluate single csv file
# evaluator = Evaluator()
# prob_df = pd.read_csv('result/toy.csv/predictions.csv')
# report = evaluator.evaluate_csv(prob_df, dataset, output_path)
