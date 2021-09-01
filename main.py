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
config_path = "data/config"
method = "holodetect"
key2model = {
    "holodetect": HoloDetector
}
hparams = load_config(config_path)
getattr(hparams, method).num_gpus = 1
getattr(hparams, method).num_examples = 2
detector = key2model[method](getattr(hparams, method))
# data_path should contain raw & cleaned directories
name2raw, name2cleaned, name2groundtruth = read_dataset(data_path)

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
# ec_str_pairs = list(zip(name2cleaned['hospital.csv']['ProviderNumber'], name2raw['hospital.csv']['ProviderNumber']))
# generator = NCGenerator()
# data, labels = generator.fit_transform(ec_str_pairs, name2raw['hospital.csv']['ProviderNumber'].values.tolist())
# print(data, labels)

# =========
# Detection
# =========
# for name in name2raw.keys():
#     detection = detector.detect(name2raw[name], name2cleaned[name])
#     detection.to_csv(Path(output_path) / "prediction.csv", index=False)

# ==========
# Evaluation
# ==========
evaluator = Evaluator()
# evaluator.evaluate(detector, name2raw, name2cleaned, name2groundtruth, data_path, output_path)
# Evaluate single csv file
prob_df = pd.read_csv('result/prediction.csv')
for name in name2raw.keys():
    report = evaluator.evaluate_csv(prob_df, name2raw[name], name2cleaned[name], name2groundtruth[name], data_path, output_path)
    print(report)
