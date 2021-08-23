from utils import *
from channel import *
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


# data = [['6061x2', '60612']]
# charNoisyChannel = CharNoisyChannel()
# transformations = []
# transformations.extend(charNoisyChannel.learn_transformation(err, clr) for err, clr in data)
# print(charNoisyChannel.fit(data))
# print(charNoisyChannel.get_exceptions())

# ==========
# Detection
# ==========
name2raw, name2cleaned, name2groundtruth = read_dataset(data_path)
for name in name2raw.keys():
    detection = detector.detect(name2raw[name], name2cleaned[name])
    print(detection)

