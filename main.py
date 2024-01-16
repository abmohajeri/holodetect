from dotenv import load_dotenv

from detection.holodetect import HoloDetector
from evaluation import *
from utils import *
from utils.helpers import read_dataset
import time
# ======================
# Environment Variables
# ======================
t0 = time.time()
load_dotenv()
data_path = os.getenv('data_path')
output_path = os.getenv('output_path')
config_path = os.getenv('config_path')

hparams = Namespace(**yaml.full_load(open(config_path)))
detector = HoloDetector(hparams)
dataset = read_dataset(data_path)  # data_path should contain raw & cleaned directories
training_data = read_dataset(data_path, [0, 20])

constraint = False  # Change This to True if Constraints Provided
constraints = Parser().load_denial_constraints(training_data['raw'].columns, data_path + '/constraints.txt') if constraint else None
dataset['constraints'] = constraints
training_data['constraints'] = constraints

# ==========
# Evaluation
# ==========
evaluator = Evaluator()
evaluator.evaluate(detector, dataset, training_data, output_path)
print("*******Runtime********")
print(time.time() - t0)
