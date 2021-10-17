import csv
import pandas as pd
from utils.helpers import diff_dfs, not_equal
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# Output from - to - id - col - score (probability) - result (true or false)
class Report:
    def __init__(self, prob_df, dataset, threshold=0.5):
        raw_df = dataset['raw']
        cleaned_df = dataset['clean']
        groundtruth_df = dataset['groundtruth']
        self.prob_df = prob_df
        self.threshold = threshold
        detected_df = prob_df.applymap(lambda x: x >= self.threshold)
        flat_result = detected_df.stack().values.tolist()
        ground_truth = groundtruth_df.stack().values.tolist()
        self.report = pd.DataFrame(
            classification_report(ground_truth, flat_result, output_dict=True, zero_division=0)
        ).transpose()
        self.matrix = pd.DataFrame(
            confusion_matrix(ground_truth, flat_result, labels=[True, False]),
            columns=["True", "False"],
        )
        self.failures, self.scores = self.debug(
            raw_df, cleaned_df, groundtruth_df, detected_df, prob_df
        )

    def debug(self, raw_df, cleaned_df, groundtruth_df, result_df, prob_df):
        def get_prob(x):
            return prob_df.loc[x["id"], x["col"]]
        fn_df = diff_dfs(raw_df, cleaned_df)
        if fn_df is not None:
            fn_df["prediction"] = fn_df.apply(get_prob, axis=1)
            fn_df = fn_df[fn_df["prediction"] >= self.threshold]
            diff_mask = not_equal(groundtruth_df, result_df)
            ne_stacked = diff_mask.stack()
            changed = ne_stacked[ne_stacked]
            changed.index.names = ["id", "col"]
            difference_locations = np.where(diff_mask)
            changed_from = raw_df.values[difference_locations]
            changed_to = cleaned_df.values[difference_locations]
            fp_df = pd.DataFrame(
                {"from": changed_from, "to": changed_to}, index=changed.index
            )
            fp_df["prediction"] = 0.0
            fp_df["id"] = fp_df.index.get_level_values("id")
            fp_df["col"] = fp_df.index.get_level_values("col")
            fp_df["prediction"] = fp_df.apply(get_prob, axis=1)
            fp_df = fp_df[fp_df["prediction"] <= self.threshold]
            concat_df = pd.concat([fp_df, fn_df], ignore_index=True)
        else:
            concat_df = pd.DataFrame(columns=["from", "to", "id", "col", "prediction"])
        score_df = pd.DataFrame()
        score_sr = raw_df.stack()
        score_df["from"] = score_sr.values.tolist()
        score_df["to"] = cleaned_df.stack().values.tolist()
        score_df.index = score_sr.index
        score_df.index.names = ["id", "col"]
        score_df["id"] = score_df.index.get_level_values("id")
        score_df["col"] = score_df.index.get_level_values("col")
        score_df["score"] = prob_df.stack().values.tolist()
        score_df["result"] = groundtruth_df.stack().values.tolist()
        return concat_df, score_df

    def serialize(self, output_path):
        prob_path = output_path / "predictions.csv"
        report_path = output_path / "report.csv"
        debug_path = output_path / "debug.csv"
        matrix_path = output_path / "matrix.csv"
        score_path = output_path / "scores.csv"
        output_path.mkdir(parents=True, exist_ok=True)
        self.prob_df.to_csv(prob_path, index=False)
        self.report["index"] = self.report.index
        self.report.to_csv(report_path, index=False, quoting=csv.QUOTE_ALL)
        self.failures.to_csv(debug_path, index=False, quoting=csv.QUOTE_ALL)
        self.matrix["index"] = pd.Series(["True", "False"])
        self.matrix.to_csv(matrix_path, index=None, quoting=csv.QUOTE_ALL)
        self.scores.to_csv(score_path, index=None, quoting=csv.QUOTE_ALL)
