from pathlib import Path
from evaluation.report import Report
from loguru import logger


class Evaluator:
    def evaluate(self, detector, name2raw, name2cleaned, name2groundtruth, data_path, output_path):
        name2report = {}
        for name in name2raw.keys():
            logger.info(f"Evaluating on {name}...")
            prob_df = detector.detect(name2raw[name], name2cleaned[name])
            report = Report(prob_df, name2raw[name], name2cleaned[name], name2groundtruth[name])
            logger.info("Report:\n" + str(report.report))
            report.serialize(Path(output_path + '/' + Path(data_path).name + '/' + Path(name).stem))
            name2report[name] = report.report
        return name2report

    def evaluate_csv(self, prob_df, raw_df, cleaned_df, groundtruth_df, data_path, output_path):
        logger.info(f"Evaluating CSV...")
        report = Report(prob_df, raw_df, cleaned_df, groundtruth_df)
        logger.info("Report:\n" + str(report.report))
        report.serialize(Path(output_path + '/' + Path(data_path).name))
        name2report = report.report
        return name2report