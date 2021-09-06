from pathlib import Path
from evaluation.report import Report
from loguru import logger


class Evaluator:
    def evaluate(self, detector, dataset, training, output_path):
        logger.info(f"Evaluating ...")
        prob_df = detector.detect(dataset, training)
        report = Report(prob_df, dataset)
        logger.info("Report:\n" + str(report.report))
        report.serialize(Path(output_path + '/' + dataset['name']))
        return report.report

    def evaluate_csv(self, prob_df, dataset, output_path):
        logger.info(f"Evaluating CSV...")
        report = Report(prob_df, dataset)
        logger.info("Report:\n" + str(report.report))
        report.serialize(Path(output_path + '/' + dataset['name']))
        return report.report