"""Test cyclops report module model report."""

from unittest import TestCase

import numpy as np

from cyclops.report import ModelCardReport
from cyclops.report.model_card.sections import ModelDetails


class TestModelCardReport(TestCase):
    """Test ModelCardReport."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_card_report = ModelCardReport("reports")

    def test_instantiation_with_optional_output_dir(self):
        """Test instantiation with optional output_dir."""
        assert self.model_card_report.output_dir == "reports"

    def test_log_owner_with_name(self):
        """Test log_owner with name."""
        self.model_card_report.log_owner(name="John Doe")
        assert (
            self.model_card_report._model_card.model_details.owners[0].name
            == "John Doe"
        )

    def test_log_owner_with_name_and_contact(self):
        """Test log_owner with name and contact."""
        self.model_card_report.log_owner(
            name="John Doe",
            contact="john.doe@example.com",
        )
        assert (
            self.model_card_report._model_card.model_details.owners[0].name
            == "John Doe"
        )
        assert (
            self.model_card_report._model_card.model_details.owners[0].contact
            == "john.doe@example.com"
        )

    def test_log_owner_with_name_and_role(self):
        """Test log_owner with name and role."""
        self.model_card_report.log_owner(name="John Doe", role="Developer")
        assert (
            self.model_card_report._model_card.model_details.owners[0].name
            == "John Doe"
        )
        assert (
            self.model_card_report._model_card.model_details.owners[0].role
            == "Developer"
        )

    def test_valid_name_and_description(self):
        """Test valid name and description."""
        self.model_card_report.log_descriptor(
            name="ethical_considerations",
            description="This model was trained on data collected from a potentially biased source.",
            section_name="considerations",
        )

        section = self.model_card_report._model_card.get_section("considerations")
        descriptor = section.ethical_considerations

        assert (
            descriptor[0].description
            == "This model was trained on data collected from a potentially biased source."
        )

    def test_log_user_with_description_to_considerations_section(self):
        """Test log_user with description to considerations section."""
        self.model_card_report.log_user(description="This is a user description")
        assert len(self.model_card_report._model_card.considerations.users) == 1
        assert (
            self.model_card_report._model_card.considerations.users[0].description
            == "This is a user description"
        )

    def test_log_performance_metric(self):
        """Test log_performance_metric."""
        self.model_card_report.log_quantitative_analysis(
            analysis_type="performance",
            name="accuracy",
            value=0.85,
            metric_slice="test",
            decision_threshold=0.8,
            description="Accuracy of the model on the test set",
            pass_fail_thresholds=[0.9, 0.85, 0.8],
            pass_fail_threshold_fns=[lambda x, t: x >= t for _ in range(3)],
        )
        assert (
            self.model_card_report._model_card.quantitative_analysis.performance_metrics[
                0
            ].type
            == "accuracy"
        )
        assert (
            self.model_card_report._model_card.quantitative_analysis.performance_metrics[
                0
            ].value
            == 0.85
        )
        assert (
            self.model_card_report._model_card.quantitative_analysis.performance_metrics[
                0
            ].slice
            == "test"
        )
        assert (
            self.model_card_report._model_card.quantitative_analysis.performance_metrics[
                0
            ].decision_threshold
            == 0.8
        )
        assert (
            self.model_card_report._model_card.quantitative_analysis.performance_metrics[
                0
            ].description
            == "Accuracy of the model on the test set"
        )
        assert (
            len(
                self.model_card_report._model_card.quantitative_analysis.performance_metrics[
                    0
                ].tests,
            )
            == 3
        )

    def test_log_quantitative_analysis_performance(self):
        """Test log_quantitative_analysis (performance)."""
        self.model_card_report.log_quantitative_analysis(
            analysis_type="performance",
            name="accuracy",
            value=0.85,
        )
        assert (
            self.model_card_report._model_card.quantitative_analysis.performance_metrics[
                0
            ].type
            == "accuracy"
        )
        assert (
            self.model_card_report._model_card.quantitative_analysis.performance_metrics[
                0
            ].value
            == 0.85
        )

    def test_log_quantitative_analysis_fairness(self):
        """Test log_quantitative_analysis (fairness)."""
        self.model_card_report.log_quantitative_analysis(
            analysis_type="fairness",
            name="disparate_impact",
            value=0.9,
        )
        assert (
            self.model_card_report._model_card.fairness_analysis.fairness_reports[
                0
            ].type
            == "disparate_impact"
        )
        assert (
            self.model_card_report._model_card.fairness_analysis.fairness_reports[
                0
            ].value
            == 0.9
        )

    def test_log_from_dict(self):
        """Test log_from_dict."""
        data = {
            "datasets": "mnist",
            "Description": "dataset of digits from 0 to 9",
            "overview": "Handwritten 28x28 pixel image",
        }
        self.model_card_report.log_from_dict(data, "overview")
        assert (
            self.model_card_report._model_card.overview.overview
            == "Handwritten 28x28 pixel image"
        )

    def test_log_version(self):
        """Test log_version."""
        self.model_card_report._model_card.model_details = ModelDetails()
        self.model_card_report.log_version("1.2.0", description="Added new feature")
        assert (
            self.model_card_report._model_card.model_details.version.version == "1.2.0"
        )

        assert (
            self.model_card_report._model_card.model_details.version.description
            == "Added new feature"
        )

    def test_log_license(self):
        """Test adding license to licenses."""
        self.model_card_report.log_license("Apache-2.0")
        assert (
            self.model_card_report._model_card.model_details.licenses[0].identifier
            == "Apache-2.0"
        )

    def test_log_citation(self):
        """Test adding citation to model details."""
        cite = """@misc{vaswani2023attention,
            title={Attention Is All You Need},
            author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
            year={2023},
            eprint={1706.03762},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
            }"""
        self.model_card_report.log_citation(cite)
        assert (
            self.model_card_report._model_card.model_details.citations[0].content
            == cite
        )

    def test_log_reference(self):
        """Test adding reference to model details."""
        ref = (
            "https://vectorinstitute.github.io/cyclops/api/reference/api/evaluator.html"
        )
        self.model_card_report.log_reference(ref)
        assert (
            self.model_card_report._model_card.model_details.references[0].link == ref
        )

    def test_log_regulation(self):
        """Test adding regulations to model details."""
        reg = "sample regulation requirement"
        self.model_card_report.log_regulation(reg)
        assert (
            self.model_card_report._model_card.model_details.regulatory_requirements[
                0
            ].regulation
            == reg
        )

    def test_log_model_param(self):
        """Test logging model parameters."""
        params = {"w_1": [1.0, 0.49], "b_1": [0.32, 0.4]}
        self.model_card_report.log_model_parameters(params)
        assert self.model_card_report._model_card.model_parameters.b_1 == params["b_1"]
        assert self.model_card_report._model_card.model_parameters.w_1 == params["w_1"]

    def test_log_dataset(self):
        """Test logging information about the dataset."""
        descr = "dataset of digits from 0 to 9"
        cite = """@article{deng2012mnist,
            title={The mnist database of handwritten digit images for machine learning research},
            author={Deng, Li},
            journal={IEEE Signal Processing Magazine},
            volume={29},
            number={6},
            pages={141--142},
            year={2012},
            publisher={IEEE}
            }"""
        self.model_card_report.log_dataset(description=descr, citation=cite)
        assert self.model_card_report._model_card.datasets.data[0].description == descr

        assert (
            self.model_card_report._model_card.datasets.data[0].citation.content == cite
        )

    def test_log_use_case(self):
        """Test adding a use case to a section of the report."""
        usecase = "Medical imaging and segmentaion"
        self.model_card_report.log_use_case(usecase, kind="primary")
        assert (
            self.model_card_report._model_card.considerations.use_cases[0].description
            == usecase
        )

    def test_log_risk(self):
        """Test adding a risk to a section of the report."""
        risk = "Ethical Considerations #2"
        mitigation = "Mitigation strategy #2"
        self.model_card_report.log_risk(risk, mitigation)

        assert (
            self.model_card_report._model_card.considerations.ethical_considerations[
                0
            ].risk
            == risk
        )
        assert (
            self.model_card_report._model_card.considerations.ethical_considerations[
                0
            ].mitigation_strategy
            == mitigation
        )

    def test_fairness_assessment(self):
        """Test adding a fairness assessment to a section of the report."""
        affected_group = "Group #3"
        benefit = "Benefit #2"
        harm = "Harm #5"
        mitigation = "Mitigation strategy #2"

        self.model_card_report.log_fairness_assessment(
            affected_group, benefit, harm, mitigation
        )

        assert (
            self.model_card_report._model_card.considerations.fairness_assessment[
                0
            ].affected_group
            == affected_group
        )
        assert (
            self.model_card_report._model_card.considerations.fairness_assessment[
                0
            ].benefits
            == benefit
        )
        assert (
            self.model_card_report._model_card.considerations.fairness_assessment[
                0
            ].harms
            == harm
        )
        assert (
            self.model_card_report._model_card.considerations.fairness_assessment[
                0
            ].mitigation_strategy
            == mitigation
        )

    def test_export(self):
        """Test exporing model card report to html file."""
        affected_group = "Group #3"
        benefit = "Benefit #2"
        harm = "Harm #5"
        mitigation = "Mitigation strategy #2"

        self.model_card_report.log_fairness_assessment(
            affected_group, benefit, harm, mitigation
        )
        self.model_card_report.log_quantitative_analysis(
            analysis_type="performance",
            name="BinaryAccuracy",
            description="Accuracy of the model on the test set",
            value=0.85,
            metric_slice="overall",
            decision_threshold=0.7,
            pass_fail_thresholds=[0.6, 0.65, 0.7],
            pass_fail_threshold_fns=[lambda x, t: x >= t for _ in range(3)],
            sample_size=100,
        )
        self.model_card_report.log_quantitative_analysis(
            analysis_type="performance",
            name="BinaryF1Score",
            value=0.65,
            metric_slice="overall",
            decision_threshold=0.8,
            description="F1 score of the model on the test set",
            pass_fail_thresholds=[0.9, 0.85, 0.8],
            pass_fail_threshold_fns=[lambda x, t: x >= t for _ in range(3)],
            sample_size=100,
        )
        self.model_card_report.log_owner(name="John Doe")

        report_path = self.model_card_report.export(interactive=False, save_json=False)
        assert isinstance(report_path, str)


def test_log_performance_metrics():
    """Test log_performance_metrics."""
    report = ModelCardReport()

    # Mock results
    results = {
        "overall/BinaryAccuracy": np.array(0.85),
        "overall/BinaryPrecision": np.array(0.78),
        "overall/BinaryRecall": np.array(0.92),
        "overall/BinaryF1Score": np.array(0.84),
        "overall/BinaryAUROC": np.array(0.91),
        "overall/BinaryAveragePrecision": np.array(0.88),
        "overall/sample_size": 1000,
        "slice1/BinaryAccuracy": np.array(0.82),
        "slice1/BinaryPrecision": np.array(0.75),
        "slice1/BinaryRecall": np.array(0.89),
        "slice1/BinaryF1Score": np.array(0.81),
        "slice1/BinaryAUROC": np.array(0.88),
        "slice1/BinaryAveragePrecision": np.array(0.85),
        "slice1/sample_size": 500,
    }

    # Mock metric descriptions
    metric_descriptions = {
        "BinaryAccuracy": "The proportion of all instances that are correctly predicted.",
        "BinaryPrecision": "The proportion of predicted positive instances that are correctly predicted.",
        "BinaryRecall": "The proportion of actual positive instances that are correctly predicted.",
        "BinaryF1Score": "The harmonic mean of precision and recall.",
        "BinaryAUROC": "The area under the ROC curve.",
        "BinaryAveragePrecision": "The area under the precision-recall curve.",
    }

    # Test with a single threshold
    report.log_performance_metrics(
        results, metric_descriptions, pass_fail_thresholds=0.8
    )

    # Check if metrics were logged correctly
    assert report._model_card.quantitative_analysis is not None
    assert (
        len(report._model_card.quantitative_analysis.performance_metrics) == 12
    )  # 6 metrics * 2 slices

    # Check a few specific metrics
    metrics = report._model_card.quantitative_analysis.performance_metrics

    overall_accuracy = next(
        m for m in metrics if m.type == "BinaryAccuracy" and m.slice == "overall"
    )
    assert overall_accuracy.value == 0.85
    assert (
        overall_accuracy.description
        == "The proportion of all instances that are correctly predicted."
    )
    assert overall_accuracy.sample_size == 1000
    assert overall_accuracy.tests[0].threshold == 0.8
    assert overall_accuracy.tests[0].passed

    slice1_precision = next(
        m for m in metrics if m.type == "BinaryPrecision" and m.slice == "slice1"
    )
    assert slice1_precision.value == 0.75
    assert (
        slice1_precision.description
        == "The proportion of predicted positive instances that are correctly predicted."
    )
    assert slice1_precision.sample_size == 500
    assert slice1_precision.tests[0].threshold == 0.8
    assert not slice1_precision.tests[0].passed

    # Reset the report
    report = ModelCardReport()

    # Test with per-metric thresholds
    pass_fail_thresholds = {
        "overall/BinaryAccuracy": 0.9,
        "overall/BinaryPrecision": 0.75,
        "slice1/BinaryRecall": 0.85,
    }
    report.log_performance_metrics(
        results, metric_descriptions, pass_fail_thresholds=pass_fail_thresholds
    )

    metrics = report._model_card.quantitative_analysis.performance_metrics

    overall_accuracy = next(
        m for m in metrics if m.type == "BinaryAccuracy" and m.slice == "overall"
    )
    assert overall_accuracy.tests[0].threshold == 0.9
    assert not overall_accuracy.tests[0].passed

    overall_precision = next(
        m for m in metrics if m.type == "BinaryPrecision" and m.slice == "overall"
    )
    assert overall_precision.tests[0].threshold == 0.75
    assert overall_precision.tests[0].passed

    slice1_recall = next(
        m for m in metrics if m.type == "BinaryRecall" and m.slice == "slice1"
    )
    assert slice1_recall.tests[0].threshold == 0.85
    assert slice1_recall.tests[0].passed

    slice1_f1 = next(
        m for m in metrics if m.type == "BinaryF1Score" and m.slice == "slice1"
    )
    assert slice1_f1.tests[0].threshold == 0.7  # Default threshold
    assert slice1_f1.tests[0].passed
