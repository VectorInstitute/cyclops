"""Test cyclops report module model report."""

from unittest import TestCase

from cyclops.report import ModelCardReport


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
