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
