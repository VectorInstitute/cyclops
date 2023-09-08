"""Model Card."""

import inspect
from typing import Optional

from pydantic import BaseModel, Extra, Field

from cyclops.report.model_card.base import BaseModelCardConfig, BaseModelCardSection
from cyclops.report.model_card.sections import (
    Considerations,
    Datasets,
    ExplainabilityAnalysis,
    FairnessAnalysis,
    ModelDetails,
    ModelParameters,
    QuantitativeAnalysis,
)


class ModelCard(BaseModel):
    """Model Card for reporting information about a model.

    Schema adapted from: https://github.com/cylynx/verifyml Based on:
    https://arxiv.org/abs/1810.03993.

    """

    class Config(BaseModelCardConfig):
        """Model Card configuration."""

        extra: Extra = Extra.forbid

    model_details: Optional[ModelDetails] = Field(
        None,
        description="Descriptive metadata for the model.",
    )
    model_parameters: Optional[ModelParameters] = Field(
        None,
        description="Technical metadata for the model.",
    )
    datasets: Optional[Datasets] = Field(
        None,
        description="Information about the datasets used to train, validate \
        and/or test the model.",
    )
    considerations: Optional[Considerations] = Field(
        None,
        description=inspect.cleandoc(
            """
            Any considerations related to model construction, training, and
             application""",
        ),
    )
    quantitative_analysis: Optional[QuantitativeAnalysis] = Field(
        None,
        description="Quantitative analysis of model performance.",
    )
    explainability_analysis: Optional[ExplainabilityAnalysis] = Field(
        None,
        description="Explainability analysis being reported.",
    )
    fairness_analysis: Optional[FairnessAnalysis] = Field(
        None,
        description="Fairness analysis being reported.",
    )

    def get_section(self, section_name: str) -> BaseModelCardSection:
        """Retrieve a section from the model card.

        Raises
        ------
        ValueError
            If the given `section_name` is not in the model card.
        TypeError
            If the given `section_name` is not a subclass of `BaseModel`.

        """
        sections = self.__fields__
        if section_name not in sections:
            raise ValueError(
                f"Section `{section_name}` not found in model card. "
                f"Available sections are: {list(sections.keys())}",
            )

        section: Optional[BaseModelCardSection] = getattr(self, section_name)
        if section is None:
            section = sections[section_name].type_()
            setattr(self, section_name, section)

        if not issubclass(section.__class__, BaseModel):
            raise TypeError(
                f"Expected section `{section_name}` to be a subclass of `BaseModel`."
                f" Got {section.__class__} instead.",
            )

        return section
