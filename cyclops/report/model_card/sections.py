"""Model Card sections."""

import inspect
from typing import List, Optional

from pydantic import Field, StrictStr

from cyclops.report.model_card.base import BaseModelCardSection
from cyclops.report.model_card.fields import (
    Citation,
    Dataset,
    ExplainabilityReport,
    FairnessAssessment,
    FairnessReport,
    GraphicsCollection,
    KeyVal,
    License,
    Owner,
    PerformanceMetric,
    Reference,
    RegulatoryRequirement,
    Risk,
    UseCase,
    User,
    Version,
)


class ModelDetails(BaseModelCardSection):
    """Details about the model."""

    description: Optional[StrictStr] = Field(
        None,
        description=(
            "A high-level description of the model and its usage for a general "
            "audience."
        ),
    )
    version: Optional[Version] = Field(None, description="The version of the model.")
    owners: Optional[List[Owner]] = Field(
        description="The individuals or teams who own the model.",
        default_factory=list,
        unique_items=True,
    )
    licenses: Optional[List[License]] = Field(
        description="The license information for the model.",
        default_factory=list,
        unique_items=True,
    )
    citations: Optional[List[Citation]] = Field(
        description="How should the model be cited?",
        default_factory=list,
        unique_items=True,
    )
    references: Optional[List[Reference]] = Field(
        description="Provide any additional references the reader may need.",
        default_factory=list,
        unique_items=True,
    )
    path: Optional[StrictStr] = Field(None, description="Where is this model stored?")
    regulatory_requirements: Optional[List[RegulatoryRequirement]] = Field(
        description=(
            "Provide any regulatory requirements that the model should comply to."
        ),
        default_factory=list,
        unique_items=True,
    )


class ModelParameters(BaseModelCardSection):
    """Parameters for the model."""

    model_architecture: Optional[StrictStr] = Field(
        None,
        description="Specifies the architecture of your model.",
    )
    input_format: Optional[StrictStr] = Field(
        None,
        description="Describes the data format for inputs to your model.",
    )
    input_format_map: Optional[List[KeyVal]] = Field(
        description=inspect.cleandoc(
            """
            A mapping of input format to the data format for inputs to your model.
            """,
        ),
        default_factory=list,
    )
    output_format: Optional[StrictStr] = Field(
        None,
        description="Describes the data format for outputs from your model.",
    )
    output_format_map: Optional[List[KeyVal]] = Field(
        description=inspect.cleandoc(
            """
            A mapping of output format to the data format for outputs from your model.
            """,
        ),
        default_factory=list,
    )


class Datasets(BaseModelCardSection):
    """Datasets used to train/validate/evaluate the model."""

    data: Optional[List[Dataset]] = Field(
        description="Specifies the datasets used to train and evaluate your model.",
        default_factory=list,
    )
    graphics: Optional[GraphicsCollection] = Field(
        None,
        description="A collection of graphics related to the datasets.",
    )


class Considerations(BaseModelCardSection):
    """Considerations for the model."""

    # uses, risks/social impact, bias + recommendations for mitigation, limitations
    users: Optional[List[User]] = Field(
        description="Who are the primary intended users of the model?",
        default_factory=list,
        unique_items=True,
    )
    use_cases: Optional[List[UseCase]] = Field(
        description="What are the intended use cases of the model?",
        default_factory=list,
        unique_items=True,
    )
    fairness_assessment: Optional[List[FairnessAssessment]] = Field(
        description="""
        How does the model affect groups at risk of being systematically disadvantaged?
        What are the harms and benefits to the various affected groups?
        """,
        default_factory=list,
    )
    ethical_considerations: Optional[List[Risk]] = Field(
        description="What are the ethical risks involved in application of this model?",
        default_factory=list,
        unique_items=True,
    )


class QuantitativeAnalysis(BaseModelCardSection):
    """Quantitative analysis of the model."""

    performance_metrics: Optional[List[PerformanceMetric]] = Field(
        description="The performance metrics being reported.",
        default_factory=list,
    )
    graphics: Optional[GraphicsCollection] = Field(
        None,
        description="A collection of graphics related to the metrics.",
    )


class ExplainabilityAnalysis(BaseModelCardSection):
    """Explainability analysis of the model."""

    explainability_reports: Optional[List[ExplainabilityReport]] = Field(
        description=(
            "Model explainability report e.g. feature importance, decision trees etc."
        ),
        default_factory=list,
    )
    graphics: Optional[GraphicsCollection] = Field(
        None,
        description="A collection of graphics related the explainability analysis.",
    )


class FairnessAnalysis(BaseModelCardSection):
    """Fairness analysis of the model."""

    fairness_reports: Optional[List[FairnessReport]] = Field(
        description=(
            "Fairness report to evaluate the model performance on various groups."
        ),
        default_factory=list,
    )
    graphics: Optional[GraphicsCollection] = Field(
        None,
        description="A collection of graphics related to the fairness analysis.",
    )
