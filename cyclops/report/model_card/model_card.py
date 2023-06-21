"""Model Card schema."""

from datetime import date as dt_date
from datetime import datetime as dt_datetime
from typing import Any, List, Optional, Union

from pydantic import (  # pylint: disable=no-name-in-module
    BaseModel,
    Extra,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
)

# pylint: disable=too-few-public-methods


class BaseModelCardField(BaseModel):
    """Base class for model card fields."""

    class Config:
        """Global config for model card fields."""

        extra = Extra.allow
        smart_union = True
        validate_all = True
        validate_assignment = True


class Owner(BaseModelCardField):
    """Information about the model/data owner(s)."""

    name: Optional[StrictStr] = Field(
        None, description="The name/organization of the model owner."
    )
    contact: Optional[StrictStr] = Field(
        None, description="The contact information for the model owner(s)."
    )
    role: Optional[StrictStr] = Field(
        None, description="The role of the person e.g. developer, owner, auditor."
    )


class Version(BaseModelCardField):
    """Model or dataset version information."""

    name: Optional[StrictStr] = Field(None, description="The name of the version.")
    date: Optional[Union[dt_date, dt_datetime]] = Field(
        None, description="The date this version was released."
    )
    diff: Optional[StrictStr] = Field(
        None, description="The changes from the previous version."
    )


class License(BaseModelCardField):
    """Model or dataset license information."""

    identifier: Optional[StrictStr] = Field(
        None,
        description="""A standard SPDX license identifier
        (https://spdx.org/licenses/), or\n "proprietary" for an unlicensed module.""",
    )
    custom_text: Optional[StrictStr] = Field(
        None, description="The text of a custom license."
    )


class Reference(BaseModelCardField):
    """Reference to additional resources related to the model or dataset."""

    reference: Optional[StrictStr] = Field(
        None, description="A reference to a resource e.g. paper, repository, demo, etc."
    )


class Citation(BaseModelCardField):
    """Citation information for the model or dataset."""

    style: Optional[StrictStr] = Field(None, description="The citation style.")
    citation: Optional[StrictStr] = Field(
        None, description="The citation content (BibTeX)."
    )


class RegulatoryRequirement(BaseModelCardField):
    """Regulatory requirements for the model or dataset."""

    regulation: Optional[StrictStr] = Field(None, description="Name of the regulation")


class ModelDetails(BaseModelCardField):
    """Details about the model."""

    name: Optional[StrictStr] = Field(None, description="The name of the model.")
    overview: Optional[StrictStr] = Field(
        None, description="A brief, one-line description of the model."
    )
    documentation: Optional[StrictStr] = Field(
        None, description="A more thorough description of the model and its usage."
    )
    owners: Optional[List[Owner]] = Field(
        description="The individuals or teams who own the model.",
        default_factory=list,
        unique_items=True,
    )
    version: Optional[Version] = Field(None, description="The version of the model.")
    licenses: Optional[List[License]] = Field(
        description="The license information for the model.",
        default_factory=list,
        unique_items=True,
    )
    references: Optional[List[Reference]] = Field(
        description="Provide any additional references the reader may need.",
        default_factory=list,
        unique_items=True,
    )
    citations: Optional[List[Citation]] = Field(
        description="How should the model be cited?",
        default_factory=list,
        unique_items=True,
    )
    path: Optional[StrictStr] = Field(None, description="Where is this model stored?")
    regulatory_requirements: Optional[List[RegulatoryRequirement]] = Field(
        description="Provide any regulatory requirements that the model should \
            comply to.",
        default_factory=list,
        unique_items=True,
    )


class Graphic(BaseModelCardField):
    """A graphic to be displayed in the model card."""

    name: Optional[StrictStr] = Field(None, description="The name of the graphic.")
    image: Optional[StrictStr] = Field(
        None, description="The image, encoded as a base64 string."
    )


class GraphicsCollection(BaseModelCardField):
    """A collection of graphics to be displayed in the model card."""

    description: Optional[StrictStr] = Field(
        None, description="A description of the Graphics collection."
    )
    collection: Optional[List[Graphic]] = Field(
        description="A collection of graphics.",
        default_factory=list,
    )


class SensitiveData(BaseModelCardField):
    """Details about sensitive data used in the model."""

    sensitive_data: Optional[List[StrictStr]] = Field(
        description="A description of any sensitive data that may be present in a \
            dataset.\n Be sure to note PII information such as names, addresses, \
            phone numbers,\n etc. Preferably, such info should be scrubbed from \
            a dataset if possible.\n Note that even non-identifying information, \
            such as zip code, age, race,\n and gender, can be used to identify \
            individuals when aggregated. Please\n describe any such fields here.",
        default_factory=list,
        unique_items=True,
    )
    sensitive_data_used: Optional[List[StrictStr]] = Field(
        description="A list of sensitive data used in the deployed model.",
        default_factory=list,
        unique_items=True,
    )
    justification: Optional[StrictStr] = Field(
        None,
        description="Please include a justification of the need to use the fields \
            in deployment.",
    )


class Dataset(BaseModelCardField):
    """Details about the dataset."""

    name: Optional[StrictStr] = Field(None, description="The name of the dataset.")
    split: Optional[StrictStr] = Field(
        None, description="The split of the dataset e.g. train, test, validation."
    )
    size: Optional[StrictInt] = Field(
        None, description="The number of samples in the dataset."
    )
    attributes: Optional[List[StrictStr]] = Field(
        description="The attributes/column names in the dataset.",
        default_factory=list,
    )
    sensitive: Optional[SensitiveData] = Field(
        None, description="Does this dataset contain any human, PII, or sensitive data?"
    )
    graphics: Optional[GraphicsCollection] = Field(
        None, description="Visualizations of the dataset."
    )
    dataset_creation_summary: Optional[StrictStr] = Field(
        None, description="A brief description of how the dataset was created."
    )
    owners: Optional[List[Owner]] = Field(
        description="The individuals or teams who created/contributed to the dataset.",
        default_factory=list,
        unique_items=True,
    )
    version: Optional[Version] = Field(None, description="The version of the dataset.")
    licenses: Optional[List[License]] = Field(
        description="The license information for the dataset.",
        default_factory=list,
        unique_items=True,
    )
    references: Optional[List[Reference]] = Field(
        description="Provide any additional references for the dataset e.g. paper.",
        default_factory=list,
        unique_items=True,
    )
    citations: Optional[List[Citation]] = Field(
        description="How should the dataset be cited?",
        default_factory=list,
        unique_items=True,
    )


class KeyVal(BaseModelCardField):
    """A key-value pair."""

    key: Optional[StrictStr] = None
    value: Any = None


class ModelParameters(BaseModelCardField):
    """Parameters for the model."""

    model_architecture: Optional[StrictStr] = Field(
        None, description="Specifies the architecture of your model."
    )
    data: Optional[List[Dataset]] = Field(
        description="Specifies the datasets used to train and evaluate your model.",
        default_factory=list,
    )
    input_format: Optional[StrictStr] = Field(
        None, description="Describes the data format for inputs to your model."
    )
    input_format_map: Optional[List[KeyVal]] = Field(
        description="A mapping of input format to the data format for inputs to your \
            model.",
        default_factory=list,
    )
    output_format: Optional[StrictStr] = Field(
        None, description="Describes the data format for outputs from your model."
    )
    output_format_map: Optional[List[KeyVal]] = Field(
        description="A mapping of output format to the data format for outputs from \
            your model.",
        default_factory=list,
    )


class Test(BaseModelCardField):
    """Tests for the model."""

    name: Optional[StrictStr] = Field(None, description="The name of the test.")
    description: Optional[StrictStr] = Field(
        None, description="User-friendly description of the test."
    )
    threshold: Optional[StrictFloat] = Field(
        None, description="Threshold required to pass the test."
    )
    result: Optional[Union[StrictFloat, StrictInt]] = Field(
        None, description="Result returned by the test."
    )
    passed: Optional[StrictBool] = Field(
        None, description="Whether the model result satisfies the given threshold."
    )
    graphics: Optional[GraphicsCollection] = Field(
        None, description="A collection of visualizations related to the test."
    )


class PerformanceMetric(BaseModelCardField):
    """Performance metrics for model evaluation."""

    type: Optional[StrictStr] = Field(
        None, description="The type of performance metric."
    )
    value: Optional[Union[StrictFloat, StrictInt]] = Field(
        None, description="The value of the performance metric."
    )
    slice: Optional[StrictStr] = Field(
        None,
        description="The name of the slice this metric was computed on.\n By default, \
            assume this metric is not sliced.",
    )
    description: Optional[StrictStr] = Field(
        None, description="User-friendly description of the performance metric."
    )
    graphics: Optional[GraphicsCollection] = Field(
        None,
        description="A collection of visualizations related to the metric and slice.",
    )
    tests: Optional[List[Test]] = Field(
        description="A collection of tests associated with the metric.",
        default_factory=list,
    )


class QuantitativeAnalysis(BaseModelCardField):
    """Quantitative analysis of the model."""

    performance_metrics: Optional[List[PerformanceMetric]] = Field(
        description="The performance metrics being reported.",
        default_factory=list,
    )
    graphics: Optional[GraphicsCollection] = Field(
        None,
        description="A collection of visualizations of model performance.\n Retain \
            for backward compatibility with model scorecard.\n Prefer to use \
            GraphicsCollection within PerformanceMetric.",
    )


class User(BaseModelCardField):
    """Details about the user."""

    description: Optional[StrictStr] = Field(
        None, description="A description of a user."
    )


class UseCase(BaseModelCardField):
    """Details about the use case."""

    description: Optional[StrictStr] = Field(
        None, description="A description of a use case."
    )


class Limitation(BaseModelCardField):
    """Details about the limitations of the model."""

    description: Optional[StrictStr] = Field(
        None, description="A description of the limitation."
    )


class Tradeoff(BaseModelCardField):
    """A description of the tradeoffs of the model."""

    description: Optional[StrictStr] = Field(
        None, description="A description of the tradeoff."
    )


class Risk(BaseModelCardField):
    """A description of the risks posed by the model."""

    name: Optional[StrictStr] = Field(None, description="The name of the risk.")
    mitigation_strategy: Optional[StrictStr] = Field(
        None,
        description="A mitigation strategy that you've implemented, or one you suggest \
            to users.",
    )


class FairnessAssessment(BaseModelCardField):
    """Details on the fairness assessment of the model."""

    group_at_risk: Optional[StrictStr] = Field(
        None,
        description="The groups or individuals at risk of being systematically \
            disadvantaged by the model.",
    )
    benefits: Optional[StrictStr] = Field(
        None, description="Expected benefits to the identified groups."
    )
    harms: Optional[StrictStr] = Field(
        None, description="Expected harms to the identified groups."
    )
    mitigation_strategy: Optional[StrictStr] = Field(
        None,
        description="With respect to the benefits and harms outlined, please describe \
            any mitigation strategy implemented.",
    )


class Considerations(BaseModelCardField):
    """Considerations for the model."""

    users: Optional[List[User]] = Field(
        description="Who are the intended users of the model?",
        default_factory=list,
        unique_items=True,
    )
    use_cases: Optional[List[UseCase]] = Field(
        description="What are the intended use cases of the model?",
        default_factory=list,
        unique_items=True,
    )
    limitations: Optional[List[Limitation]] = Field(
        description="What are the known limitations of the model?",
        default_factory=list,
        unique_items=True,
    )
    tradeoffs: Optional[List[Tradeoff]] = Field(
        description="What are the known accuracy/performance tradeoffs for the model?",
        default_factory=list,
        unique_items=True,
    )
    ethical_considerations: Optional[List[Risk]] = Field(
        description="What are the ethical risks involved in application of this model?",
        default_factory=list,
        unique_items=True,
    )
    fairness_assessment: Optional[List[FairnessAssessment]] = Field(
        description="How does the model affect groups at risk of being systematically \
            disadvantaged?\n What are the harms and benefits to the various affected \
            groups?",
        default_factory=list,
    )


class ExplainabilityReport(BaseModelCardField):
    """Explainability reports for the model."""

    type: Optional[StrictStr] = Field(
        None, description="The type of explainability method."
    )
    slice: Optional[StrictStr] = Field(
        None,
        description="The name of the slice the explainability method was computed \
            on.\n By default, assume this metric is not sliced.",
    )
    description: Optional[StrictStr] = Field(
        None, description="User-friendly description of the explainability method."
    )
    graphics: Optional[GraphicsCollection] = Field(
        None,
        description="A collection of visualizations related to the explainability \
            method.",
    )
    tests: Optional[List[Test]] = Field(
        description="A collection of tests associated with the explainability method.",
        default_factory=list,
    )


class ExplainabilityAnalysis(BaseModelCardField):
    """Explainability analysis of the model."""

    explainability_reports: Optional[List[ExplainabilityReport]] = Field(
        description="Model explainability report e.g. feature importance, decision \
            trees etc.",
        default_factory=list,
    )


class FairnessReport(BaseModelCardField):
    """Fairness reports for the model."""

    type: Optional[StrictStr] = Field(
        None, description="The type of fairness study conducted."
    )
    slice: Optional[StrictStr] = Field(
        None,
        description="The name of the slice the fairness report was computed on.\
            \n By default, assume this metric is not sliced.",
    )
    segment: Optional[StrictStr] = Field(
        None,
        description="Segment of dataset which the fairness report is meant to assess.\
            \n e.g. age, gender, age and gender, etc.",
    )
    description: Optional[StrictStr] = Field(
        None, description="User-friendly description of the fairness method."
    )
    graphics: Optional[GraphicsCollection] = Field(
        None,
        description="A collection of visualizations related to the fairness method.",
    )
    tests: Optional[List[Test]] = Field(
        description="Tests related to fairness considerations.", default_factory=list
    )


class FairnessAnalysis(BaseModelCardField):
    """Fairness analysis of the model."""

    fairness_reports: Optional[List[FairnessReport]] = Field(
        description="Fairness report to evaluate the model performance on various \
            groups.",
        default_factory=list,
    )


class ModelCard(BaseModelCardField):
    """Model Card for reporting information about a model.

    Schema adapted from: https://github.com/cylynx/verifyml Based on:
    https://arxiv.org/abs/1810.03993.

    """

    model_details: Optional[ModelDetails] = Field(
        None, description="Descriptive metadata for the model."
    )
    model_parameters: Optional[ModelParameters] = Field(
        None, description="Technical metadata for the model."
    )
    considerations: Optional[Considerations] = Field(
        None,
        description="Any considerations related to model construction, training, and \
            application",
    )
    quantitative_analysis: Optional[QuantitativeAnalysis] = Field(
        None, description="Quantitative analysis of model performance."
    )
    explainability_analysis: Optional[ExplainabilityAnalysis] = Field(
        None, description="Explainability analysis being reported."
    )
    fairness_analysis: Optional[FairnessAnalysis] = Field(
        None, description="Fairness analysis being reported."
    )
