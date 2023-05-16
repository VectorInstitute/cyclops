from typing import Any, List, Optional, Union
from datetime import date as dt_date, datetime as dt_datetime

from pydantic import BaseModel, Extra, Field


class Owner(BaseModel):
    """Information about the model/data owner(s)."""

    class Config:
        extra = Extra.allow

    name: Optional[str] = Field(
        None, description="The name/organization of the model owner."
    )
    contact: Optional[str] = Field(
        None, description="The contact information for the model owner(s)."
    )
    role: Optional[str] = Field(
        None, description="The role of the person e.g. developer, owner, auditor."
    )


class Version(BaseModel):
    """Model or dataset version information."""

    class Config:
        extra = Extra.allow

    name: Optional[str] = Field(None, description="The name of the version.")
    date: Optional[Union[dt_date, dt_datetime]] = Field(
        None, description="The date this version was released."
    )
    diff: Optional[str] = Field(
        None, description="The changes from the previous version."
    )


class License(BaseModel):
    """Model or dataset license information."""

    class Config:
        extra = Extra.allow

    identifier: Optional[str] = Field(
        None,
        description="""A standard SPDX license identifier
        (https://spdx.org/licenses/), or\n "proprietary" for an unlicensed module.""",
    )
    custom_text: Optional[str] = Field(
        None, description="The text of a custom license."
    )


class Reference(BaseModel):
    """Reference to additional resources related to the model or dataset."""

    class Config:
        extra = Extra.allow

    reference: Optional[str] = Field(
        None, description="A reference to a resource e.g. paper, repository, demo, etc."
    )


class Citation(BaseModel):
    """Citation information for the model or dataset."""

    class Config:
        extra = Extra.allow

    style: Optional[str] = Field(None, description="The citation style.")
    citation: Optional[str] = Field(None, description="The citation content (BibTeX).")


class RegulatoryRequirement(BaseModel):
    """Regulatory requirements for the model or dataset."""

    class Config:
        extra = Extra.allow

    regulation: Optional[str] = Field(None, description="Name of the regulation")


class ModelDetails(BaseModel):
    """Details about the model."""

    class Config:
        extra = Extra.allow

    name: Optional[str] = Field(None, description="The name of the model.")
    overview: Optional[str] = Field(
        None, description="A brief, one-line description of the model."
    )
    documentation: Optional[str] = Field(
        None, description="A more thorough description of the model and its usage."
    )
    owners: Optional[List[Owner]] = Field(
        None, description="The individuals or teams who own the model."
    )
    version: Optional[Version] = Field(None, description="The version of the model.")
    licenses: Optional[List[License]] = Field(
        None, description="The license information for the model."
    )
    references: Optional[List[Reference]] = Field(
        None, description="Provide any additional references the reader may need."
    )
    citations: Optional[List[Citation]] = Field(
        None, description="How should the model be cited?"
    )
    path: Optional[str] = Field(None, description="Where is this model stored?")
    regulatory_requirements: Optional[List[RegulatoryRequirement]] = Field(
        None,
        description="Provide any regulatory requirements that the model should \
            comply to.",
    )


class Graphic(BaseModel):
    """A graphic to be displayed in the model card."""

    class Config:
        extra = Extra.allow

    name: Optional[str] = Field(None, description="The name of the graphic.")
    image: Optional[str] = Field(
        None, description="The image, encoded as a base64 string."
    )


class GraphicsCollection(BaseModel):
    """A collection of graphics to be displayed in the model card."""

    class Config:
        extra = Extra.allow

    description: Optional[str] = Field(
        None, description="A description of the Graphics collection."
    )
    collection: Optional[List[Graphic]] = Field(
        None, description="A collection of graphics."
    )


class SensitiveData(BaseModel):
    """Details about sensitive data used in the model."""

    class Config:
        extra = Extra.allow

    sensitive_data: Optional[List[str]] = Field(
        None,
        description="A description of any sensitive data that may be present in a \
            dataset.\n Be sure to note PII information such as names, addresses, \
            phone numbers,\n etc. Preferably, such info should be scrubbed from \
            a dataset if possible.\n Note that even non-identifying information, \
            such as zip code, age, race,\n and gender, can be used to identify \
            individuals when aggregated. Please\n describe any such fields here.",
    )
    sensitive_data_used: Optional[List[str]] = Field(
        None, description="A list of sensitive data used in the deployed model."
    )
    justification: Optional[str] = Field(
        None,
        description="Please include a justification of the need to use the fields \
            in deployment.",
    )


class Dataset(BaseModel):
    """Details about the dataset."""

    class Config:
        extra = Extra.allow

    name: Optional[str] = Field(None, description="The name of the dataset.")
    split: Optional[str] = Field(
        None, description="The split of the dataset e.g. train, test, validation."
    )
    size: Optional[str] = Field(
        None, description="The number of samples in the dataset."
    )
    attributes: Optional[List[str]] = Field(
        None, description="The attributes/column names in the dataset."
    )
    sensitive: Optional[SensitiveData] = Field(
        None, description="Does this dataset contain any human, PII, or sensitive data?"
    )
    graphics: Optional[GraphicsCollection] = Field(
        None, description="Visualizations of the dataset."
    )
    dataset_creation_summary: Optional[str] = Field(
        None, description="A brief description of how the dataset was created."
    )
    owners: Optional[List[Owner]] = Field(
        None,
        description="The individuals or teams who created/contributed to the dataset.",
    )
    version: Optional[Version] = Field(None, description="The version of the dataset.")
    licenses: Optional[List[License]] = Field(
        None, description="The license information for the dataset."
    )
    references: Optional[List[Reference]] = Field(
        None,
        description="Provide any additional references for the dataset e.g. paper.",
    )
    citations: Optional[List[Citation]] = Field(
        None, description="How should the dataset be cited?"
    )


class KeyVal(BaseModel):
    """A key-value pair."""

    class Config:
        extra = Extra.allow

    key: Optional[str] = None
    value: Any = None


class ModelParameters(BaseModel):
    """Parameters for the model."""

    class Config:
        extra = Extra.allow

    model_architecture: Optional[str] = Field(
        None, description="Specifies the architecture of your model."
    )
    data: Optional[List[Dataset]] = Field(
        None,
        description="Specifies the datasets used to train and evaluate your model.",
    )
    input_format: Optional[str] = Field(
        None, description="Describes the data format for inputs to your model."
    )
    input_format_map: Optional[List[KeyVal]] = None
    output_format: Optional[str] = Field(
        None, description="Describes the data format for outputs from your model."
    )
    output_format_map: Optional[List[KeyVal]] = None


class Test(BaseModel):
    """Tests for the model."""

    class Config:
        extra = Extra.allow

    name: Optional[str] = Field(None, description="The name of the test.")
    description: Optional[str] = Field(
        None, description="User-friendly description of the test."
    )
    threshold: Optional[float] = Field(
        None, description="Threshold required to pass the test."
    )
    result: Optional[float] = Field(None, description="Result returned by the test.")
    passed: Optional[bool] = Field(
        None, description="Whether the model result satisfies the given threshold."
    )
    graphics: Optional[GraphicsCollection] = Field(
        None, description="A collection of visualizations related to the test."
    )


class PerformanceMetric(BaseModel):
    """Performance metrics for model evaluation."""

    class Config:
        extra = Extra.allow

    type: Optional[str] = Field(None, description="The type of performance metric.")
    value: Optional[float] = Field(
        None, description="The value of the performance metric."
    )
    slice: Optional[str] = Field(
        None,
        description="The name of the slice this metric was computed on.\n By default, \
            assume this metric is not sliced.",
    )
    description: Optional[str] = Field(
        None, description="User-friendly description of the performance metric."
    )
    graphics: Optional[GraphicsCollection] = Field(
        None,
        description="A collection of visualizations related to the metric and slice.",
    )
    tests: Optional[List[Test]] = Field(
        None, description="A collection of tests associated with the metric."
    )


class QuantitativeAnalysis(BaseModel):
    """Quantitative analysis of the model."""

    class Config:
        extra = Extra.allow

    performance_metrics: Optional[List[PerformanceMetric]] = Field(
        None, description="The performance metrics being reported."
    )
    graphics: Optional[GraphicsCollection] = Field(
        None,
        description="A collection of visualizations of model performance.\n Retain \
            for backward compatibility with model scorecard.\n Prefer to use \
            GraphicsCollection within PerformanceMetric.",
    )


class User(BaseModel):
    """Details about the user."""

    class Config:
        extra = Extra.allow

    description: Optional[str] = Field(None, description="A description of a user.")


class UseCase(BaseModel):
    """Details about the use case."""

    class Config:
        extra = Extra.allow

    description: Optional[str] = Field(None, description="A description of a use case.")


class Limitation(BaseModel):
    """Details about the limitations of the model."""

    class Config:
        extra = Extra.allow

    description: Optional[str] = Field(
        None, description="A description of the limitation."
    )


class Tradeoff(BaseModel):
    """A description of the tradeoffs of the model."""

    class Config:
        extra = Extra.allow

    description: Optional[str] = Field(
        None, description="A description of the tradeoff."
    )


class Risk(BaseModel):
    """A description of the risks posed by the model."""

    class Config:
        extra = Extra.allow

    name: Optional[str] = Field(None, description="The name of the risk.")
    mitigation_strategy: Optional[str] = Field(
        None,
        description="A mitigation strategy that you've implemented, or one you suggest \
            to users.",
    )


class FairnessAssessment(BaseModel):
    """Details on the fairness assessment of the model."""

    class Config:
        extra = Extra.allow

    group_at_risk: Optional[str] = Field(
        None,
        description="The groups or individuals at risk of being systematically \
            disadvantaged by the model.",
    )
    benefits: Optional[str] = Field(
        None, description="Expected benefits to the identified groups."
    )
    harms: Optional[str] = Field(
        None, description="Expected harms to the identified groups."
    )
    mitigation_strategy: Optional[str] = Field(
        None,
        description="With respect to the benefits and harms outlined, please describe \
            any mitigation strategy implemented.",
    )


class Considerations(BaseModel):
    """Considerations for the model."""

    class Config:
        extra = Extra.allow

    users: Optional[List[User]] = Field(
        None, description="Who are the intended users of the model?"
    )
    use_cases: Optional[List[UseCase]] = Field(
        None, description="What are the intended use cases of the model?"
    )
    limitations: Optional[List[Limitation]] = Field(
        None, description="What are the known limitations of the model?"
    )
    tradeoffs: Optional[List[Tradeoff]] = Field(
        None,
        description="What are the known accuracy/performance tradeoffs for the model?",
    )
    ethical_considerations: Optional[List[Risk]] = Field(
        None,
        description="What are the ethical risks involved in application of this model?",
    )
    fairness_assessment: Optional[List[FairnessAssessment]] = Field(
        None,
        description="How does the model affect groups at risk of being systematically \
            disadvantaged?\n What are the harms and benefits to the various affected \
            groups?",
    )


class ExplainabilityReport(BaseModel):
    """Explainability reports for the model."""

    class Config:
        extra = Extra.allow

    type: Optional[str] = Field(None, description="The type of explainability method.")
    slice: Optional[str] = Field(
        None,
        description="The name of the slice the explainability method was computed \
            on.\n By default, assume this metric is not sliced.",
    )
    description: Optional[str] = Field(
        None, description="User-friendly description of the explainability method."
    )
    graphics: Optional[GraphicsCollection] = Field(
        None,
        description="A collection of visualizations related to the explainability \
            method.",
    )
    tests: Optional[List[Test]] = Field(
        None,
        description="A collection of tests associated with the explainability method.",
    )


class ExplainabilityAnalysis(BaseModel):
    """Explainability analysis of the model."""

    class Config:
        extra = Extra.allow

    explainability_reports: Optional[List[ExplainabilityReport]] = Field(
        None,
        description="Model explainability report e.g. feature importance, decision \
            trees etc.",
    )


class FairnessReport(BaseModel):
    """Fairness reports for the model."""

    class Config:
        extra = Extra.allow

    type: Optional[str] = Field(
        None, description="The type of fairness study conducted."
    )
    slice: Optional[str] = Field(
        None,
        description="The name of the slice the fairness report was computed on.\
            \n By default, assume this metric is not sliced.",
    )
    segment: Optional[str] = Field(
        None,
        description="Segment of dataset which the fairness report is meant to assess.\
            \n e.g. age, gender, age and gender, etc.",
    )
    description: Optional[str] = Field(
        None, description="User-friendly description of the fairness method."
    )
    graphics: Optional[GraphicsCollection] = Field(
        None,
        description="A collection of visualizations related to the fairness method.",
    )
    tests: Optional[List[Test]] = Field(
        None, description="Tests related to fairness considerations."
    )


class FairnessAnalysis(BaseModel):
    """Fairness analysis of the model."""

    class Config:
        extra = Extra.allow

    fairness_reports: Optional[List[FairnessReport]] = Field(
        None,
        description="Fairness report to evaluate the model performance on various \
            groups.",
    )


class ModelCard(BaseModel):
    """Model Card for reporting information about a model.

    Schema adapted from: https://github.com/cylynx/verifyml
    Based on: https://arxiv.org/abs/1810.03993.
    """

    class Config:
        extra = Extra.allow

    model_details: Optional[ModelDetails] = Field(
        None, description="Descriptive metadata for the model."
    )
    model_parameters: Optional[ModelParameters] = Field(
        None, description="Technical metadata for the model."
    )
    quantitative_analysis: Optional[QuantitativeAnalysis] = Field(
        None, description="Quantitative analysis of model performance."
    )
    considerations: Optional[Considerations] = Field(
        None,
        description="Any considerations related to model construction, training, and \
            application",
    )
    explainability_analysis: Optional[ExplainabilityAnalysis] = Field(
        None, description="Explainability analysis being reported."
    )
    fairness_analysis: Optional[FairnessAnalysis] = Field(
        None, description="Fairness analysis being reported."
    )
