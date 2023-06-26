"""Model Card schema."""
import inspect
from datetime import date as dt_date
from datetime import datetime as dt_datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from license_expression import ExpressionError, get_license_index, get_spdx_licensing
from pybtex import PybtexEngine
from pybtex.exceptions import PybtexError
from pydantic import (
    AnyUrl,
    BaseConfig,
    BaseModel,
    Extra,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    root_validator,
    validator,
)

# pylint: disable=too-few-public-methods


class BaseModelCardField(BaseModel):
    """Base class for model card fields."""

    class Config(BaseConfig):
        """Global config for model card fields."""

        extra: Extra = Extra.allow
        smart_union: bool = True
        validate_all: bool = True
        validate_assignment: bool = True
        allowable_sections: List[str] = []  # sections this field is allowed in
        list_factory: bool = False  # whether to use a list factory for this field
        json_encoders = {np.ndarray: lambda v: v.tolist()}


class Owner(
    BaseModelCardField,
    allowable_sections=["model_details", "datasets"],
    list_factory=True,
):
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


class Version(
    BaseModelCardField,
    allowable_sections=["model_details", "datasets"],
    list_factory=False,
):
    """Model or dataset version information."""

    version_str: Optional[StrictStr] = Field(
        None, description="The version string of the model."
    )
    date: Optional[Union[dt_date, dt_datetime]] = Field(
        None, description="The date this version was released."
    )
    description: Optional[StrictStr] = Field(
        None, description="A description of the version, e.g. what changed?"
    )


class License(
    BaseModelCardField,
    allowable_sections=["model_details", "datasets"],
    list_factory=True,
):
    """Model or dataset license information."""

    identifier: Optional[StrictStr] = Field(
        None,
        description=(
            "A standard SPDX license identifier (https://spdx.org/licenses/). "
            "Use one of the following values for special cases: 'proprietary', "
            "'unlicensed', 'unknown'."
        ),
    )
    text: Optional[StrictStr] = Field(
        None,
        description="The license text, which be used to provide a custom license.",
    )
    text_url: Optional[AnyUrl] = Field(None, description="A URL to the license text.")

    @root_validator(skip_on_failure=True)
    def validate_spdx_identifier(  # pylint: disable=no-self-argument
        cls: "License", values: Dict[str, StrictStr]
    ) -> Dict[str, Union[StrictStr, AnyUrl]]:
        """Validate the SPDX license identifier."""
        spdx_id = values["identifier"]
        try:
            get_spdx_licensing().parse(spdx_id, validate=True)
            if spdx_id not in [None, ""] and values.get("text_url") is None:
                values["text_url"] = cls._get_license_text_url(spdx_id)  # type: ignore
        except ExpressionError as exc:
            if spdx_id.lower() not in ["proprietary", "unlicensed", "unknown"]:
                raise ValueError(
                    "Expected a valid SPDX license identifier "
                    f"(https://spdx.org/licenses/). Got {spdx_id} instead."
                ) from exc
        return values

    @staticmethod
    def _get_license_text_url(identifier: Optional[str]) -> Optional[str]:
        """Get the license text for a given SPDX identifier."""
        if identifier in ["proprietary", "unlicensed", "unknown", None]:
            return None

        identifier = str(identifier).lower()

        # get the license index
        license_index = get_license_index()
        license_index_dict = {
            item["spdx_license_key"].lower(): item
            for item in license_index
            if item.get("spdx_license_key")
        }

        text_urls = license_index_dict[identifier].get("text_urls")
        if text_urls is not None and isinstance(text_urls, list) and len(text_urls) > 0:
            return text_urls[0]  # type: ignore[no-any-return]

        return None


class Reference(BaseModelCardField, list_factory=True):
    """Reference to additional resources related to the model or dataset."""

    link: Optional[AnyUrl] = Field(None, description="A URL to the reference resource.")


class Citation(
    BaseModelCardField,
    allowable_sections=["model_details", "datasets"],
    list_factory=True,
):
    """Citation information for the model or dataset."""

    content: Optional[StrictStr] = Field(
        None, description="The citation content e.g. BibTeX, APA, etc."
    )

    @validator("content")
    def parse_content(  # pylint: disable=no-self-argument
        cls: "Citation", value: StrictStr
    ) -> StrictStr:
        """Parse the citation content."""
        try:
            formatted_citation: StrictStr = PybtexEngine().format_from_string(
                value, style="unsrt", output_backend="text"
            )
            if formatted_citation != "":
                value = formatted_citation
        except PybtexError as exc:
            raise ValueError(f"Could not parse citation: {exc}") from exc
        return value


class RegulatoryRequirement(
    BaseModelCardField, allowable_sections=["model_details"], list_factory=True
):
    """Regulatory requirements for the model or dataset."""

    regulation: Optional[StrictStr] = Field(None, description="Name of the regulation")


class ModelDetails(BaseModelCardField):
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


class Graphic(BaseModelCardField, list_factory=True):
    """A graphic to be displayed in the model card."""

    name: Optional[StrictStr] = Field(None, description="The name of the graphic.")
    image: Optional[StrictStr] = Field(
        None, description="The image, encoded as a base64 string or an html string."
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


class SensitiveData(
    BaseModelCardField, allowable_sections=["datasets"], list_factory=True
):
    """Details about sensitive data used in the model."""

    sensitive_data: Optional[List[StrictStr]] = Field(
        description=(
            "A description of any sensitive data that may be present in a dataset. "
            "Be sure to note PII information such as names, addresses, phone numbers, "
            "etc. Preferably, such info should be scrubbed from a dataset if "
            "possible. Note that even non-identifying information, such as zip code, "
            "age, race, and gender, can be used to identify individuals when "
            "aggregated. Please describe any such fields here."
        ),
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
        description=inspect.cleandoc(
            """
            Please include a justification of the need to use the fields in deployment.
            """
        ),
    )


class Dataset(
    BaseModelCardField, allowable_sections=["model_parameters"], list_factory=True
):
    """Details about the dataset."""

    description: Optional[StrictStr] = Field(
        None, description="A high-level description of the dataset."
    )
    citations: Optional[List[Citation]] = Field(
        description="How should the dataset be cited?",
        default_factory=list,
        unique_items=True,
    )
    references: Optional[List[Reference]] = Field(
        description="Provide any additional links to resources the reader may need.",
        default_factory=list,
        unique_items=True,
    )
    licenses: Optional[List[License]] = Field(
        description="The license information for the dataset.",
        default_factory=list,
        unique_items=True,
    )
    version: Optional[Version] = Field(None, description="The version of the dataset.")
    features: Optional[List[StrictStr]] = Field(
        description="A list of features in the dataset.",
        default_factory=list,
        unique_items=True,
    )
    graphics: Optional[GraphicsCollection] = Field(
        None, description="Visualizations of the dataset."
    )
    split: Optional[StrictStr] = Field(
        None, description="The split of the dataset e.g. train, test, validation."
    )
    size: Optional[StrictInt] = Field(
        None, description="The number of samples in the dataset."
    )
    sensitive: Optional[SensitiveData] = Field(
        None, description="Does this dataset contain any human, PII, or sensitive data?"
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
    input_format: Optional[StrictStr] = Field(
        None, description="Describes the data format for inputs to your model."
    )
    input_format_map: Optional[List[KeyVal]] = Field(
        description=inspect.cleandoc(
            """
            A mapping of input format to the data format for inputs to your model.
            """
        ),
        default_factory=list,
    )
    output_format: Optional[StrictStr] = Field(
        None, description="Describes the data format for outputs from your model."
    )
    output_format_map: Optional[List[KeyVal]] = Field(
        description=inspect.cleandoc(
            """
            A mapping of output format to the data format for outputs from your model.
            """
        ),
        default_factory=list,
    )
    data: Optional[List[Dataset]] = Field(
        description="Specifies the datasets used to train and evaluate your model.",
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


class PerformanceMetric(
    BaseModelCardField,
    allowable_sections=["quantitative_analysis"],
    list_factory=True,
    arbitrary_types_allowed=True,
):
    """Performance metrics for model evaluation."""

    type: Optional[StrictStr] = Field(
        None, description="The type of performance metric."
    )
    value: Optional[
        Union[
            StrictFloat,
            StrictInt,
            npt.NDArray[Union[np.int_, np.float_]],
            List[Union[StrictFloat, StrictInt]],
            Tuple[
                Union[
                    StrictFloat,
                    StrictInt,
                    npt.NDArray[Union[np.int_, np.float_]],
                    List[Union[StrictFloat, StrictInt]],
                ],
                ...,
            ],
        ]
    ] = Field(None, description="The value of the performance metric.")
    slice: Optional[StrictStr] = Field(
        None,
        description=inspect.cleandoc(
            """
            The name of the slice this metric was computed on. By default, assume that
            this metric is not sliced.""",
        ),
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


class User(
    BaseModelCardField,
    allowable_sections=["considerations", "dataset"],
    list_factory=True,
):
    """Details about the user."""

    description: Optional[StrictStr] = Field(
        None, description="A description of a user."
    )


class UseCase(
    BaseModelCardField,
    allowable_sections=["considerations", "dataset"],
    list_factory=True,
):
    """Details about the use case."""

    description: Optional[StrictStr] = Field(
        None, description="A description of a use case."
    )
    kind: Optional[Literal["primary", "downstream", "out-of-scope"]] = Field(
        None,
        description=inspect.cleandoc(
            """
            The scope of the use case. Must be one of 'primary', 'downstream', or
            'out-of-scope'."""
        ),
    )

    @validator("kind")
    def kind_must_be_valid(  # pylint: disable=no-self-argument
        cls: "UseCase", value: str
    ) -> str:
        """Validate the use case kind."""
        if isinstance(value, str):
            value = value.lower()

        if value not in ["primary", "downstream", "out-of-scope"]:
            raise ValueError(
                "Use case kind must be one of 'primary', 'downstream', or "
                "'out-of-scope'."
            )
        return value


class Risk(
    BaseModelCardField,
    allowable_sections=["considerations", "dataset"],
    list_factory=True,
):
    """A description of the risks posed by the model."""

    name: Optional[StrictStr] = Field(None, description="The name of the risk.")
    mitigation_strategy: Optional[StrictStr] = Field(
        None,
        description=(
            "A mitigation strategy that you've implemented, or one you suggest to "
            "users."
        ),
    )


class FairnessAssessment(
    BaseModelCardField,
    allowable_sections=["considerations", "dataset"],
    list_factory=True,
):
    """Details on the fairness assessment of the model."""

    affected_group: Optional[StrictStr] = Field(
        None,
        description=(
            "The groups or individuals at risk of being systematically disadvantaged "
            "by the model."
        ),
    )
    benefits: Optional[StrictStr] = Field(
        None, description="Expected benefits to the identified groups."
    )
    harms: Optional[StrictStr] = Field(
        None, description="Expected harms to the identified groups."
    )
    mitigation_strategy: Optional[StrictStr] = Field(
        None,
        description=(
            "With respect to the benefits and harms outlined, please describe any "
            "mitigation strategy implemented."
        ),
    )


class Considerations(BaseModelCardField):
    """Considerations for the model."""

    # uses, risks/social impact, bias + recommendations for mitigation, limitations
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


class ExplainabilityReport(BaseModelCardField):
    """Explainability reports for the model."""

    type: Optional[StrictStr] = Field(
        None, description="The type of explainability method."
    )
    slice: Optional[StrictStr] = Field(
        None,
        description="""
        The name of the slice the explainability method was computed on.
        By default, assume this metric is not sliced.
        """,
    )
    description: Optional[StrictStr] = Field(
        None, description="User-friendly description of the explainability method."
    )
    graphics: Optional[GraphicsCollection] = Field(
        None,
        description=(
            "A collection of visualizations related to the explainability method."
        ),
    )
    tests: Optional[List[Test]] = Field(
        description="A collection of tests associated with the explainability method.",
        default_factory=list,
    )


class ExplainabilityAnalysis(BaseModelCardField):
    """Explainability analysis of the model."""

    explainability_reports: Optional[List[ExplainabilityReport]] = Field(
        description=(
            "Model explainability report e.g. feature importance, decision trees etc."
        ),
        default_factory=list,
    )


class FairnessReport(BaseModelCardField):
    """Fairness reports for the model."""

    type: Optional[StrictStr] = Field(
        None, description="The type of fairness study conducted."
    )
    slice: Optional[StrictStr] = Field(
        None,
        description="""
        The name of the slice the fairness report was computed on.
        By default, assume this metric is not sliced.
        """,
    )
    segment: Optional[StrictStr] = Field(
        None,
        description=inspect.cleandoc(
            """
            Segment(s) of dataset which the fairness report is meant to assess e.g.
            age, gender, age and gender, etc.""",
        ),
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
        description=(
            "Fairness report to evaluate the model performance on various groups."
        ),
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
        description=inspect.cleandoc(
            """
            Any considerations related to model construction, training, and
             application""",
        ),
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
