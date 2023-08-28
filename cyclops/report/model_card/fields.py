"""Model card field definitions."""

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
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    root_validator,
    validator,
)

from cyclops.report.model_card.base import BaseModelCardField


# ruff: noqa: A003


class Owner(
    BaseModelCardField,
    composable_with=["ModelDetails", "Dataset"],
    list_factory=True,
):
    """Information about the model/data owner(s)."""

    name: Optional[StrictStr] = Field(
        None,
        description="The name/organization of the model owner.",
    )
    contact: Optional[StrictStr] = Field(
        None,
        description="The contact information for the model owner(s).",
    )
    role: Optional[StrictStr] = Field(
        None,
        description="The role of the person e.g. developer, owner, auditor.",
    )


class Version(
    BaseModelCardField,
    composable_with=["ModelDetails", "Dataset"],
    list_factory=False,
):
    """Model or dataset version information."""

    version_str: Optional[StrictStr] = Field(
        None,
        description="The version string of the model.",
    )
    date: Optional[Union[dt_date, dt_datetime]] = Field(
        None,
        description="The date this version was released.",
    )
    description: Optional[StrictStr] = Field(
        None,
        description="A description of the version, e.g. what changed?",
    )


class License(
    BaseModelCardField,
    composable_with=["ModelDetails", "Dataset"],
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
    def validate_spdx_identifier(
        cls: "License",  # noqa: N805
        values: Dict[str, StrictStr],
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
                    f"(https://spdx.org/licenses/). Got {spdx_id} instead.",
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


class Reference(
    BaseModelCardField,
    list_factory=True,
    composable_with=["ModelDetails", "Dataset"],
):
    """Reference to additional resources related to the model or dataset."""

    link: Optional[AnyUrl] = Field(None, description="A URL to the reference resource.")


class Citation(
    BaseModelCardField,
    composable_with=["ModelDetails", "Dataset"],
    list_factory=True,
):
    """Citation information for the model or dataset."""

    content: Optional[StrictStr] = Field(
        None,
        description="The citation content in BibTeX format.",
    )

    @validator("content")
    def parse_content(
        cls: "Citation",  # noqa: N805
        value: StrictStr,
    ) -> StrictStr:
        """Parse the citation content."""
        try:
            formatted_citation = PybtexEngine().format_from_string(
                value,
                style="unsrt",
                output_backend="text",
            )
            if formatted_citation == "":
                raise ValueError(f"Expected a valid BibTeX entry. Got {value}.")
        except PybtexError as exc:
            raise ValueError(
                f"Encountered an error while parsing the citation content: {value}"
                f"\n\n{exc}",
            ) from exc
        return value


class RegulatoryRequirement(
    BaseModelCardField,
    composable_with=["ModelDetails"],
    list_factory=True,
):
    """Regulatory requirements for the model or dataset."""

    regulation: Optional[StrictStr] = Field(None, description="Name of the regulation")


class Graphic(
    BaseModelCardField,
    list_factory=True,
    composable_with=["GraphicsCollection"],
):
    """A graphic to be displayed in the model card."""

    name: Optional[StrictStr] = Field(None, description="The name of the graphic.")
    image: Optional[StrictStr] = Field(
        None,
        description="The image, encoded as a base64 string or an html string.",
    )


class GraphicsCollection(BaseModelCardField, composable_with="Any"):
    """A collection of graphics to be displayed in the model card."""

    description: Optional[StrictStr] = Field(
        None,
        description="A description of the Graphics collection.",
    )
    collection: Optional[List[Graphic]] = Field(
        description="A collection of graphics.",
        default_factory=list,
    )


class SensitiveData(BaseModelCardField, composable_with=["Dataset"], list_factory=True):
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
            """,
        ),
    )


class Dataset(BaseModelCardField, composable_with=["Datasets"], list_factory=True):
    """Details about the dataset."""

    description: Optional[StrictStr] = Field(
        None,
        description="A high-level description of the dataset.",
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
        None,
        description="Visualizations of the dataset.",
    )
    split: Optional[StrictStr] = Field(
        None,
        description="The split of the dataset e.g. train, test, validation.",
    )
    size: Optional[StrictInt] = Field(
        None,
        description="The number of samples in the dataset.",
    )
    sensitive_data: Optional[SensitiveData] = Field(
        None,
        description="Does this dataset contain any human, PII, or sensitive data?",
    )


class KeyVal(
    BaseModelCardField,
    list_factory=True,
    composable_with=["ModelParameters"],
):
    """A key-value pair."""

    key: Optional[StrictStr] = None
    value: Any = None


class Test(
    BaseModelCardField,
    composable_with=["PerformanceMetric", "FairnessReport", "ExplainabilityReport"],
    list_factory=True,
):
    """Tests for the model."""

    name: Optional[StrictStr] = Field(None, description="The name of the test.")
    description: Optional[StrictStr] = Field(
        None,
        description="User-friendly description of the test.",
    )
    threshold: Optional[StrictFloat] = Field(
        None,
        description="Threshold required to pass the test.",
    )
    result: Optional[Union[StrictFloat, StrictInt]] = Field(
        None,
        description="Result returned by the test.",
    )
    passed: Optional[StrictBool] = Field(
        None,
        description="Whether the model result satisfies the given threshold.",
    )
    graphics: Optional[GraphicsCollection] = Field(
        None,
        description="A collection of visualizations related to the test.",
    )


class PerformanceMetric(
    BaseModelCardField,
    composable_with=["QuantitativeAnalysis"],
    list_factory=True,
    arbitrary_types_allowed=True,
):
    """Performance metrics for model evaluation."""

    type: Optional[StrictStr] = Field(
        None,
        description="The type of performance metric.",
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
        None,
        description="User-friendly description of the performance metric.",
    )
    graphics: Optional[GraphicsCollection] = Field(
        None,
        description="A collection of visualizations related to the metric and slice.",
    )
    tests: Optional[List[Test]] = Field(
        description="A collection of tests associated with the metric.",
        default_factory=list,
    )


class User(
    BaseModelCardField,
    composable_with=["Considerations", "Dataset"],
    list_factory=True,
):
    """Details about the user."""

    description: Optional[StrictStr] = Field(
        None,
        description="A description of a user.",
    )


class UseCase(
    BaseModelCardField,
    composable_with=["Considerations", "Dataset"],
    list_factory=True,
):
    """Details about the use case."""

    description: Optional[StrictStr] = Field(
        None,
        description="A description of a use case.",
    )
    kind: Optional[Literal["primary", "out-of-scope"]] = Field(
        None,
        description=inspect.cleandoc(
            """
            The scope of the use case. Must be one of 'primary', 'downstream', or
            'out-of-scope'.""",
        ),
    )

    @validator("kind")
    def kind_must_be_valid(
        cls: "UseCase",  # noqa: N805
        value: str,
    ) -> str:
        """Validate the use case kind."""
        if isinstance(value, str):
            value = value.lower()

        if value not in ["primary", "out-of-scope"]:
            raise ValueError(
                "Use case kind must be one of 'primary', or 'out-of-scope'.",
            )
        return value


class Risk(
    BaseModelCardField,
    composable_with=["Considerations", "Dataset"],
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
    composable_with=["Considerations", "Dataset"],
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
        None,
        description="Expected benefits to the identified groups.",
    )
    harms: Optional[StrictStr] = Field(
        None,
        description="Expected harms to the identified groups.",
    )
    mitigation_strategy: Optional[StrictStr] = Field(
        None,
        description=(
            "With respect to the benefits and harms outlined, please describe any "
            "mitigation strategy implemented."
        ),
    )


class ExplainabilityReport(
    BaseModelCardField,
    composable_with=["ExplainabilityAnalysis"],
):
    """Explainability reports for the model."""

    type: Optional[StrictStr] = Field(
        None,
        description="The type of explainability method.",
    )
    slice: Optional[StrictStr] = Field(
        None,
        description="""
        The name of the slice the explainability method was computed on.
        By default, assume this metric is not sliced.
        """,
    )
    description: Optional[StrictStr] = Field(
        None,
        description="User-friendly description of the explainability method.",
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


class FairnessReport(BaseModelCardField, composable_with=["FairnessAnalysis"]):
    """Fairness reports for the model."""

    type: Optional[StrictStr] = Field(
        None,
        description="The type of fairness study conducted.",
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
        None,
        description="User-friendly description of the fairness method.",
    )
    graphics: Optional[GraphicsCollection] = Field(
        None,
        description="A collection of visualizations related to the fairness method.",
    )
    tests: Optional[List[Test]] = Field(
        description="Tests related to fairness considerations.",
        default_factory=list,
    )
