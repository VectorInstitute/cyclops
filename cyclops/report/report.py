"""Cyclops report module."""

import base64
import os
from datetime import date as dt_date
from datetime import datetime as dt_datetime
from io import BytesIO
from re import sub as re_sub
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

import jinja2
import plotly.graph_objects as go
from PIL import Image
from plotly.graph_objects import Figure
from plotly.io import write_image
from plotly.offline import get_plotlyjs
from pydantic import BaseModel, StrictStr, create_model
from scour import scour

from cyclops.report.model_card import ModelCard  # type: ignore[attr-defined]
from cyclops.report.model_card.base import BaseModelCardField
from cyclops.report.model_card.fields import (
    Citation,
    Dataset,
    ExplainabilityReport,
    FairnessAssessment,
    FairnessReport,
    Graphic,
    GraphicsCollection,
    License,
    Owner,
    PerformanceMetric,
    Reference,
    RegulatoryRequirement,
    Risk,
    SensitiveData,
    Test,
    UseCase,
    User,
    Version,
)
from cyclops.report.utils import (
    _object_is_in_model_card_module,
    _raise_if_not_dict_with_str_keys,
    str_to_snake_case,
)


_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
_DEFAULT_TEMPLATE_FILENAME = "cyclops_generic_template_light.jinja"


class ModelCardReport:
    """Model card report.

    This class serves as an interface to populate a `ModelCard` object and generate
    an HTML report from it.

    Parameters
    ----------
    output_dir : str, optional
        Path to the directory where the model card report will be saved. If not
        provided, the report will be saved in the current working directory.

    """

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self.output_dir = output_dir or os.getcwd()
        self._model_card = ModelCard()  # type: ignore[call-arg]

    @classmethod
    def from_json_file(
        cls,
        path: str,
        output_dir: Optional[str] = None,
    ) -> "ModelCardReport":
        """Load a model card from a file.

        Parameters
        ----------
        path : str
            The path to a JSON file containing model card data.
        output_dir : str, optional
            The directory to save the report to. If not provided, the report will
            be saved in a directory called `cyclops_reports` in the current working
            directory.

        Returns
        -------
        ModelCardReport
            The model card report.

        """
        model_card = ModelCard.parse_file(path)
        report = ModelCardReport(output_dir=output_dir)
        report._model_card = model_card
        return report

    def _log_field(
        self,
        data: Dict[str, Any],
        section_name: str,
        field_name: str,
        field_type: Type[BaseModel],
    ) -> None:
        """Populate a field in the model card.

        Parameters
        ----------
        data : Dict[str, Any]
            Data to populate the field with.
        section_name : str
            Name of the section to populate.
        field_name : str
            Name of the field to populate. If the field does not exist, it will be
            created and added to the section.
        field_type : BaseModel
            Type of the field to populate.

        """
        section_name = str_to_snake_case(section_name)
        section = self._model_card.get_section(section_name)
        field_value = field_type.parse_obj(data)

        if field_name in section.__fields__:
            section.update_field(field_name, field_value)
        else:
            field_name = str_to_snake_case(field_name)
            section.add_field(field_name, field_value)

    def log_from_dict(self, data: Dict[str, Any], section_name: str) -> None:
        """Populate fields in the model card from a dictionary.

        The keys of the dictionary serve as the field names in the specified section.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary of data to populate the fields with.
        section_name : str
            Name of the section to populate.

        """
        _raise_if_not_dict_with_str_keys(data)
        section_name = str_to_snake_case(section_name)
        section = self._model_card.get_section(section_name)

        # get data already in section and update with new data
        section_data = section.dict()
        section_data.update(data)

        populated_section = section.__class__.parse_obj(section_data)
        setattr(self._model_card, section_name, populated_section)

    def log_descriptor(
        self,
        name: str,
        description: str,
        section_name: str,
        **extra: Any,
    ) -> None:
        """Add a descriptor to a section of the report.

        This method will create a new pydantic `BaseModel` subclass with the given
        name, which has a field named `description` of type `str`. As long as the
        descriptor name does not conflict with a defined class in the `model_card`
        module, the descriptor can be added to any section of the report.

        Parameters
        ----------
        name : str
            The name of the descriptor.
        description : str
            A description of the descriptor.
        section_name : str
            The section of the report to add the descriptor to.
        **extra
            Any extra fields to add to the descriptor.

        Raises
        ------
        KeyError
            If the given section name is not valid.
        ValueError
            If the given name conflicts with a defined class in the `model_card` module.

        Examples
        --------
        >>> from cylops.report import ModelCardReport
        >>> report = ModelCardReport()
        >>> report.log_descriptor(
        ...     name="tradeoff",
        ...     description="We trade off performance for interpretability.",
        ...     section_name="considerations",
        ... )

        """
        # use `name` to create BaseModel subclass
        field_obj = create_model(
            "".join(char for char in name.title() if not char.isspace()),  # PascalCase
            __base__=BaseModelCardField,
            __cls_kwargs__={"list_factory": True},  # all descriptors are lists
            description=(
                StrictStr,
                None,
            ),  # <field_name>=(<field_type>, <default_value>)
        )

        # make sure the field_obj doesn't conflict with any of the existing objects
        if _object_is_in_model_card_module(field_obj):
            raise ValueError(
                "Encountered name conflict when trying to create a descriptor for "
                f"{name}. Please use a different name.",
            )

        self._log_field(
            data={"description": description, **extra},
            section_name=section_name,
            field_name=str_to_snake_case(name),
            field_type=field_obj,
        )

    def _log_graphic_collection(
        self,
        graphic: Graphic,
        description: str,
        section_name: str,
    ) -> None:
        # get the section
        section_name = str_to_snake_case(section_name)
        section = self._model_card.get_section(section_name)

        # append graphic to exisiting GraphicsCollection or create new one
        if (
            "graphics" in section.__fields__
            and section.__fields__["graphics"].type_ is GraphicsCollection
            and section.graphics is not None  # type: ignore
        ):
            section.graphics.collection.append(graphic)  # type: ignore
        else:
            self._log_field(
                data={"description": description, "collection": [graphic]},
                section_name=section_name,
                field_name="graphics",
                field_type=GraphicsCollection,
            )

    def log_image(self, img_path: str, caption: str, section_name: str) -> None:
        """Add an image to a section of the report.

        Parameters
        ----------
        img_path : str
            The path to the image file.
        caption : str
            The caption for the image.
        section_name : str
            The section of the report to add the image to.

        Raises
        ------
        KeyError
            If the given section name is not valid.
        ValueError
            If the given image path does not exist.

        """
        if not os.path.exists(img_path):
            raise ValueError(f"Image path {img_path} does not exist.")

        with Image.open(img_path) as img:
            buffered = BytesIO()
            img.save(buffered, format=img.format)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

        graphic = Graphic.parse_obj(
            {"name": caption, "image": f"data:image/{img.format};base64,{img_base64}"},
        )

        self._log_graphic_collection(graphic, "Images", section_name)

    def log_plotly_figure(
        self,
        fig: Figure,
        caption: str,
        section_name: str,
        interactive: bool = True,
    ) -> None:
        """Add a plotly figure to a section of the report.

        Parameters
        ----------
        fig : Figure
            The plotly figure to add.
        caption : str
            The caption for the figure.
        section_name : str
            The section of the report to add the figure to.
        interactive : bool, optional, default=True
            Whether or not the figure should be an interactive plot.

        Raises
        ------
        KeyError
            If the given section name is not valid.

        """
        if interactive:
            data = {
                "name": caption,
                "image": fig.to_html(full_html=False, include_plotlyjs=False),
            }
        else:
            bytes_buffer = BytesIO()
            write_image(fig, bytes_buffer, format="svg", validate=True)

            scour_options = scour.sanitizeOptions()
            scour_options.remove_descriptive_elements = True
            svg: str = scour.scourString(bytes_buffer.getvalue(), options=scour_options)

            # convert svg to base64
            svg = base64.b64encode(svg.encode("utf-8")).decode("utf-8")

            data = {"name": caption, "image": f"data:image/svg+xml;base64,{svg}"}

        graphic = Graphic.parse_obj(data)  # create Graphic object from data

        self._log_graphic_collection(graphic, "Plots", section_name)

    def log_owner(
        self,
        name: str,
        contact: Optional[str] = None,
        role: Optional[str] = None,
        section_name: str = "model_details",
        **extra: Any,
    ) -> None:
        """Add an owner to a section of the report.

        Parameters
        ----------
        name : str
            The name of the owner.
        contact : str, optional
            The contact information for the owner.
        role : str, optional
            The role of the owner.
        section_name : str, optional
            The name of the section of the report to log the owner to. If not provided,
            the owner will be added to the `model_details` section, representing
            the model owner.
        **extra
            Any extra fields to add to the Owner.

        Raises
        ------
        KeyError
            If the given section name is not valid.

        """
        self._log_field(
            data={"name": name, "contact": contact, "role": role, **extra},
            section_name=section_name,
            field_name="owners",
            field_type=Owner,
        )

    def log_version(
        self,
        version_str: str,
        date: Optional[Union[dt_date, dt_datetime, str, int, float]] = None,
        description: Optional[str] = None,
        section_name: str = "model_details",
        **extra: Any,
    ) -> None:
        """Add a version to a section of the report.

        Parameters
        ----------
        version_str : str
            The version number or identifier as a string. This can be a semantic
            version number, e.g. "1.0.0", or a custom identifier, e.g. "v1".
        date : Union[dt_date, dt_datetime, str, int, float], optional
            The date of the version. This can be a datetime/date object, an integer
            or float representing a UNIX timestamp, or a string in the format
            `YYYY-MM-DD[T]HH:MM[:SS[.ffffff]][Z or [±]HH[:]MM]]` or `YYYY-MM-DD`.
        description : str, optional
            A description of the version. This can be used to summarize the changes
            made in the version or to provide additional context.
        section_name : str, optional
            The section of the report to add the version to. If not provided,
            the version will be added to the `model_details` section, representing
            the version of the model as a whole.
        **extra
            Any extra fields to add to the Version.

        Raises
        ------
        KeyError
            If the given section name is not valid.

        """
        self._log_field(
            data={
                "version": version_str,
                "date": date,
                "description": description,
                **extra,
            },
            section_name=section_name,
            field_name="version",
            field_type=Version,
        )

    def log_license(
        self,
        identifier: str,
        text: Optional[str] = None,
        section_name: str = "model_details",
        **extra: Any,
    ) -> None:
        """Add a license to a section of the report.

        Parameters
        ----------
        identifier : str
            The SPDX identifier of the license, e.g. "Apache-2.0".
            See https://spdx.org/licenses/ for a list of valid identifiers.
            For custom licenses, set the `identifier` to "unknown", "unlicensed",
            or "proprietary" and provide the full license text in the `text` field,
            if available.
        text : str, optional
            The full text of the license. This is useful for custom licenses
            that are not in the SPDX list.
        section_name : str, optional
            The section of the report to add the license to. If not provided,
            the license will be added to the `model_details` section, representing
            the license for the model as a whole.
        **extra
            Any extra fields to add to the License.

        Raises
        ------
        KeyError
            If the given section name is not valid.

        Notes
        -----
        If the license is not found in the SPDX list, the license text will be
        left blank. If the license text is provided, it will be used instead.

        """
        self._log_field(
            data={"identifier": identifier, "text": text, **extra},
            section_name=section_name,
            field_name="licenses",
            field_type=License,
        )

    def log_citation(
        self,
        citation: str,
        section_name: str = "model_details",
        **extra: Any,
    ) -> None:
        """Add a citation to a section of the report.

        Parameters
        ----------
        citation : str
            The citation content.
        section_name : str, optional
            The section of the report to add the citation to. If not provided,
            the citation will be added to the `model_details` section, representing
            the citation for the model.
        **extra

        Raises
        ------
        KeyError
            If the given section name is not valid.

        Notes
        -----
        If the citation content is a valid BibTeX entry, the citation will be
        formatted as plain text and added to the report.

        """
        self._log_field(
            data={"content": citation, **extra},
            section_name=section_name,
            field_name="citations",
            field_type=Citation,
        )

    def log_reference(
        self,
        link: str,
        section_name: str = "model_details",
        **extra: Any,
    ) -> None:
        """Add a reference to a section of the report.

        Parameters
        ----------
        link : str
            A link to a resource that provides relevant context.
        section_name : str, optional
            The section of the report to add the reference to. If not provided,
            the reference will be added to the `model_details` section, representing
            the reference for the model.
        **extra
            Any extra fields to add to the Reference.

        Raises
        ------
        KeyError
            If the given section name is not valid.

        """
        self._log_field(
            data={"link": link, **extra},
            section_name=section_name,
            field_name="references",
            field_type=Reference,
        )

    def log_regulation(
        self,
        regulation: str,
        section_name: str = "model_details",
        **extra: Any,
    ) -> None:
        """Add a regulatory requirement to a section of the report.

        Parameters
        ----------
        regulation : str
            The regulatory requirement that must be complied with.
        section_name : str, optional
            The section of the report to add the regulatory requirement to.
            If not provided, the regulatory requirement will be added to the
            `model_details` section.

        Raises
        ------
        KeyError
            If the given section name is not valid.

        """
        self._log_field(
            data={"regulation": regulation, **extra},
            section_name=section_name,
            field_name="regulatory_requirements",
            field_type=RegulatoryRequirement,
        )

    def log_model_parameters(self, params: Dict[str, Any]) -> None:
        """Log model parameters.

        Parameters
        ----------
        params : Dict[str, Any]
            A dictionary of model parameters.

        """
        self.log_from_dict(params, section_name="model_parameters")

    def log_dataset(
        self,
        description: Optional[str] = None,
        citation: Optional[str] = None,
        link: Optional[str] = None,
        license_id: Optional[str] = None,
        version: Optional[str] = None,
        features: Optional[List[str]] = None,
        split: Optional[str] = None,
        sensitive_features: Optional[List[str]] = None,
        sensitive_feature_justification: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Log information about the dataset used to train/evaluate the model.

        Parameters
        ----------
        description : str, optional
            A description of the dataset.
        citation : str, optional
            A citation for the dataset. This can be a BibTeX entry or a plain-text
            citation.
        link : str, optional
            A link to a resource that provides relevant context e.g. the homepage
            of the dataset.
        license_id : str, optional
            The SPDX identifier of the license, e.g. "Apache-2.0".
            See https://spdx.org/licenses/ for a list of valid identifiers.
            For custom licenses, set the `identifier` to "unknown", "unlicensed",
            or "proprietary".
        version : str, optional
            The version of the dataset.
        features : list of str, optional
            The names of the features used to train/evaluate the model.
        split : str, optional
            The name of the split used to train/evaluate the model.
        sensitive_features : list of str, optional
            The names of the sensitive features used to train/evaluate the model.
        sensitive_feature_justification : str, optional
            A justification for the sensitive features used to train/evaluate the
            model.
        **extra
            Any extra fields to add to the Dataset.

        Raises
        ------
        AssertionError
            If the sensitive features are not in the features list.

        """
        # sensitive features must be in features
        if features is None and sensitive_features is not None:
            assert all(
                feature in features for feature in sensitive_features  # type: ignore
            ), "All sensitive features must be in the features list."

        # TODO: plot dataset distribution
        data = {
            "description": description,
            "citation": Citation(content=citation),
            "reference": Reference(link=link),  # type: ignore
            "license": License(identifier=license_id),  # type: ignore
            "version": Version(version_str=version),  # type: ignore
            "features": features,
            "split": split,
            "sensitive_data": SensitiveData(
                sensitive_data_used=sensitive_features,
                justification=sensitive_feature_justification,
            ),
            **extra,
        }
        self._log_field(
            data=data,
            section_name="datasets",
            field_name="data",
            field_type=Dataset,
        )

    def log_user(
        self,
        description: str,
        section_name: str = "considerations",
        **extra: Any,
    ) -> None:
        """Add a user description to a section of the report.

        Parameters
        ----------
        description : str
            A description of the user.
        section_name : str, optional
            The section of the report to add the user to. If not provided, the user
            will be added to the `considerations` section.
        **extra
            Any extra fields to add to the User.

        Raises
        ------
        KeyError
            If the given section name is not valid.

        """
        self._log_field(
            data={"description": description, **extra},
            section_name=section_name,
            field_name="users",
            field_type=User,
        )

    def log_use_case(
        self,
        description: str,
        kind: Literal["primary", "out-of-scope"],
        section_name: str = "considerations",
        **extra: Any,
    ) -> None:
        """Add a use case to a section of the report.

        Parameters
        ----------
        description : str
            A description of the use case.
        kind : Literal["primary", "out-of-scope"]
            The kind of use case - either "primary" or "out-of-scope".
        section_name : str, optional
            The section of the report to add the use case to. If not provided,
            the use case will be added to the `considerations` section.
        **extra
            Any extra fields to add to the UseCase.

        Raises
        ------
        KeyError
            If the given section name is not valid.

        """
        self._log_field(
            data={"description": description, "kind": kind, **extra},
            section_name=section_name,
            field_name="use_cases",
            field_type=UseCase,
        )

    def log_risk(
        self,
        risk: str,
        mitigation_strategy: str,
        section_name: str = "considerations",
        **extra: Any,
    ) -> None:
        """Add a risk to a section of the report.

        Parameters
        ----------
        risk : str
            A description of the risk.
        mitigation_strategy : str
            A description of the mitigation strategy.
        section_name : str, optional
            The section of the report to add the risk to. If not provided, the
            risk will be added to the `considerations` section.
        **extra
            Any extra information to add in relation to the risk.

        Raises
        ------
        KeyError
            If the given section name is not valid.

        """
        self._log_field(
            data={
                "risk": risk,
                "mitigation_strategy": mitigation_strategy,
                **extra,
            },
            section_name=section_name,
            field_name="ethical_considerations",
            field_type=Risk,
        )

    def log_fairness_assessment(
        self,
        affected_group: str,
        benefit: str,
        harm: str,
        mitigation_strategy: str,
        section_name: str = "considerations",
        **extra: Any,
    ) -> None:
        """Add a fairness assessment to a section of the report.

        Parameters
        ----------
        affected_group : str
            A description of the affected group.
        benefit : str
            A description of the benefit(s) to the affected group.
        harm : str
            A description of the harm(s) to the affected group.
        mitigation_strategy : str
            A description of the mitigation strategy.
        section_name : str, optional
            The section of the report to add the fairness assessment to. If not
            provided, the fairness assessment will be added to the `considerations`
            section.
        **extra
            Any extra information to add in relation to the fairness assessment.

        Raises
        ------
        KeyError
            If the given section name is not valid.

        """
        self._log_field(
            data={
                "affected_group": affected_group,
                "benefits": benefit,
                "harms": harm,
                "mitigation_strategy": mitigation_strategy,
                **extra,
            },
            section_name=section_name,
            field_name="fairness_assessment",
            field_type=FairnessAssessment,
        )

    def log_quantitative_analysis(
        self,
        analysis_type: Literal["performance", "fairness", "explainability"],
        name: str,
        value: Any,
        metric_slice: Optional[str] = None,
        decision_threshold: Optional[float] = None,
        description: Optional[str] = None,
        pass_fail_thresholds: Optional[Union[float, List[float]]] = None,
        pass_fail_threshold_fns: Optional[
            Union[Callable[[Any, float], bool], List[Callable[[Any, float], bool]]]
        ] = None,
        **extra: Any,
    ) -> None:
        """Add a quantitative analysis to the report.

        Parameters
        ----------
        analysis_type : Literal["performance", "fairness", "explainability"]
            The type of analysis to log.
        name : str
            The name of the metric.
        value : Any
            The value of the metric.
        metric_slice : str, optional
            The name of the slice. If not provided, the slice name will be "overall".
        decision_threshold : float, optional
            The decision threshold for the metric.
        description : str, optional
            A description of the metric.
        pass_fail_thresholds : Union[float, List[float]], optional
            The pass/fail threshold(s) for the metric. If a single threshold is
            provided, a single test will be created. If multiple thresholds are
            provided, multiple tests will be created.
        pass_fail_threshold_fns : Union[Callable[[Any, float], bool],
                                  List[Callable[[Any, float], bool]]], optional
            The pass/fail threshold function(s) for the metric. If a single function
            is provided, a single test will be created. If multiple functions are
            provided, multiple tests will be created.
        **extra
            Any extra fields to add to the metric.

        Raises
        ------
        ValueError
            If the given metric type is not valid.

        """
        if analysis_type not in ["performance", "fairness", "explainability"]:
            raise ValueError(
                f"Invalid metric type {analysis_type}. Must be one of 'performance', "
                "'fairness', or 'explainability'.",
            )

        section_name: str
        field_name: str
        field_type: Any

        section_name, field_name, field_type = {
            "performance": (
                "quantitative_analysis",
                "performance_metrics",
                PerformanceMetric,
            ),
            "fairness": ("fairness_analysis", "fairness_reports", FairnessReport),
            "explainability": (
                "explainability_analysis",
                "explainability_reports",
                ExplainabilityReport,
            ),
        }[analysis_type]

        data = {
            "type": name,
            "value": value,
            "slice": metric_slice,
            "decision_threshold": decision_threshold,
            "description": description,
            **extra,
        }

        # TODO: create graphics

        if pass_fail_thresholds is not None and pass_fail_threshold_fns is not None:
            if isinstance(pass_fail_thresholds, float):
                pass_fail_thresholds = [pass_fail_thresholds]
            if callable(pass_fail_threshold_fns):
                pass_fail_threshold_fns = [pass_fail_threshold_fns]

            # create Test objects
            tests = []
            for threshold, threshold_fn in zip(
                pass_fail_thresholds,
                pass_fail_threshold_fns,
            ):
                tests.append(
                    Test(
                        name=f"{name}/{metric_slice}" if metric_slice else name,
                        description=None,
                        threshold=threshold,
                        result=value,
                        passed=threshold_fn(value, threshold),
                        graphics=None,
                    ),
                )

            data["tests"] = tests

        self._log_field(
            data=data,
            section_name=section_name,
            field_name=field_name,
            field_type=field_type,
        )

    def log_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add a performance metric to the `Quantitative Analysis` section.

        Parameters
        ----------
        metrics : Dict[str, Any]
            A dictionary of performance metrics. The keys should be the name of the
            metric, and the values should be the value of the metric. If the metric
            is a slice metric, the key should be the slice name followed by a slash
            and then the metric name (e.g. "slice_name/metric_name"). If no slice
            name is provided, the slice name will be "overall".

        Raises
        ------
        TypeError
            If the given metrics are not a dictionary with string keys.

        """
        _raise_if_not_dict_with_str_keys(metrics)
        for metric_name, metric_value in metrics.items():
            name_split = metric_name.split("/")
            if len(name_split) == 1:
                slice_name = "overall"
                metric_name = name_split[0]  # noqa: PLW2901
            else:  # everything before the last slash is the slice name
                slice_name = "/".join(name_split[:-1])
                metric_name = name_split[-1]  # noqa: PLW2901

            # TODO: create plot

            self._log_field(
                data={"type": metric_name, "value": metric_value, "slice": slice_name},
                section_name="quantitative_analysis",
                field_name="performance_metrics",
                field_type=PerformanceMetric,
            )

    # TODO: MERGE/COMPARE MODEL CARDS

    def _validate(self) -> None:
        """Validate the model card."""
        ModelCard.validate(self._model_card.dict())

    def _write_file(self, path: str, content: str) -> None:
        """Write a file to the given path.

        If the path does not exist, create it.

        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w+", encoding="utf-8") as f_handle:
            f_handle.write(content)

    def _jinja_loader(self, template_dir: str) -> jinja2.FileSystemLoader:
        """Create a jinja2 file system loader."""
        return jinja2.FileSystemLoader(template_dir)

    def _get_jinja_template(
        self,
        template_path: Optional[str] = None,
    ) -> jinja2.Template:
        """Get a jinja2 template."""
        _template_path = template_path or os.path.join(
            _TEMPLATE_DIR,
            _DEFAULT_TEMPLATE_FILENAME,
        )
        template_dir = os.path.dirname(_template_path)
        template_file = os.path.basename(_template_path)

        jinja_env = jinja2.Environment(
            loader=self._jinja_loader(template_dir),
            autoescape=True,
            auto_reload=True,
            cache_size=0,
        )

        def regex_replace(string: str, find: str, replace: str) -> str:
            """Replace a regex pattern with a string."""
            return re_sub(find, replace, string)

        def empty(x: Optional[List[Any]]) -> bool:
            """Check if a variable is empty."""
            empty = True
            if x is not None:
                for _, obj in x:
                    if isinstance(obj, list):
                        if len(obj) > 0:
                            empty = False
                    elif isinstance(obj, GraphicsCollection):
                        if len(obj.collection) > 0:  # type: ignore[arg-type]
                            empty = False
                    elif obj is not None:
                        empty = False
            return empty

        def donut_chart_tests(tests: List[Test]) -> Graphic:
            """Create a plotly donut chart for the given tests."""
            colors = ["green", "red"]
            passed = 0
            failed = 0
            for test in tests:
                if test.passed:
                    passed += 1
                else:
                    failed += 1

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=["Passed", "Failed"],
                        values=[passed, failed],
                        hole=0.4,
                        pull=[0.005, 0.005],
                        textinfo="percent",
                        marker={"colors": colors},
                        showlegend=False,
                        title=f"<b>{passed}/{passed+failed}</b><br>Tests Passed",
                    ),
                ],
            )
            # Increase font size and change font
            fig.update_layout(
                font={
                    "family": "Courier New, monospace",
                    "size": 20,
                    "color": "#7f7f7f",
                },
                autosize=False,
                width=400,
                height=400,
                margin={"l": 0, "r": 0, "b": 0, "t": 0, "pad": 0},
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            data = {
                "name": "Tests (Pass/Fail) Donut Chart",
                "image": fig.to_html(full_html=False, include_plotlyjs=False),
            }
            return Graphic.parse_obj(data)  # create Graphic object from data

        jinja_env.filters["donut_chart_tests"] = donut_chart_tests
        jinja_env.filters["regex_replace"] = regex_replace
        jinja_env.tests["list"] = lambda x: isinstance(x, list)
        jinja_env.tests["empty"] = empty
        jinja_env.tests["hasattr"] = hasattr
        jinja_env.tests["None"] = lambda x: x is None
        jinja_env.tests["int"] = lambda x: isinstance(x, int)
        jinja_env.tests["float"] = lambda x: isinstance(x, float)
        jinja_env.tests["bool"] = lambda x: isinstance(x, bool)

        return jinja_env.get_template(template_file)

    def export(
        self,
        output_filename: Optional[str] = None,
        template_path: Optional[str] = None,
        interactive: bool = True,
        save_json: bool = True,
    ) -> str:
        """Export the model card report to an HTML file.

        Parameters
        ----------
        output_filename : str, optional
            The name of the output file. If not provided, the file will be named
            with the current date and time.
        template_path : str, optional
            The path to the jinja2 template to use. The default is None, which uses
            the default template provided by CyclOps.
        save_json : bool, optional
            Whether to save the model card as a JSON file. The default is True.

        Returns
        -------
        str
            Path of the saved HTML report file.

        """
        assert (
            output_filename is None
            or isinstance(output_filename, str)
            and output_filename.endswith(".html")
        ), "`output_filename` must be a string ending with '.html'"
        self._validate()
        template = self._get_jinja_template(template_path=template_path)

        def sweep_tests(model_card: Any, tests: List[Any]) -> None:
            """Sweep model card to find all instances of Test."""
            for field in model_card:
                if isinstance(field, tuple):
                    field = field[1]  # noqa: PLW2901
                if isinstance(field, Test):
                    tests.append(field)
                if hasattr(field, "__fields__"):
                    sweep_tests(field, tests)
                if isinstance(field, list) and len(field) != 0:
                    for item in field:
                        if isinstance(item, Test):
                            if len(field) == 1:
                                tests.append(field[0])
                            else:
                                tests.append(field)
                        else:
                            sweep_tests(item, tests)

        def sweep_graphics(model_card: Any, graphics: list[Any], caption: str) -> None:
            """Sweep model card to find all instances of Test."""
            for field in model_card:
                if isinstance(field, tuple):
                    field = field[1]  # noqa: PLW2901
                if isinstance(field, Graphic) and field.name == caption:
                    graphics.append(field)
                if hasattr(field, "__fields__"):
                    sweep_graphics(field, graphics, caption)
                if isinstance(field, list) and len(field) != 0:
                    for item in field:
                        if isinstance(item, Graphic):
                            if item.name == caption:
                                graphics.append(item)
                        else:
                            sweep_graphics(item, graphics, caption)

        func_dict = {"sweep_tests": sweep_tests, "sweep_graphics": sweep_graphics}
        template.globals.update(func_dict)

        plotlyjs = get_plotlyjs() if interactive else None
        content = template.render(model_card=self._model_card, plotlyjs=plotlyjs)

        # write to file
        today = dt_date.today().strftime("%Y-%m-%d")
        now = dt_datetime.now().strftime("%H-%M-%S")
        report_path = os.path.join(
            self.output_dir,
            "cyclops_reports",
            today,
            now,
            output_filename or "model_card.html",
        )
        self._write_file(report_path, content)

        if save_json:
            json_path = report_path.replace(".html", ".json")
            self._write_file(
                json_path,
                self._model_card.json(indent=2, exclude_unset=True),
            )

        return report_path
