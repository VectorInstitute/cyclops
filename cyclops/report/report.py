"""Cyclops report module."""  # pylint: disable=too-many-lines
import base64
import inspect
import keyword
import os
from datetime import date as dt_date
from datetime import datetime as dt_datetime
from io import BytesIO
from re import sub as re_sub
from typing import Any, Dict, List, Literal, Optional, Type, Union

import jinja2
from plotly.graph_objects import Figure
from plotly.io import write_image
from pydantic import BaseModel, StrictStr, create_model
from pydantic.fields import FieldInfo, ModelField
from scour import scour

from cyclops.report.model_card import (
    BaseModelCardField,
    Citation,
    Dataset,
    FairnessAssessment,
    Graphic,
    GraphicsCollection,
    License,
    ModelCard,
    Owner,
    PerformanceMetric,
    Reference,
    RegulatoryRequirement,
    Risk,
    SensitiveData,
    UseCase,
    User,
    Version,
)
from cyclops.report.utils import (
    _object_is_in_model_card_module,
    _raise_if_not_dict_with_str_keys,
    str_to_snake_case,
)

# pylint: disable=fixme

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
_DEFAULT_TEMPLATE_FILENAME = "cyclops_generic_template_dark.jinja"


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

    _model_card = ModelCard()  # type: ignore[call-arg]

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self.output_dir = output_dir or os.getcwd()

    @classmethod
    def from_json_file(
        cls, path: str, output_dir: Optional[str] = None
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
        report._model_card = model_card  # pylint: disable=protected-access
        return report

    def _get_section(self, name: str) -> BaseModel:
        """Get a section of the model card.

        Raises
        ------
        KeyError
            If the given section name is not in the model card.
        TypeError
            If the given section name is not a subclass of `BaseModel`.

        """
        name_ = str_to_snake_case(name)
        model_card_sections = self._model_card.__fields__
        if name_ not in model_card_sections:
            raise KeyError(
                f"Expected `section_name` to be in {list(model_card_sections.keys())}."
                f" Got {name} instead."
            )

        # instantiate section if not already instantiated
        section_type = model_card_sections[name_].type_
        if not isinstance(getattr(self._model_card, name_), section_type):
            setattr(self._model_card, name_, section_type())

        model_card_section: BaseModel = getattr(self._model_card, name_)
        if not issubclass(model_card_section.__class__, BaseModel):
            raise TypeError(
                f"Expected section `{name}` to be a subclass of `BaseModel`."
                f" Got {model_card_section.__class__} instead."
            )

        return model_card_section

    def _log_field(
        self,
        data: Any,
        section_name: str,
        field_name: str,
        field_type: Optional[Type[BaseModel]] = None,
    ) -> None:
        """Populate a field in the model card.

        Parameters
        ----------
        data : Any
            Data to populate the field with.
        section_name : str
            Name of the section to populate.
        field_name : str
            Name of the field to populate. If the field does not exist, it will be
            created and added to the section.
        field_type : BaseModel, optional
            Type of the field to populate. If not provided, the type will be inferred
            from the data.

        Raises
        ------
        ValueError
            If `field_name` is not a valid python identifier.

        """
        section = self._get_section(section_name)

        section_fields = section.__fields__
        if field_name in section_fields:
            field = section_fields[field_name]
            field_type = field.type_  # [!] can be any (serializable) type
            field_value = _get_field_value(field_type, data)

            _check_allowable_sections(  # check if field can be added to section
                field=field_value, section_name=section_name, field_name=field_name
            )

            if field.default_factory == list:
                # NOTE: pydantic does not trigger validation when appending to a list,
                # but if `validate_assignment` is set to `True`, then validation will
                # be triggered when the list is assigned to the field.
                field_values = getattr(section, field_name, [])
                field_values.append(field_value)
                setattr(section, field_name, field_values)  # trigger validation
            else:
                setattr(section, field_name, field_value)
        else:
            if not field_name.isidentifier() or keyword.iskeyword(field_name):
                raise ValueError(
                    f"Expected `field_name` to be a valid python identifier."
                    f" Got {field_name} instead."
                )

            field_value = _get_field_value(field_type, data)
            _check_allowable_sections(  # check if field is allowed in section
                field=field_value, section_name=section_name, field_name=field_name
            )

            type_ = field_type or type(field_value)
            default_factory = None
            if (
                isinstance(field_value, BaseModel)
                and hasattr(field_value.__config__, "list_factory")
                and getattr(field_value.__config__, "list_factory") is True
            ):
                default_factory = list
                field_value = [field_value]  # add field as a list
                type_ = List[type_]  # type: ignore[valid-type]

            setattr(section, field_name, field_value)

            # modify __fields__ to include new field
            section_fields[field_name] = ModelField(
                name=field_name,
                type_=type_,
                required=False,
                class_validators=None,
                model_config=BaseModelCardField.Config,
                default_factory=default_factory,
                field_info=FieldInfo(unique_items=True)
                if default_factory == list
                else None,
            )

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
        section_name_ = str_to_snake_case(section_name)
        section = self._get_section(section_name_)
        populated_section = section.parse_obj(data)
        setattr(self._model_card, section_name_, populated_section)

    def log_descriptor(
        self, name: str, description: str, section_name: str, **extra: Any
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
            If the given name conflicts with a defined class in the `model_card`
            module.

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
                f"{name}. Please use a different name."
            )

        self._log_field(
            data={"description": description, **extra},
            section_name=section_name,
            field_name=str_to_snake_case(name),
            field_type=field_obj,
        )

    def log_plotly_figure(self, fig: Figure, alt: str, section_name: str) -> None:
        """Add a plotly figure to a section of the report.

        Parameters
        ----------
        fig : Figure
            The plotly figure to add.
        alt : str
            The alt text for the figure.
        section_name : str
            The section of the report to add the figure to.

        Raises
        ------
        KeyError
            If the given section name is not valid.

        """
        buffer = BytesIO()
        write_image(fig, buffer, format="svg", validate=True)

        scour_options = scour.sanitizeOptions()
        scour_options.remove_descriptive_elements = True
        svg: str = scour.scourString(buffer.getvalue(), options=scour_options)

        # convert svg to base64
        svg = base64.b64encode(svg.encode("utf-8")).decode("utf-8")

        self._log_field(
            data={"name": alt, "image": f"data:image/svg+xml;base64,{svg}"},
            section_name=section_name,
            field_name="figures",
            field_type=Graphic,
        )

    # loggers for `Model Details` section
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
        self, citation: str, section_name: str = "model_details", **extra: Any
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
        self, link: str, section_name: str = "model_details", **extra: Any
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

    # loggers for `Model Parameters` section
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
            section_name="model_parameters",
            field_name="data",
            field_type=Dataset,
        )

    # loggers for `Considerations` section
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
        kind: Optional[Literal["primary", "downstream", "out-of-scope"]] = "primary",
        section_name: str = "considerations",
        **extra: Any,
    ) -> None:
        """Add a use case to a section of the report.

        Parameters
        ----------
        description : str
            A description of the use case.
        kind : Literal["primary", "downstream", "out-of-scope"], optional
            The kind of use case. If not provided, the use case will be
            considered primary.
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

    # loggers for `Quantitative Analysis` section
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
                metric_name = name_split[0]
            else:  # everything before the last slash is the slice name
                slice_name = "/".join(name_split[:-1])
                metric_name = name_split[-1]

            # TODO: create plot

            self._log_field(
                data={"type": metric_name, "value": metric_value, "slice": slice_name},
                section_name="quantitative_analysis",
                field_name="performance_metrics",
                field_type=PerformanceMetric,
            )

    # TODO: MERGE/COMPARE MODEL CARDS

    # EXPORTING THE REPORT
    def validate(self) -> None:
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
        self, template_path: Optional[str] = None
    ) -> jinja2.Template:
        """Get a jinja2 template."""
        _template_path = template_path or os.path.join(
            _TEMPLATE_DIR, _DEFAULT_TEMPLATE_FILENAME
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

        jinja_env.tests["list"] = lambda x: isinstance(x, list)
        jinja_env.tests["class"] = inspect.isclass
        jinja_env.filters["regex_replace"] = regex_replace
        jinja_env.tests["empty"] = empty

        return jinja_env.get_template(template_file)

    def export(
        self,
        output_filename: Optional[str] = None,
        template_path: Optional[str] = None,
        save_json: bool = True,
    ) -> None:
        """Export the model card report to an HTML file.

        Parameters
        ----------
        output_filename : str, optional
            The name of the output file. If not provided, the file will be named
            with the current date and time.
        template_path : str, optional
            The path to the jinja2 template to use. The default is None, which uses
            the default template provided by Cylops.
        save_json : bool, optional
            Whether to save the model card as a JSON file. The default is True.

        """
        assert (
            output_filename is None
            or isinstance(output_filename, str)
            and output_filename.endswith(".html")
        ), "`output_filename` must be a string ending with '.html'"
        self.validate()
        template = self._get_jinja_template(template_path=template_path)
        content = template.render(model_card=self._model_card)

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
                json_path, self._model_card.json(indent=2, exclude_unset=True)
            )


def _get_field_value(field_type: Any, data: Any) -> Any:
    """Get the value of a field."""
    if field_type is None:
        return data

    if issubclass(field_type, BaseModel):
        _raise_if_not_dict_with_str_keys(data)
        return field_type(**data)

    # explicitly handle `Union` types
    if field_type.__class__.__module__ == "typing" and field_type.__origin__ == Union:
        # try to match `data` to one of the types in the `Union`
        for union_type in field_type.__args__:
            try:
                return _get_field_value(union_type, data)
            except TypeError:
                pass
        # if no match is found, raise an error
        raise TypeError(
            f"Expected `data` to be one of {field_type.__args__} types."
            f"Got {type(data)} instead."
        )

    return data


def _check_allowable_sections(
    field: BaseModel, section_name: str, field_name: str
) -> None:
    """Check if a field can be added to a section.

    Parameters
    ----------
    field : BaseModel
        The field to add to the section.
    section_name : str
        The name of the section.
    field_name : str
        The name of the field.

    Raises
    ------
    ValueError
        If the field cannot be added to the section.

    """
    if (
        field is not None
        and isinstance(field, BaseModel)
        and hasattr(field.__config__, "allowable_sections")
    ):
        allowable_sections = getattr(field.__config__, "allowable_sections")
        if (allowable_sections is not None and len(allowable_sections) > 0) and (
            section_name not in allowable_sections
        ):
            raise ValueError(
                f"Field `{field_name}` cannot be added to section `{section_name}`."
                f"Expected section to be one of {allowable_sections}."
            )
