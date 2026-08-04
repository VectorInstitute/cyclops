"""Base classes for model card fields and sections."""

import keyword
import typing
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic.fields import FieldInfo


def _unwrap_optional(annotation: Any) -> Any:
    """Unwrap `Optional[SomeType]`/`Union[SomeType, None]` to `SomeType`."""
    if typing.get_origin(annotation) is Union:
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def unwrap_optional_type(field_info: FieldInfo) -> Any:
    """Get the annotated type of a field, unwrapping `Optional`/`Union[..., None]`.

    All model card fields/sections are declared as `Optional[SomeType]`, so
    `field_info.annotation` is `Union[SomeType, None]`. This returns `SomeType`.
    """
    return _unwrap_optional(field_info.annotation)


def _get_list_item_type(field_info: FieldInfo) -> Any:
    """Get the item type of a `List[X]`/`Optional[List[X]]` field annotation."""
    annotation = _unwrap_optional(field_info.annotation)
    if typing.get_origin(annotation) is list:
        args = typing.get_args(annotation)
        if args:
            return args[0]
    return annotation


def _check_composable_fields(
    cls: type,
    values: Dict[str, Any],
) -> Dict[str, Any]:
    """Check that the type of the field is allowed in the section."""
    for attr, value in values.items():
        if issubclass(type(value), BaseModelCardField) and (
            value.composable_with is None
            or (
                value.composable_with != "Any"
                and cls.__name__ not in value.composable_with
            )
        ):
            raise ValueError(
                f"Field `{attr}`(type={type(value)}) is not allowed in "
                f"`{cls.__name__}` section.",
            )

    return values


# Shared config for model card fields and sections.
# - extra: allow extra attributes on the field/section object.
# - validate_default: validate default values.
# - validate_assignment: validate attribute assignments.
# - json_encoders: custom JSON encoders.
_MODEL_CARD_CONFIG = ConfigDict(
    extra="allow",
    validate_default=True,
    validate_assignment=True,
    json_encoders={np.ndarray: lambda v: v.tolist()},
)


class BaseModelCardField(BaseModel):
    """Base class for model card fields.

    Class Parameters
    ----------------
    composable_with : Literal["Any"], List[str], optional
        The sections this field can be dynamically composed with. If "Any",
        the field can be composed with any subclass of `BaseModelCardSection`
        or `BaseModelCardField`. If None, the field cannot be dynamically
        composed with any other fields or sections - it must be explicitly
        added to a section. If a list of strings, the strings are the names
        treated as class names for subclasses of `BaseModelCardSection` or
        `BaseModelCardField`.
    list_factory : bool, default=False
        Whether multiple instances of this field can be added to a section
        in a list.

    """

    composable_with: ClassVar[Optional[Union[Literal["Any"], List[str]]]] = "Any"
    list_factory: ClassVar[bool] = False

    model_config = _MODEL_CARD_CONFIG

    def __init_subclass__(
        cls,
        composable_with: Optional[Union[Literal["Any"], List[str]]] = "Any",
        list_factory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Capture the `composable_with`/`list_factory` class arguments."""
        super().__init_subclass__(**kwargs)
        cls.composable_with = composable_with
        cls.list_factory = list_factory

    @model_validator(mode="before")
    @classmethod
    def _validate_composition(cls, values: Any) -> Any:
        if isinstance(values, dict):
            return _check_composable_fields(cls, values)
        return values


class BaseModelCardSection(BaseModel):
    """Base class for model card sections."""

    model_config = _MODEL_CARD_CONFIG

    @model_validator(mode="before")
    @classmethod
    def _validate_composition(cls, values: Any) -> Any:
        if isinstance(values, dict):
            return _check_composable_fields(cls, values)
        return values

    def update_field(self, name: str, value: Any) -> None:
        """Update the field with the given name to the given value.

        Appends to the field if it is a list.

        Parameters
        ----------
        name : str
            Name of the field to update.
        value : Any
            Value to update the field to.

        Raises
        ------
        ValueError
            If the field does not exist.

        """
        if name not in type(self).model_fields:
            raise ValueError(f"Field {name} does not exist.")

        field = type(self).model_fields[name]
        if field.default_factory == list or isinstance(getattr(self, name), list):  # noqa: E721
            item_type = _get_list_item_type(field)
            if (
                isinstance(value, BaseModel)
                and isinstance(item_type, type)
                and not isinstance(value, item_type)
            ):
                # pydantic v2 does not automatically coerce an arbitrary
                # BaseModel instance into a differently-typed nested model
                # (unlike pydantic v1); dump it to a dict first so the list
                # validation below can construct the expected item type.
                value = value.model_dump()

            # NOTE: pydantic does not trigger validation when appending to a list,
            # but if `validate_assignment` is set to `True`, then validation will
            # be triggered when the list is assigned to the field.
            field_values = getattr(self, name, [])
            field_values.append(value)
            setattr(self, name, field_values)  # trigger validation
        else:
            setattr(self, name, value)

    def add_field(self, name: str, value: Any) -> None:
        """Dynamically add a field to the section.

        Parameters
        ----------
        name : str
            Name of the field to add.
        value : Any
            Value to add to the field.

        Raises
        ------
        ValueError
            If the field name is not a valid Python identifier.

        """
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(
                f"Expected `field_name` to be a valid Python identifier."
                f" Got {name} instead.",
            )

        if isinstance(value, BaseModelCardField) and value.list_factory is True:
            value = [value]  # add field as a list

        # with `extra="allow"`, assigning a name that is not a declared model
        # field stores it as an extra attribute, which is included in
        # `model_dump`/`model_dump_json` output like any other field
        setattr(self, name, value)
