"""Base classes for model card fields and sections."""

import keyword
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
from pydantic import BaseConfig, BaseModel, Extra, root_validator
from pydantic.fields import FieldInfo, ModelField


def _check_composable_fields(
    cls: BaseModel,
    values: Dict[str, Any],
) -> Dict[str, Any]:
    """Check that the type of the field is allowed in the section."""
    for attr, value in values.items():
        if issubclass(type(value), BaseModelCardField) and (
            value.__config__.composable_with is None
            or (
                value.__config__.composable_with != "Any"
                and cls.__name__ not in value.__config__.composable_with  # type: ignore [attr-defined] # noqa: E501
            )
        ):
            print(issubclass(type(value), BaseModelCardField))
            print(value.__config__.composable_with)
            raise ValueError(
                f"Field `{attr}`(type={type(value)}) is not allowed in "
                f"`{cls.__name__}` section.",  # type: ignore[attr-defined]
            )

    return values


class BaseModelCardConfig(BaseConfig):
    """Global config for model card fields.

    Attributes
    ----------
        extra : Extra
            Whether to allow extra attributes in the field object.
        smart_union : bool
           Whether `Union` should check all allowed types before even trying to
           coerce.
        validate_all : bool
            Whether to validate all attributes in the field object.
        validate_assignment : bool
            Whether to validate assignments of attributes in the field object.
        json_encoders : Dict[Any, Callable]
            Custom JSON encoders.

    """

    extra: Extra = Extra.allow
    smart_union: bool = True
    validate_all: bool = True
    validate_assignment: bool = True
    json_encoders: Dict[Any, Callable[..., Any]] = {np.ndarray: lambda v: v.tolist()}


class BaseModelCardField(BaseModel):
    """Base class for model card fields."""

    class Config(BaseModelCardConfig):
        """Global config for model card fields.

        Attributes
        ----------
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

        composable_with: Optional[Union[Literal["Any"], List[str]]] = "Any"
        list_factory: bool = False

    _validate_composition = root_validator(pre=True, allow_reuse=True)(
        _check_composable_fields,
    )


class BaseModelCardSection(BaseModel):
    """Base class for model card sections."""

    Config = BaseModelCardConfig

    _validate_composition = root_validator(pre=True, allow_reuse=True)(
        _check_composable_fields,
    )

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
        if name not in self.__fields__:
            raise ValueError(f"Field {name} does not exist.")

        field = self.__fields__[name]
        if field.default_factory == list or isinstance(getattr(self, name), list):  # noqa: E721
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

        type_ = type(value)
        default_factory = None
        if isinstance(value, list):
            default_factory = list
        if (
            isinstance(value, BaseModelCardField)
            and value.__config__.list_factory is True  # type: ignore[attr-defined]
        ):
            default_factory = list
            value = [value]  # add field as a list
            type_ = List[type_]  # type: ignore[valid-type]

        setattr(self, name, value)

        # modify __fields__ to include new field
        self.__fields__[name] = ModelField(
            name=name,
            type_=type_,
            required=False,
            class_validators=None,
            model_config=BaseModelCardField.Config,
            default_factory=default_factory,
            field_info=FieldInfo(unique_items=True)
            if default_factory == list  # noqa: E721
            else None,
        )
