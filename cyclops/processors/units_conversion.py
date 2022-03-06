"""[WIP] Units conversion module."""

import re

import numpy as np

from cyclops.processors.string_ops import normalize_special_characters


def get_scale(be_like, actual):
    """Scale every measurement to a standard unit."""
    replacements = {
        "milliliters": "ml",
        "millimeters": "mm",
        "gm": "g",
        "x10 ": "x10e",
        "tril": "x10e12",
        "bil": "x10e9",
    }
    if isinstance(be_like, str) & isinstance(actual, str):
        # check if anything should be replaced:
        for k, v in replacements.items():
            be_like = be_like.replace(k, v)
            actual = actual.replace(k, v)
        scale = 1

        # check if both have x10^X terms in them
        multipliers = ["x10e6", "x10e9", "x10e12"]
        if any(item in be_like for item in multipliers) and any(
            item in actual for item in multipliers
        ):
            # then adjust
            scale *= 1000 ** -multipliers.index(
                re.search(r"x10e\d+", actual)[0]
            ) * 1000 ** multipliers.index(re.search(r"x10e\d+", be_like)[0])
            return scale

        be_like_list = be_like.split(
            "/"
        )  # split the numerator and denominators for the comparator units
        actual_list = actual.split(
            "/"
        )  # split the numerator and denominators for the units to be converted
        if len(be_like_list) == len(actual_list):
            success = 1
            for i, be_like_item in enumerate(be_like_list):
                try:
                    scale *= convert_unit(actual_list[i], be_like_item) ** (
                        1 if i > 0 else -1
                    )
                except Exception:
                    success = 0
                    # could not convert between units
                    break
            if success:
                return scale
    return "could not convert"


def clean(item_name):
    """[TODO]: Add docstring."""
    return str(item_name).replace("%", "%%").replace(
        "'", "''"
    ), normalize_special_characters(str(item_name))


def convert_unit(from_unit, to_unit):
    """Convert units found in labs.

    Parameters
    ----------
    from_unit: str
        The unit that we are trying to convert.
    to_unit: str
        The target to convert to.

    Returns
    -------
    int
        The scale of one from_unit to to_unit.
    """
    if from_unit == to_unit:
        return 1  # scale is 1
    # create a conversion matrix cmat
    prefixes = ["p", "n", "u", "m", "", "k"]
    c_mat = np.concatenate(
        [
            np.logspace(-i * 3, (-i + len(prefixes) - 1) * 3, len(prefixes)).reshape(
                1, -1
            )
            for i in range(len(prefixes))
        ],
        axis=0,
    )

    for base_unit in ["g", "mol", "l"]:  # this order is safe
        # handle mismatched baseunits
        if (base_unit in from_unit) & (base_unit in to_unit):
            units = [item + base_unit for item in prefixes]
            assert from_unit in units, f"{from_unit} is not in {units}"
            assert to_unit in units, f"{to_unit} is not in {units}"
            return c_mat[
                units.index(from_unit), units.index(to_unit)
            ]  # scale is the result of this
        elif (base_unit in from_unit) + (base_unit in to_unit) == 1:
            raise Exception(f"Base units do not match for {from_unit} and {to_unit}")
    raise Exception(
        f"Either {from_unit} or {to_unit} are not multiples of ['g', 'mol', 'l']"
    )


def convert_units(units_dict):
    """[TODO]: Add docstring."""
    conversion_list = []
    for k, v in units_dict.items():
        if len(v) > 1:
            for item in v[1:]:
                scale = get_scale(v[0][0], item[0])
                if not isinstance(scale, str):
                    conversion_list.append(
                        (
                            k[0],
                            k[1],
                            item[0],
                            get_scale(v[0][0], item[0]),
                            v[0][0],
                            item[1],
                        )
                    )  # key: (original unit, scale, to_unit)
    return conversion_list
