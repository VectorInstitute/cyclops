"""Utility functions used in data extraction."""

from collections import Counter
from typing import Iterable, List, Union

import numpy as np
import pandas as pd
import re

from cyclops.processors.constants import TRAJECTORIES


def normalize_special_characters(item: str) -> str:
    """Replace special characters with string equivalents.

    Parameters
    ----------
    item: str
        Input string.

    Returns
    -------
    str
        Output string after normalizing.
    """
    replacements = {
        "(": " ",
        ")": " ",
        ",": " ",
        "%": " percent ",
        "+": " plus ",
        "#": " number ",
        "&": " and ",
        "'s": "",
        "/": " per ",
    }
    for replacee, replacement in replacements.items():
        item = item.replace(replacee, replacement)

    item = item.strip()
    item = re.sub(r"\s+", "_", item)
    item = re.sub(r"[^0-9a-z_()]+", "_", item)
    item = re.sub(r"(?s:(^[0-9_].+))", "a_\1", item)
    return item


def count_occurrences(items: Iterable) -> List:
    """Count number of occurrences of the items.

    Parameters
    ----------
    items: Iterable
        Iterable of items to count the number of repeated values.

    Returns
    -------
    List
        (item, count) ordered by count, descending order.
    """
    counter = Counter(items)
    return sorted(counter.items(), key=lambda x: x[1], reverse=True)


def convert_to_numeric(x):
    """Convert different strings to numeric values.

    Parameters
    ----------
    x: str
        Input string.

    Returns
    -------
    Union[int, float]
        Converted numeric output.
    """
    if x in (None, np.nan):
        return np.nan
    if not isinstance(x, str):
        # Originally this case implicitly returned None.
        raise TypeError(f"Expected string, received {type(x)}")

    if is_range(x):
        try:
            return compute_range_avg(x)
        except Exception:
            print(x)
            raise
    return re.sub("^-?[^0-9.]", "", str(x))


def is_range(x: str) -> bool:
    """Test if x matches range pattern.

    e.g. "2 to 5" or "2 - 5"

    Parameters
    ----------
    x: str
        Input string to test if its a range.

    Returns
    -------
    bool
        True if categorical, False otherwise.
    """
    # [TODO:] Why is a space required? Isn't 1-5 all right, too?
    categorical_pattern = re.compile(r"-?\d+\s+(?:to|-)\s+(-?\d+)")
    return categorical_pattern.search(x) is not None


def compute_range_avg(item: str) -> Union[int, float]:
    """Compute the average of a range.

    For instance, 5 - 7 -> 6, and 1 - 4 -> 2.5

    Parameters
    ----------
    item: str
        Input string which mentions a range.

    Returns
    -------
    Union[int, float]
        Computed average of range.
    """
    pattern_str = r"^(?P<first>-?\d+)\s*(?:to|-)\s*(?P<second>-?\d+)$"
    pattern = re.compile(pattern_str)
    if not (matched := pattern.search(item)):
        raise ValueError(f"'item' does not match expected pattern {pattern_str}")
    return (int(matched.group("first")) + int(matched.group("second"))) / 2


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
            for i in range(len(be_like_list)):
                try:
                    scale *= convert_unit(actual_list[i], be_like_list[i]) ** (
                        1 if i > 0 else -1
                    )
                except Exception:
                    success = 0
                    # could not convert between units
                    break
            if success:
                return scale
    return "could not convert"


def simple_imput(mean_vals):
    """[TODO]: Add docstring."""
    idx = pd.IndexSlice
    # do simple imputation
    # mask
    mask = 1 - mean_vals.isna()
    # measurement
    measurement = mean_vals.copy()

    # these expressions necessarily need to be executed seperately
    subset_data = measurement.loc[idx[:, :, 0], :]
    data_means = measurement.mean()
    subset_data = subset_data.fillna(data_means)
    measurement.loc[idx[:, :, 0], :] = subset_data.values

    measurement = measurement.ffill()
    # time_since
    is_absent = 1 - mask
    hours_of_absence = is_absent.groupby(["patient_id", "genc_id"]).cumsum()
    time_df = hours_of_absence - hours_of_absence[is_absent == 0].fillna(method="ffill")
    time_df = time_df.fillna(0)

    final_data = pd.concat(
        [measurement, mask, time_df], keys=["measurement", "mask", "time"], axis=1
    )
    final_data.columns = final_data.columns.swaplevel(0, 1)
    final_data.sort_index(axis="columns", inplace=True)

    nancols = 0

    try:
        nancols = np.sum(
            [a == 0 for a in final_data.loc[:, idx[:, "mask"]].sum().values]
        )
        print(nancols)
    except Exception:
        print("could not get nancols")
        pass

    print(nancols, "/", len(sorted(set(final_data.columns.get_level_values(0)))))
    return final_data


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


def get_category(code, trajectories=TRAJECTORIES):
    """Get ICD10 category.

    df['ICD10'].apply(get_category, args=(trajectories,))
    """
    if code is None:
        return np.nan
    try:
        code = str(code)
    except Exception:
        return np.nan
    for item, value in trajectories.items():
        # check that code is greater than value_1
        if re.sub("[^a-zA-Z]", "", code).upper() > value[0][0].upper():
            # example, code is T and comparator is S
            pass
        elif (re.sub("[^a-zA-Z]", "", code).upper() == value[0][0].upper()) and (
            float(insert_decimal(re.sub("[^0-9]", "", code), index=2))
            >= int(value[0][1:])
        ):
            # example S21 > s00
            pass
        else:
            continue

        # check that code is less than value_2
        if re.sub("[^a-zA-Z]", "", code).upper() < value[1][0].upper():
            # example, code is S and comparator is T
            #             print(value[0], code, value[1])
            return "_".join(value)
        elif (re.sub("[^a-zA-Z]", "", code).upper() == value[1][0].upper()) and (
            int(float(insert_decimal(re.sub("[^0-9]", "", code), index=2)))
            <= int(value[1][1:])
        ):
            # example S21 > s00
            #             print(value[0], code, value[1])
            return "_".join(value)
        else:
            continue
    raise Exception("Code cannot be converted: {}".format(code))


def transform_diagnosis(data):
    """Apply categorical ICD10 filters and encode as one-hot vector."""
    data = pd.concat(
        (
            data,
            pd.get_dummies(
                data.loc[:, "mr_diagnosis"].apply(get_category, args=(TRAJECTORIES,)),
                dummy_na=True,
                columns=TRAJECTORIES.keys(),
                prefix="icd10",
            ),
        ),
        axis=1,
    )
    return data


def insert_decimal(string, index=2):
    """[TODO]: Add docstring."""
    return string[:index] + "." + string[index:]


def convert_units(units_dict):
    """[TODO]: Add docstring."""
    conversion_list = []
    for k, v in units_dict.items():
        if len(v) > 1:
            for item in v[1:]:
                scale = get_scale(v[0][0], item[0])
                if not (isinstance(scale, str)):
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
