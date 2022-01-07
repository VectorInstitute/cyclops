import numpy as np
import pandas as pd
import re

# ---------------------------constants -----------------------
HOSPITAL_ID = {
    "THPM": 0,
    "SBK": 1,
    "UHNTG": 2,
    "SMH": 3,
    "UHNTW": 4,
    "THPC": 5,
    "PMH": 6,
    "MSH": 7,
}
TRAJECTORIES = {
    "Certain infectious and parasitic diseases": ("A00", "B99"),
    "Neoplasms": ("C00", "D49"),
    "Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism": (
        "D50",
        "D89",
    ),
    "Endocrine, nutritional and metabolic diseases": ("E00", "E89"),
    "Mental, Behavioral and Neurodevelopmental disorders": ("F01", "F99"),
    "Diseases of the nervous system": ("G00", "G99"),
    "Diseases of the eye and adnexa": ("H00", "H59"),
    "Diseases of the ear and mastoid process": ("H60", "H95"),
    "Diseases of the circulatory system": ("I00", "I99"),
    "Diseases of the respiratory system": ("J00", "J99"),
    "Diseases of the digestive system": ("K00", "K95"),
    "Diseases of the skin and subcutaneous tissue": ("L00", "L99"),
    "Diseases of the musculoskeletal system and connective tissue": ("M00", "M99"),
    "Diseases of the genitourinary system": ("N00", "N99"),
    "Pregnancy, childbirth and the puerperium": ("O00", "O99"),
    "Certain conditions originating in the perinatal period": ("P00", "P96"),
    "Congenital malformations, deformations and chromosomal abnormalities": (
        "Q00",
        "Q99",
    ),
    "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified": (
        "R00",
        "R99",
    ),
    "Injury, poisoning and certain other consequences of external causes": (
        "S00",
        "T88",
    ),
    "External causes of morbidity": ("V00", "Y99"),
    "COVID19": ("U07", "U08"),
    "Factors influencing health status and contact with health services": (
        "Z00",
        "Z99",
    ),
}

DRUG_SCREEN = [
    "amitriptyline",
    "amphetamine",
    "barbiturates",
    "barbiturates_scn",
    "barbiturates_and_sedatives_blood",
    "benzodiazepine_scn",
    "benzodiazepines_screen",
    "cannabinoids",
    "clozapine",
    "cocaine",
    "cocaine_metabolite",
    "codeine",
    "cocaine_metabolite",
    "codeine_metabolite_urine",
    "desipramine",
    "dextromethorphan",
    "dim_per_dip_metabolite",
    "dimen_per_diphenhydramine",
    "doxepin",
    "ephedrine_per_pseudo",
    "fluoxetine",
    "hydrocodone",
    "hydromorphone",
    "imipramine",
    "lidocaine",
    "mda_urine",
    "mdma_ecstacy",
    "methadone",
    "meperidine_urine",
    "methadone_metabolite_urine",
    "methamphetamine",
    "morphine",
    "morphine_metabolite_urine",
    "nortriptyline",
    "olanzapine_metabolite_u",
    "olanzapine_urine",
    "opiates_urine",
    "oxycodone",
    "oxycodone_cobas",
    "oxycodone_metabolite",
    "phenylpropanolamine",
    "propoxyphene",
    "sertraline",
    "trazodone",
    "trazodone_metabolite",
    "tricyclics_scn",
    "venlafaxine",
    "venlafaxine_metabolite",
]
# --------------------------------- -----------------------


def filter_string(item):
    item = item.replace(")", " ")
    item = item.replace("(", " ")
    item = item.replace("%", " percent ")
    item = item.replace("+", " plus ")
    item = item.replace("#", " number ")
    item = item.replace("&", " and ")
    item = item.replace("'s", "")
    item = item.replace(",", " ")
    item = item.replace("/", " per ")
    item = " ".join(item.split())
    item = item.strip()

    item = item.replace(" ", "_")
    item = re.sub("[^0-9a-z_()]+", "_", item)
    if len(item) > 1:
        if item[0] in "1234567890_":
            item = "a_" + item
    return item


def get_scale(be_like, actual):
    """
    This function is applied to scale every measurement to a standard unit
    """
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
                re.search("x10e\d+", actual)[0]
            ) * 1000 ** multipliers.index(re.search("x10e\d+", be_like)[0])
            return scale

        be_like_list = be_like.split(
            "/"
        )  # split the numerator and denominators for the comparator units
        actual_list = actual.split(
            "/"
        )  # split the numerator and denominators for the units that need to be converted
        if len(be_like_list) == len(actual_list):
            success = 1
            for i in range(len(be_like_list)):
                try:
                    scale *= convert(actual_list[i], be_like_list[i]) ** (
                        1 if i > 0 else -1
                    )
                except:
                    success = 0
                    # could not convert between units
                    break
            if success:
                return scale
    return "could not convert"


def name_count(items):
    """
    Inputs:
        items (list): a list of itemsto count the number of repeating values
    Returns:
        list of tuples with the name of the occurence and the count of each occurence
    """
    all_items = {}
    for item in items:
        if item in all_items.keys():
            all_items[item] += 1
        else:
            all_items[item] = 1
    return sorted([(k, v) for k, v in all_items.items()], key=lambda x: -x[1])


def x_to_numeric(x):
    """
    Handle different strings to convert to numeric values(which can't be done in psql)
    """
    if x is None:
        return np.nan
    elif x is np.nan:
        return np.nan
    elif isinstance(x, str):
        # check if it matches the categorical pattern "2 to 5" or 2 - 5
        if re.search(r"-?\d+ +(to|-) +-?\d+", x) is not None:
            try:
                return numeric_categorical(x)
            except:
                print(x)
                raise
        return re.sub("^-?[^0-9.]", "", str(x))


def numeric_categorical(item):
    x = None
    locals_ = locals()

    item = (
        "(" + item.replace("to", " + ").replace("-", " + ") + ")/2"
    )  # this neglects negative ranges. todo, find a fast regex filter
    items = item.replace("  ", " ").replace("(", "( ").split(" ")
    item = " ".join([i.lstrip("0") for i in items])
    exec("x=" + item, globals(), locals_)
    return locals_["x"]


def simple_imput(mean_vals):
    """ """
    idx = pd.IndexSlice
    # do simple imputation
    # mask
    mask = 1 - mean_vals.isna()
    # measurement
    measurement = mean_vals.copy()

    print(measurement.loc[idx[:, :, 0], :].head())
    print(measurement.loc[idx[:, :, 0], :].values.shape)
    print(measurement.mean().values.shape)

    print(measurement.loc[idx[:, :, 0, :]].groupby("genc_id").count())

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
    except:
        print("could not get nancols")
        pass

    print(nancols, "/", len(sorted(set(final_data.columns.get_level_values(0)))))
    return final_data


def clean(item_name):
    return str(item_name).replace("%", "%%").replace("'", "''"), filter_string(
        str(item_name)
    )


def convert(from_unit, to_unit):
    """
    converts many units found in labs
    Inputs:
        from_unit (str): the unit that we are trying to convert
        to_unit (str): the target to convert to
    Returns:
        (int): the scale of one from_unit to to_unit
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
    """
    Usage:
    df['ICD10'].apply(get_category, args=(trajectories,))
    """
    if code is None:
        return np.nan
    try:
        code = str(code)
    except:
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
    # apply the categorical ICD10 filter and one hot encode:
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
    return string[:index] + "." + string[index:]


def convert_units(units_dict):
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
