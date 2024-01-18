from typing import Dict, Hashable, List, Optional
import warnings
from copy import deepcopy
from dataclasses import dataclass, field

import datetime
from datetime import timedelta

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN

from fecg.utils.common import to_list_optional
from fecg.utils.dates.dates import datetime_to_unix, has_time
from fecg.utils.pairs import (
    get_pairs,
    pairs_to_groups,
    split_pairs,
)
from fecg.utils.pandas.groupby import groupby_agg_mode
from fecg.utils.pandas.join import reset_index_merge
from fecg.utils.pandas.index import (
    index_structure_equal,
    is_multiindex,
)
from fecg.utils.pandas.pandas import check_cols, combine_nonoverlapping, or_conditions
from fecg.utils.pandas.type import is_datetime_series


def cluster_date_group(dates, dbscan):
    dbscan.fit(dates.values.reshape(-1, 1))

    return pd.Series(dbscan.labels_)


def cluster_dates(dates, dbscan: DBSCAN):
    # Convert to Unix for clustering
    unix = datetime_to_unix(dates)

    # Create clusters for each group
    clusters = unix.groupby(level=0).apply(cluster_date_group, dbscan)

    clusters.index = clusters.index.droplevel(1)
    clusters = clusters.replace({-1: np.nan}).astype("Int64")

    return clusters


def get_date_clusters(dates, max_neighbourhood_delta: datetime.timedelta):
    check_cols(dates, ["date", "approx"], raise_err_on_missing=True)

    dbscan = DBSCAN(
        eps=max_neighbourhood_delta.total_seconds(),
        min_samples=2,
    )
    clusters = cluster_dates(dates["date"], dbscan)
    clusters.rename("cluster", inplace=True)

    # Combine into the original data
    clusters = pd.concat([dates, clusters], axis=1)

    return clusters


def cluster_analysis(unres_hard, clusters):
    index_col = clusters.index.names

    # Get the max cluster size for each group
    cluster_size = clusters.reset_index().groupby(index_col + ["cluster"]).size()
    cluster_size.rename("cluster_size", inplace=True)

    max_sizes = cluster_size.groupby(level=0).agg("max")

    clusters_of_max_size = reset_index_merge(
        cluster_size,
        max_sizes,
        on=index_col + ["cluster_size"],
        how="inner",
        index_col=index_col,
    )["cluster"]
    clusters_of_max_size
    clusters_of_max_size = clusters_of_max_size.to_frame()
    clusters_of_max_size["is_max_size"] = True

    # The below averaging methods only make sense if there is a single max cluster,
    # so ignore groups with several clusters of same size
    clusters_of_max_size_vcs = clusters_of_max_size.index.value_counts()

    clusters_of_max_size = clusters_of_max_size[~clusters_of_max_size.index.isin(
        clusters_of_max_size_vcs.index[clusters_of_max_size_vcs > 1]
    )]

    # Get the is_max_size column into clusters
    clusters = reset_index_merge(
        clusters,
        clusters_of_max_size,
        how="left",
        on=index_col + ["cluster"],
        index_col=index_col,
    )
    clusters["is_max_size"].fillna(False, inplace=True)

    # Get only the dates in the largest cluster
    clusters_largest = clusters[clusters["is_max_size"]]

    # Get the hard dates in the largest clusters
    clusters_largest_hard = clusters_largest[~clusters_largest["approx"]]

#     # === Resolve: largest_cluster_hard_mode
#     single_modes = groupby_agg_mode(
#         unres_hard["date"].groupby(level=0),
#         single_modes_only=True,
#     )

#     largest_hard_is_mode = clusters_largest_hard.index.isin(single_modes.index)
#     largest_cluster_hard_mode = clusters_largest_hard[largest_hard_is_mode]["date"]

#     # Continue without the resolved ones
#     clusters_largest_hard = clusters_largest_hard[~largest_hard_is_mode]

    # === Resolve: largest_cluster_hard_mean ===
    # Take the average of these largest cluster hard dates
    largest_cluster_hard_mean = clusters_largest_hard.reset_index(
    ).groupby(index_col + ["cluster"])["date"].agg("mean")
    largest_cluster_hard_mean.index = largest_cluster_hard_mean.index.droplevel(1)

    # === Resolve: largest_cluster_approx_mean ===
    # Now consider the largest clusters which have only approximate values
    all_approx = clusters_largest.groupby(level=0)["approx"].all()

    clusters_largest_approx = clusters_largest[
        clusters_largest.index.isin(all_approx.index[all_approx])
    ].copy()

    largest_cluster_approx_mean = clusters_largest_approx.groupby(
        index_col + ["cluster"],
    )["date"].agg("mean")
    largest_cluster_approx_mean.index = largest_cluster_approx_mean.index.droplevel(1)

    return clusters_largest, largest_cluster_hard_mean, largest_cluster_approx_mean


def analyze_typos(dates_hard):
    index_col = dates_hard.index.names

    # Get all unique hard dates for each group
    dates_hard_unique = dates_hard["date"].reset_index().value_counts(
    ).reset_index().drop(0, axis=1).set_index(index_col)["date"]

    # Ignore any groups which only have one unique hard date
    dates_hard_unique_vcs = dates_hard_unique.index.value_counts()
    dates_hard_unique_vcs = dates_hard_unique_vcs[dates_hard_unique_vcs > 1]
    dates_hard_unique_vcs.rename("n_unique", inplace=True)

    dates_hard_unique = dates_hard_unique.loc[dates_hard_unique_vcs.index]

    def date_to_char(dates):
        chars = dates.astype(str).str.split('', expand=True)
        chars.drop(columns=[0, 5, 8, 11], inplace=True)
        chars.rename({
            1: 'y1',
            2: 'y2',
            3: 'y3',
            4: 'y4',
            6: 'm1',
            7: 'm2',
            9: 'd1',
            10: 'd2',
        }, axis=1, inplace=True)
        chars = chars.astype('uint8')

        return chars

    # Convert the dates into characters
    chars = date_to_char(dates_hard_unique)

    # Compute hard date character combinations
    pairs = chars.groupby(level=0).apply(get_pairs)
    pairs.index = pairs.index.droplevel(1)
    pairs.index.names = index_col

    pairs_x, pairs_y = split_pairs(pairs)

    # Calculate equal characters
    pairs_eq = pairs_x == pairs_y
    pairs_eq = pairs_eq.add_suffix("_eq")
    pairs_eq["n_diff"] = 8 - pairs_eq.sum(axis=1)

    # Calculate adjacent characters, e.g., 5 vs 6 or 2 vs 1
    # Convert from uint8 to int to avoid rounding issues
    pairs_adj = (pairs_x.astype(int) - pairs_y.astype(int)).abs() == 1
    pairs_adj = pairs_adj.add_suffix("_adj")
    pairs_adj["n_adj"] = pairs_adj.sum(axis=1)

    # Collect information about the typo pairs
    pairs = pd.concat([pairs_eq, pairs_adj], axis=1)

    # Incorporate date info
    # Recover the dates from the characters
    date_x = pairs_x.astype(str).agg(''.join, axis=1)
    date_x = date_x.str.slice(stop=4) + \
        "-" + date_x.str.slice(start=4, stop=6) + \
        "-" + date_x.str.slice(start=6)

    date_y = pairs_y.astype(str).agg(''.join, axis=1)
    date_y = date_y.str.slice(stop=4) + \
        "-" + date_y.str.slice(start=4, stop=6) + \
        "-" + date_y.str.slice(start=6)
    pairs["date_x"] = pd.to_datetime(date_x)
    pairs["date_y"] = pd.to_datetime(date_y)
    pairs["year"] = pairs["date_x"].dt.year == pairs["date_y"].dt.year
    pairs["month"] = pairs["date_x"].dt.month == pairs["date_y"].dt.month
    pairs["day"] = pairs["date_x"].dt.day == pairs["date_y"].dt.day

    # Check if gotten the day/month transposed
    pairs["dm_transpose"] = (pairs["date_x"].dt.day == pairs["date_y"].dt.month) & (pairs["date_x"].dt.month == pairs["date_y"].dt.day)

    # Logic for determining whether a typo or not
    certain_conds = [
        # Only one different character
        (pairs["n_diff"] == 1),

        # Two different characters with at least one adjacent
        ((pairs["n_diff"] == 2) & (pairs["n_adj"] >= 1)),

        # Day and month are transposed, but correct year
        (pairs["dm_transpose"] & pairs["year"]),
    ]
    pairs["typo_certain"] = or_conditions(certain_conds)

    pairs["typo_possible"] = pairs["n_diff"] <= 3

    # Create typo groups from pairs of possible typos
    typo_pairs = pairs[pairs["typo_certain"] | pairs["typo_possible"]]

    typo_groups = typo_pairs[["date_x", "date_y"]].astype(str).groupby(level=0).apply(
        pairs_to_groups
    ).reset_index().set_index(index_col + ["group"])["level_1"]
    typo_groups.rename("date", inplace=True)

    # Convert typos to characters
    typo_group_chars = date_to_char(typo_groups)

    def mode_scalar_or_list(series):
        mode = pd.Series.mode(series)

        if len(mode) > 1:
            return mode.to_list()

        return mode

    # Compile the most popular character options seen in each typo group
    typo_value_options = typo_group_chars.groupby(level=[0, 1]).agg(
        dict(zip(typo_group_chars.columns, [mode_scalar_or_list]*len(typo_group_chars.columns)))
    )

    """
    LEFT TO DO:
    Compile a "date_possible" object
    - Any completely filled typo_value_options (no lists) are essentially solved
    - For day/month transpositions, those would be two possible dates [1914-11-03, 1914-03-11]
    Still need to check out letter transpositions - 1956-10-02 vs 1956-10-20
    Perhaps do a mean for the one day/ten day/one month cols? The user can specify what's allowed?
    - Trade off between accuracy and just having nulls instead - date accuracy importance is use case specific

    As we go down the line of columns, disagreements become less and less important
    That means we could take a mean of two disagreeing days, but not years, or 
    thousands of years
    """

    return pairs, typo_pairs, typo_groups, typo_value_options


@dataclass
class DateReconcilerResults:
    index_col: List[Hashable]
    resolved: pd.DataFrame
    dates: pd.DataFrame
    dates_hard: pd.DataFrame
    dates_approx: pd.DataFrame
    groups: pd.DataFrame
    unres: pd.DataFrame
    unres_hard: pd.DataFrame
    unres_approx: pd.DataFrame
    unres_groups: pd.DataFrame
    clusters_largest: pd.DataFrame
    pairs: pd.DataFrame
    typo_pairs: pd.DataFrame
    typo_groups: pd.Series
    typo_value_options: pd.DataFrame


class DateReconciler:
    """

    Notes
    -----
    === Resolutions ===
    - one_entry: Group contains one entry - select this date
    - one_date: Contains multiple entries, but one unique date value - select this date
    - one_multi_hard: Group which contains multiple of the same hard dates, but not
        multiple sets of them, e.g., two instances of 1988-03-09 and two of 1974-06-20.
        Works since it's unlikely for a typo or system error to produce the same date.
    - hard_single_mode: Groups containing one hard date mode.
    ### - largest_cluster_hard_mode: If after clustering, only one cluster of max size is
    ###    found, then take the mode of the hard dates, provided there is just one mode.
    - largest_cluster_hard_mean: From the previous case, if more than one mode, then
        take the average all of the hard dates in that cluster.
    - largest_cluster_approx_mean: Same scenario as above, except the largest cluster
        had no hard dates, so instead take the average of the approx dates.

    === Hard vs approximate dates ===
    One important distinction is whether a date is approximate (approx) or not:
    - Approx: Computed, rounded, etc. - close to the real date, but maybe not equal
      (e.g., only the year was given, or computing DOB from age and event time)
    - Hard: System-defined or hand-inputted dates - these should be the true date,
      with the exception of system errors and typos

    Delta distances are computed for both hard and approx dates, but Levenshtein
    distance is only computed for hard dates.

    Approx dates take on supporting roles, e.g., is a given hard date near to many
    supporting approx dates, or can be used as a backup with no hard dates available.
    """
    def __init__(
        self,
        sources: Dict[Hashable, pd.Series],
        date_score_fn: callable,
        approx_sources: Optional[List[Hashable]] = None,
        approx_near_thresh: Optional[timedelta] = None,
        once_per_source: bool = True,
    ):
        """
        sources : dict
            Dictionary of datetime Series, where the key indicates the source.
        date_score_fn : callable
            A function which accepts a returns float between 0 and 1, where this value
            represents the score (feasibility) of the date.
        approx_sources : list of hashable, optional
            Sources where the dates have been approximated - rounded, calculated, etc.
        approx_near_thresh: datetime.timedelta, optional
            Threshold for considering approximated sources to be the same. Must be
            specified if there are any approximate sources.
        once_per_source : bool, default True
            Consider a unique index/date pair only once per source. Helpful for
            ensuring that sources with more/repeated entries don't hold more weight
        """
        # Handle approximate date sources
        if approx_sources is not None and approx_near_thresh is None:
            raise ValueError(
                "Must specify `approx_near_thresh` if `approx_sources` specified."
            )
        approx_sources = to_list_optional(approx_sources, none_to_empty=True)

        if not set(approx_sources).issubset(set(sources.keys())):
            raise ValueError(
                "`approx_sources` must be a subset of the `sources` keys."
            )

        self.dates = self._preproc_sources(sources, approx_sources, once_per_source)
        self.date_score_fn = date_score_fn

        self.approx_sources = approx_sources
        self.approx_near_thresh = approx_near_thresh


    def _preproc_sources(self, sources, approx_sources, once_per_source):
        # Preprocess the sources/dates
        dates = []
        prev_source = None

        for source, date in deepcopy(sources).items():
            try:
                # Confirm datetime dtype
                is_datetime_series(date, raise_err=True)

                # Raise an error if having a multiindex
                is_multiindex(
                    sources[list(sources.keys())[0]].index,
                    raise_err_multi=True,
                )

                # Confirm identical index structures
                if prev_source is not None:
                    index_structure_equal(
                        date.index,
                        sources[prev_source].index,
                        raise_err=True,
                    )

                # No dates can have times - it messes things up
                has_time(date, raise_err_on_time=True)

            except Exception as exc:
                raise ValueError(f"Issue with series - source {source}.") from exc

            date.dropna(inplace=True)
            date.rename("date", inplace=True)

            if once_per_source:
                index_col = date.index.names
                date = date.reset_index().drop_duplicates(
                    keep="first",
                ).set_index(index_col)["date"]

            date = date.to_frame()
            date["source"] = source
            date["approx"] = source in approx_sources

            dates.append(date)
            prev_source = source

        dates = pd.concat(dates)
        dates = dates[~dates.index.isna()]
        dates.sort_index(inplace=True)

        if not (dates["date"].dt.time == datetime.time(0)).all():
            warnings.warn(
                "Dates with times are not supported. Converting to date only."
            )

        return dates


    def _combined_resolved(self, groups, groups_resolved):
        resolved = []
        for reason, dates in groups_resolved.items():
            dates = dates.to_frame()
            dates["reason"] = reason
            dates = dates.reindex(groups.index)
            resolved.append(dates)

        return combine_nonoverlapping(resolved)


    def __call__(self):
        dates = self.dates.copy()

        index_col = list(dates.index.names)

        dates["date_str"] = dates["date"].astype(str)
        dates["date_score"] = dates["date"].apply(self.date_score_fn)

        # Split into approximate and hard dates
        dates_approx = dates[dates["approx"]].drop("approx", axis=1)
        dates_hard = dates[~dates["approx"]].drop("approx", axis=1)

        groups = dates.groupby(dates.index).size().rename("size").to_frame()
        groups["one_entry"] = groups["size"] == 1
        groups["n_approx"] = dates_approx.groupby(dates_approx.index).size()
        groups["n_approx"].fillna(0, inplace=True)

        # Groups are resolved on a case-by-case basis. Once resolved, they can be
        # ignored to avoid wasted computation. The unresolved (unres) dates/groups
        # will continue to be analyzed.
        unres = dates.copy()
        unres_hard = dates_hard.copy()
        unres_approx = dates_approx.copy()
        unres_groups = groups.copy()

        # Find and analyze typos in the hard dates
        pairs, typo_pairs, typo_groups, typo_value_options = analyze_typos(dates_hard)

        # Having extracted the typo information, drop any impossible dates (score = 0)
        # which might later confuse the analysis
        unres = unres[unres["date_score"] != 0]
        unres_hard = unres_hard[unres_hard["date_score"] != 0]
        unres_approx = unres_approx[unres_approx["date_score"] != 0]

        groups_resolved = {}
        def resolve(resolved, reason):
            nonlocal groups_resolved, unres, unres_hard, unres_approx, unres_groups

            groups_resolved[reason] = resolved

            unres = unres[
                ~unres.index.isin(resolved.index)
            ]
            unres_hard = unres_hard[
                ~unres_hard.index.isin(resolved.index)
            ]
            unres_approx = unres_approx[
                ~unres_approx.index.isin(resolved.index)
            ]
            unres_groups = unres_groups[
                ~unres_groups.index.isin(resolved.index)
            ]

        # === Resolve: one_entry ===
        one_entry = unres[
            unres.index.isin(unres_groups.index[unres_groups["size"] == 1])
        ]["date"]
        resolve(one_entry, "one_entry")

        # === Resolve: one_date ===
        vcs = unres["date"].reset_index().value_counts()
        vcs.rename("count", inplace=True)

        # Iff a given row has a count equal to its group size, then only one unique date
        instance_compare = vcs.reset_index().join(groups, how="left", on="research_id")
        instance_compare.set_index(index_col, inplace=True)
        one_date_cond = instance_compare["count"] == instance_compare["size"]
        one_date = instance_compare[one_date_cond]["date"]
        resolve(one_date, "one_date")

        # === Resolve: one_multi_hard ===
        # For each group, determine the hard dates which appear more than once
        vcs_hard = unres_hard["date"].reset_index().value_counts()
        vcs_hard_multi = vcs_hard[vcs_hard > 1]

        # Get the groups which only have a single set of these same hard dates
        # Otherwise, it may be ambiguous as to which set is the right one
        is_multi_one = vcs_hard_multi.index.droplevel(1).value_counts()
        is_multi_one = is_multi_one[is_multi_one == 1]

        one_multi_hard = vcs_hard_multi.reset_index().set_index(index_col)["date"]
        one_multi_hard = one_multi_hard.loc[is_multi_one.index]

        resolve(one_multi_hard, "one_multi_hard")

        # === Resolve: hard_single_mode ===
        hard_single_mode = groupby_agg_mode(
            unres_hard["date"].groupby(level=0),
            single_modes_only=True,
        )
        resolve(hard_single_mode, "hard_single_mode")


        # === Cluster resolutions ===
        clusters = get_date_clusters(
            unres[["date", "approx"]],
            self.approx_near_thresh,
        )

        clusters_largest, largest_cluster_hard_mean, largest_cluster_approx_mean = \
            cluster_analysis(unres_hard, clusters)

        resolve(largest_cluster_hard_mean, "largest_cluster_hard_mean")
        resolve(largest_cluster_approx_mean, "largest_cluster_approx_mean")

        # Combine all of the resolved data collected into a single DataFrame
        resolved = self._combined_resolved(groups, groups_resolved)

        return DateReconcilerResults(
            index_col=index_col,
            resolved=resolved,
            dates=dates,
            dates_hard=dates_hard,
            dates_approx=dates_approx,
            groups=groups,
            unres=unres,
            unres_hard=unres_hard,
            unres_approx=unres_approx,
            unres_groups=unres_groups,
            clusters_largest=clusters_largest,
            pairs=pairs,
            typo_pairs=typo_pairs,
            typo_groups=typo_groups,
            typo_value_options=typo_value_options,
        )
