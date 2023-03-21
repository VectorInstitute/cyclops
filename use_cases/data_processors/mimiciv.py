"""MIMICIV processor."""

import logging
from os import path
from typing import Callable, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

from cyclops.process.aggregate import (
    Aggregator,
    tabular_as_aggregated,
    timestamp_ffill_agg,
)
from cyclops.process.constants import FEATURES, NUMERIC, ORDINAL, TARGETS
from cyclops.process.feature.feature import TabularFeatures, TemporalFeatures
from cyclops.process.feature.vectorized import (
    Vectorized,
    intersect_vectorized,
    split_vectorized,
)
from cyclops.query import mimiciv as mimic
from cyclops.utils.file import (
    join,
    load_dataframe,
    load_pickle,
    save_dataframe,
    save_pickle,
    yield_dataframes,
    yield_pickled_files,
)
from cyclops.utils.log import setup_logging
from use_cases.util import get_top_events, get_use_case_params, valid_events

# pylint: disable=invalid-name, unnecessary-lambda-assignment, too-many-lines
# pylint: disable=too-many-instance-attributes,

LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


class MIMICIVProcessor:
    """MIMICIV processor."""

    def __init__(self, use_case: str, data_type: str) -> None:
        """Init processor.

        Parameters
        ----------
        use_case : str
            The use-case to process the data for.

        data_type : str
            Type of data (tabular, temporal, or combined).

        """
        self.params = get_use_case_params("mimiciv", use_case)
        self.data_type = data_type
        self._setup_paths()
        self._setup_params()
        self.aggregator = self._init_aggregator()

    ###################
    # Init methods
    ###################

    def _setup_paths(self) -> None:
        """Set up paths and dirs."""
        self.data_dir = self.params.DATA_DIR

        self.queried_dir = self.params.QUERIED_DIR
        self.cleaned_dir = self.params.CLEANED_DIR
        self.aggregated_dir = self.params.AGGREGATED_DIR
        self.vectorized_dir = self.params.VECTORIZED_DIR
        self.final_vectorized = self.params.FINAL_VECTORIZED

        self.tabular_file = self.params.TABULAR_FILE
        self.tab_features_file = self.params.TAB_FEATURES_FILE
        self.tab_slice_file = self.params.TAB_SLICE_FILE
        self.tab_aggregated_file = self.params.TAB_AGGREGATED_FILE
        self.tab_vectorized_file = self.params.TAB_VECTORIZED_FILE
        self.temp_vectorized_file = self.params.TEMP_VECTORIZED_FILE
        self.comb_vectorized_file = self.params.COMB_VECTORIZED_FILE

        self.aligned_path = self.params.ALIGNED_PATH
        self.unaligned_path = self.params.UNALIGNED_PATH

    def _setup_params(self) -> None:
        """Set up the data processing parameters."""
        self.common_feature = (
            self.params.COMMON_FEATURE if self.params.COMMON_FEATURE else None
        )
        self.skip_n = self.params.SKIP_N if self.params.SKIP_N else 0
        self.split_fractions = (
            self.params.SPLIT_FRACTIONS if self.params.SPLIT_FRACTIONS else None
        )
        self.tab_feature_params = (
            self.params.TABULAR_FEATURES if self.params.TABULAR_FEATURES else None
        )
        self.tab_norm_params = (
            self.params.TABULAR_NORM if self.params.TABULAR_NORM else None
        )
        self.tab_slice_params = (
            self.params.TABULAR_SLICE if self.params.TABULAR_SLICE else None
        )
        self.tab_agg_params = (
            self.params.TABULAR_AGG if self.params.TABULAR_AGG else None
        )

        self.temp_params = (
            self.params.TEMPORAL_PARAMS if self.params.TEMPORAL_PARAMS else None
        )
        self.temp_norm_params = (
            self.params.TEMPORAL_NORM if self.params.TEMPORAL_NORM else None
        )
        self.temp_feature_params = (
            self.params.TEMPORAL_FEATURES if self.params.TEMPORAL_FEATURES else None
        )
        self.temp_target_params = (
            self.params.TEMPORAL_TARGETS if self.params.TEMPORAL_TARGETS else None
        )
        self.timestamp_params = (
            self.params.TIMESTAMPS if self.params.TIMESTAMPS else None
        )
        self.timestep_params = self.params.TIMESTEPS if self.params.TIMESTEPS else None
        self.temp_slice_params = (
            self.params.TEMPORAL_SLICE if self.params.TEMPORAL_SLICE else None
        )
        self.temp_agg_params = (
            self.params.TEMPORAL_AGG if self.params.TEMPORAL_AGG else None
        )
        self.temp_impute_params = (
            self.params.TEMPORAL_IMPUTE if self.params.TEMPORAL_IMPUTE else None
        )

    def _init_aggregator(self) -> Aggregator:
        """Initialize the aggregator for temporal and combined processing.

        Returns
        -------
        Aggregator
            The aggregator object.

        """
        return Aggregator(
            aggfuncs=self.temp_agg_params["aggfuncs"],
            timestamp_col=self.temp_agg_params["timestamp_col"],
            time_by=self.temp_agg_params["time_by"],
            agg_by=self.temp_agg_params["agg_by"],
            timestep_size=self.temp_agg_params["timestep_size"],
            window_duration=self.temp_agg_params["window_duration"],
        )

    ###################
    # Common methods
    ###################

    def _load_batches(self, data_dir: str) -> Generator[pd.DataFrame, None, None]:
        """Load the data files saved as dataframes.

        Parameters
        ----------
        data_dir : str
            The directory path of files.

        Yields
        ------
        pandas.DataFrame
            A DataFrame.

        """
        return yield_dataframes(data_dir, skip_n=self.skip_n, log=False)

    def _normalize(self, vectorized: Vectorized) -> Vectorized:
        """Fit normalizer and normalize.

        Parameters
        ----------
        vectorized : Vectorized
            Vectorized data.

        Returns
        -------
        Vectorized
            Vectorized data after normalization.

        """
        vectorized.fit_normalizer()
        vectorized.normalize()
        return vectorized

    ###################
    # Tabular methods
    ###################

    def _load_cohort(self) -> pd.DataFrame:
        """Load the tabular cohort.

        Returns
        -------
        pd.DataFrame
            Tabular data.

        """
        return load_dataframe(self.tabular_file)

    def _get_tabular_features(self, tab_data: pd.DataFrame) -> TabularFeatures:
        """Get the tabular features as an object.

        Parameters
        ----------
        tab_data : pd.DataFrame
            Tabular data.

        Returns
        -------
        TabularFeatures
            The tabular features object.

        """
        tab_features = TabularFeatures(
            data=tab_data,
            features=self.tab_feature_params["features"],
            by=self.tab_feature_params["primary_feature"],
            force_types=self.tab_feature_params["features_types"],
        )
        save_pickle(tab_features, self.tab_features_file)
        return tab_features

    def _slice_tabular(self, tab_features: TabularFeatures) -> np.ndarray:
        """Slice the tabular data.

        Parameters
        ----------
        tab_features : TabularFeatures
            The tabular features object.

        Returns
        -------
        np.ndarray
            Array of the values of the "by" column, in the sliced dataset.

        """
        sliced_tab = tab_features.slice(
            slice_map=self.tab_slice_params["slice_map"],
            slice_query=self.tab_slice_params["slice_query"],
            replace=self.tab_slice_params["replace"],
        )
        if self.tab_slice_params["replace"]:
            save_pickle(tab_features, self.tab_slice_file)
        return sliced_tab

    def _get_tab_ordinal(self, tab_features: TabularFeatures) -> List[str]:
        """Get the names of ordinal features in the tabular data.

        Parameters
        ----------
        tab_features : TabularFeatures
            The tabular features object.

        Returns
        -------
        List[str]
            List of ordinal features.

        """
        return tab_features.features_by_type(ORDINAL)

    def _get_tab_numeric(self, tab_features: TabularFeatures) -> List[str]:
        """Get the names of numeric features in the tabular data.

        Parameters
        ----------
        tab_features : TabularFeatures
            The tabular features object.

        Returns
        -------
        List[str]
            List of numeric features.

        """
        return tab_features.features_by_type(NUMERIC)

    def _vectorize_tabular(
        self, tab_features: TabularFeatures, normalize: bool
    ) -> Vectorized:
        """Vectorize the tabular data.

        Parameters
        ----------
        tab_features : TabularFeatures
            The tabular features object.
        normalize : bool
            Whether to normalize numeric features.

        Returns
        -------
        Vectorized
            Vectorized tabular data.

        """
        tab_vectorized = tab_features.vectorize(
            to_binary_indicators=self._get_tab_ordinal(tab_features)
        )

        if normalize:
            normalizer_map = {
                feat: self.tab_norm_params["method"]
                for feat in self._get_tab_numeric(tab_features)
            }
            tab_vectorized.add_normalizer(
                FEATURES,
                normalizer_map=normalizer_map,
            )

        save_pickle(tab_vectorized, self.tab_vectorized_file)
        return tab_vectorized

    def _aggregate_tabular(
        self, tab_features: TabularFeatures, temp_vectorized: Vectorized
    ) -> pd.DataFrame:
        """Aggregate the tabular data to pose as timeseries.

        Parameters
        ----------
        tab_features : TabularFeatures
            The tabular features object.
        temp_vectorized : Vectorized
            Vectorized temporal data.

        Returns
        -------
        pd.DataFrame
            Aggregated tabular data.

        """
        tab = tab_features.get_data(
            to_binary_indicators=self._get_tab_ordinal(tab_features)
        ).reset_index()

        tab = tab[
            np.in1d(
                tab[self.common_feature].values,
                temp_vectorized.get_index(self.common_feature),
            )
        ]

        tab_aggregated = tabular_as_aggregated(
            tab=tab,
            index=self.tab_agg_params["index"],
            var_name=self.tab_agg_params["var_name"],
            value_name=self.tab_agg_params["value_name"],
            strategy=self.tab_agg_params["strategy"],
            num_timesteps=self.temp_agg_params["window_duration"]
            // self.temp_agg_params["timestep_size"],
        )
        save_dataframe(tab_aggregated, self.tab_aggregated_file)
        return tab_aggregated

    def _vectorize_agg_tabular(self, tab_aggregated: pd.DataFrame) -> Vectorized:
        """Vectorize the aggregated tabular data.

        Parameters
        ----------
        tab_aggregated : pd.DataFrame
            Aggregated tabular data.

        Returns
        -------
        Vectorized
            Vectorized tabular data.

        """
        return self.aggregator.vectorize(tab_aggregated)

    def _split_tabular(self, tab_vectorized: Vectorized) -> Tuple:
        """Split tabular data to train, validation, and test sets.

        Parameters
        ----------
        tab_vectorized : Vectorized
            Vectorized tabular data.

        Returns
        -------
        Tuple
            A tuple of datasets of splits. All splits are Vectorized objects.

        """
        fractions = self.split_fractions.copy()
        tab_train, tab_val, tab_test = split_vectorized(
            vecs=[tab_vectorized],
            fractions=fractions,
            axes=self.common_feature,
        )[0]
        return tab_train, tab_val, tab_test

    def _get_tab_train(self, tab_train: Vectorized, normalize: bool) -> Tuple:
        """Get the tabular train features (normalized) and the targets.

        Parameters
        ----------
        tab_train : Vectorized
            Vectorized tabular data.
        normalize : bool
            Whether to normalize the numeric features.

        Returns
        -------
        Tuple
            Tuple of train features and targets.

        """
        tab_train_X, tab_train_y = tab_train.split_out(
            FEATURES, self.tab_feature_params[TARGETS]
        )
        if normalize:
            tab_train_X = self._normalize(tab_train_X)

        return tab_train_X, tab_train_y

    def _get_tab_val(self, tab_val: Vectorized, normalize: bool) -> Tuple:
        """Get the tabular validation features (normalized) and the targets.

        Parameters
        ----------
        tab_val : Vectorized
            Vectorized tabular data.
        normalize : bool
            Whether to normalize the numeric features.

        Returns
        -------
        Tuple
            Tuple of validation features and targets.

        """
        tab_val_X, tab_val_y = tab_val.split_out(
            FEATURES, self.tab_feature_params[TARGETS]
        )
        if normalize:
            tab_val_X = self._normalize(tab_val_X)
        return tab_val_X, tab_val_y

    def _get_tab_test(self, tab_test: Vectorized, normalize: bool):
        """Get the tabular test features (normalized) and the targets.

        Parameters
        ----------
        tab_test : Vectorized
            Vectorized tabular data.
        normalize : bool
            Whether to normalize the numeric features.

        Returns
        -------
        Tuple
           Tuple of test features and targets.

        """
        tab_test_X, tab_test_y = tab_test.split_out(
            FEATURES, self.tab_feature_params[TARGETS]
        )
        if normalize:
            tab_test_X = self._normalize(tab_test_X)
        return tab_test_X, tab_test_y

    def _save_tabular(
        self,
        tab_train_X: Vectorized,
        tab_train_y: Vectorized,
        tab_val_X: Vectorized,
        tab_val_y: Vectorized,
        tab_test_X: Vectorized,
        tab_test_y: Vectorized,
        aligned: bool,
    ) -> None:
        """Save the tabular features and targets for all data splits.

        Parameters
        ----------
        tab_train_X : Vectorized
            Vectorized tabular features from the train set.
        tab_train_y : Vectorized
            Vectorized tabular targets from the train set.
        tab_val_X : Vectorized
            Vectorized tabular features from the validation set.
        tab_val_y : Vectorized
             Vectorized tabular targets from the validation set.
        tab_test_X : Vectorized
            Vectorized tabular features from the test set.
        tab_test_y : Vectorized
            Vectorized tabular targets from the test set.
        aligned : bool
            Whether data is aligned with the temporal data.

        """
        vectorized = [
            (tab_train_X, "tab_train_X"),
            (tab_train_y, "tab_train_y"),
            (tab_val_X, "tab_val_X"),
            (tab_val_y, "tab_val_y"),
            (tab_test_X, "tab_test_X"),
            (tab_test_y, "tab_test_y"),
        ]
        for vec, name in vectorized:
            if aligned:
                save_pickle(vec, self.aligned_path + name)
            else:
                save_pickle(vec, self.unaligned_path + name)

    ####################
    # Temporal methods
    ####################

    def _get_temporal_features(self, data: pd.DataFrame) -> TemporalFeatures:
        """Get the temporal features as an object.

        Parameters
        ----------
        data : pd.DataFrame
            Temporal data.

        Returns
        -------
        TemporalFeatures
            The temporal features object.

        """
        return TemporalFeatures(
            data,
            features=self.temp_feature_params["features"],
            by=self.temp_feature_params["groupby"],
            timestamp_col=self.temp_feature_params["timestamp_col"],
            aggregator=self.aggregator,
        )

    def _get_timestamps(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Get relevant timestamps either from tabular data or the input dataframe.

        Parameters
        ----------
        data : Optional[pd.DataFrame], optional
            The dataframe to extract the timestamp from, by default None.

        Returns
        -------
        pd.DataFrame
            Timestamps data.

        """
        if not data:
            data = load_dataframe(self.tabular_file)
        return data[self.timestamp_params["columns"]]

    def _get_start_timestamps(self) -> pd.DataFrame:
        """Get relevant start timestamps e.g. hospital admission time.

        Returns
        -------
        pd.DataFrame
            The start timestamps.

        """
        timestamps = self._get_timestamps()
        start_timestamps = (
            timestamps[self.timestamp_params["start_columns"]]
            .set_index(self.timestamp_params["start_index"])
            .rename(self.timestamp_params["rename"], axis=1)
        )
        return start_timestamps

    def _aggregate_temporal_batches(
        self,
        generator: Generator[pd.DataFrame, None, None],
        filter_fn: Optional[Callable] = None,
    ) -> None:
        """Aggregate the temporal data saved in batches.

        Parameters
        ----------
        generator : Generator[pd.DataFrame, None, None]
            Generator to yield the saved data files.
        filter_fn : Optional[Callable], optional
            Filter the data records before aggregating, by default None.

        """
        start_timestamps = self._get_start_timestamps()
        for save_count, batch in enumerate(generator):
            if filter_fn:
                batch = filter_fn(batch)
            batch = batch.reset_index(drop=True)
            temp_features = self._get_temporal_features(batch)
            aggregated = temp_features.aggregate(window_start_time=start_timestamps)
            save_dataframe(
                aggregated,
                join(self.aggregated_dir, "batch_" + f"{save_count + self.skip_n:04d}"),
            )
            del batch

    def _vectorize_temporal_batches(self, generator: Generator) -> None:
        """Vectorize the temporal features saved in batches.

        Parameters
        ----------
        generator : Generator
            Generator to yield the saved data files.

        """
        for save_count, batch in enumerate(generator):
            vec = self.aggregator.vectorize(batch)
            save_pickle(
                vec,
                join(self.vectorized_dir, "batch_" + f"{save_count + self.skip_n:04d}"),
            )

    def _vectorize_temporal_features(
        self, generator: Generator[pd.DataFrame, None, None]
    ) -> Vectorized:
        """Vectorize temporal features (no targets included).

        Parameters
        ----------
        generator: Generator[pd.DataFrame, None, None]
            Generator to yield the saved data files.

        Returns
        -------
        Vectorized
            Vectorized temporal data.

        """
        vecs = list(generator)
        join_axis = vecs[0].get_axis(self.common_feature)
        res = np.concatenate([vec.data for vec in vecs], axis=join_axis)
        indexes = vecs[0].indexes
        indexes[join_axis] = np.concatenate([vec.indexes[join_axis] for vec in vecs])
        temp_vectorized = Vectorized(res, indexes, vecs[0].axis_names)
        del res
        return temp_vectorized

    def _compute_timestep(
        self, timestamps: pd.DataFrame, timestamp_col: str
    ) -> pd.DataFrame:
        """Compute timestep for a specific timestamp feature.

        Parameters
        ----------
        timestamps : pd.DataFrame
            The timestamps data.
        timestamp_col : str
            The timestamp for which the timestep is to be computed.

        Returns
        -------
        pd.DataFrame
            Timestamps with the new timestep.

        """
        timestep_size = self.temp_params["timestep_size"]
        new_timestamp = f"{timestamp_col}_{self.timestep_params['new_timestamp']}"
        timestamps[new_timestamp] = (
            timestamps[timestamp_col] - timestamps[self.timestep_params["anchor"]]
        )

        timestep_col = f"{timestamp_col}_timestep"
        timestamps[timestep_col] = (
            timestamps[new_timestamp] / pd.Timedelta(f"{timestep_size} hour")
        ).apply(np.floor)
        return timestamps

    def _create_target(
        self,
        temp_vectorized: Vectorized,
        timestamps: pd.DataFrame,
    ) -> np.ndarray:
        """Create targets for temporal data based on the window duration.

        Parameters
        ----------
        temp_vectorized : Vectorized
            Vectorized temporal data.
        timestamps : pd.DataFrame
            The timestamps data.

        Returns
        -------
        np.ndarray
            Array of the target values.

        """
        index_order = pd.Series(temp_vectorized.get_index(self.common_feature))
        index_order = index_order.rename(self.common_feature).to_frame()
        target_timestamp = self.temp_target_params["target_timestamp"]
        target_timestep = "target_timestep"
        ref_timestamp = self.temp_target_params["ref_timestamp"]
        ref_timestep = f"{ref_timestamp}_timestep"

        timestamps["target"] = timestamps[target_timestamp] - pd.DateOffset(
            hours=self.temp_params["predict_offset"]
        )

        timestamps = self._compute_timestep(timestamps, "target")
        timestamps = self._compute_timestep(
            timestamps,
            ref_timestamp,
        )

        timesteps = timestamps[
            [
                self.common_feature,
                target_timestep,
                ref_timestep,
            ]
        ]

        aligned_timestamps = pd.merge(
            index_order, timesteps, on=self.common_feature, how="left"
        )

        num_timesteps = int(
            self.temp_params["window_duration"] / self.temp_params["timestep_size"]
        )

        arr1 = timestamp_ffill_agg(
            aligned_timestamps[target_timestep], num_timesteps, fill_nan=2
        )

        arr2 = timestamp_ffill_agg(
            aligned_timestamps[ref_timestep], num_timesteps, val=-1, fill_nan=2
        )

        targets = np.minimum(arr1, arr2)
        targets[targets == 2] = 0
        targets = np.expand_dims(np.expand_dims(targets, 0), 2)
        return targets

    def _vectorize_temporal(
        self,
        temp_vectorized: Vectorized,
        targets: np.ndarray,
        normalize: bool,
    ) -> Vectorized:
        """Vectorize the tabular data.

        Parameters
        ----------
        temp_vectorized : Vectorized
            Vectorized temporal features.
        targets: np.ndarray
            Array of temporal targets.
        normalize : bool
            Whether to normalize the data.

        Returns
        -------
        Vectorized
            Vectorized temporal data containing features and targets.

        """
        temp_vectorized = temp_vectorized.concat_over_axis(
            self.temp_feature_params["primary_feature"],
            targets,
            self.temp_feature_params[TARGETS],
        )

        if normalize:
            temp_vectorized.add_normalizer(
                self.temp_feature_params["primary_feature"],
                normalization_method=self.temp_norm_params["method"],
            )

        save_pickle(temp_vectorized, self.temp_vectorized_file)
        return temp_vectorized

    def _split_temporal(self, temp_vectorized: Vectorized) -> Tuple:
        """Split the temporal data to train, validation, and test sets.

        Parameters
        ----------
        temp_vectorized : Vectorized
            Vectorized temporal data.

        Returns
        -------
        Tuple
            A tuple of datasets of splits. All splits are Vectorized objects.

        """
        fractions = self.split_fractions.copy()
        temp_train, temp_val, temp_test = split_vectorized(
            vecs=[temp_vectorized],
            fractions=fractions,
            axes=self.common_feature,
        )[0]
        return temp_train, temp_val, temp_test

    def _get_temp_train(
        self,
        temp_train: Vectorized,
        normalize: bool,
        impute: Optional[bool] = True,
    ) -> Tuple:
        """Get the temporal train features (normalized) and the targets.

        Parameters
        ----------
        temp_train : Vectorized
            Vectorized temporal data.
        normalize : bool
            Whether to normalize the data.
        impute : bool
            Whether to impute values.

        Returns
        -------
        Tuple
            Tuple of train features and targets.

        """
        temp_train_X, temp_train_y = temp_train.split_out(
            self.temp_feature_params["primary_feature"],
            self.temp_feature_params[TARGETS],
        )
        if impute:
            temp_train_X.impute(
                self.temp_impute_params["axis"],
                self.temp_feature_params["primary_feature"],
                self.temp_impute_params["func"],
            )

        if normalize:
            temp_train_X = self._normalize(temp_train_X)

        return temp_train_X, temp_train_y

    def _get_temp_val(
        self,
        temp_val,
        normalize: bool,
        impute: Optional[bool] = True,
    ) -> Tuple:
        """Get the temporal validation features (normalized) and the targets.

        Parameters
        ----------
        temp_val : Vectorized
            Vectorized temporal data.
        normalize : bool
            Whether to normalize the data.
        impute : bool
            Whether to impute values.

        Returns
        -------
        Tuple
            Tuple of validation features and targets.

        """
        temp_val_X, temp_val_y = temp_val.split_out(
            self.temp_feature_params["primary_feature"],
            self.temp_feature_params[TARGETS],
        )
        if impute:
            temp_val_X.impute(
                self.temp_impute_params["axis"],
                self.temp_feature_params["primary_feature"],
                self.temp_impute_params["func"],
            )
        if normalize:
            temp_val_X = self._normalize(temp_val_X)
        return temp_val_X, temp_val_y

    def _get_temp_test(
        self,
        temp_test,
        normalize: bool,
        impute: Optional[bool] = True,
    ) -> Tuple:
        """Get the temporal test features (normalized) and the targets.

        Parameters
        ----------
        temp_test : Vectorized
            Vectorized temporal data.
        normalize : bool
            Whether to normalize the data.
        impute : bool
            Whether to impute values.

        Returns
        -------
        Tuple
            Tuple of test features and targets.

        """
        temp_test_X, temp_test_y = temp_test.split_out(
            self.temp_feature_params["primary_feature"],
            self.temp_feature_params[TARGETS],
        )
        if impute:
            temp_test_X.impute(
                self.temp_impute_params["axis"],
                self.temp_feature_params["primary_feature"],
                self.temp_impute_params["func"],
            )
        if normalize:
            temp_test_X = self._normalize(temp_test_X)
        return temp_test_X, temp_test_y

    def _save_temporal(
        self,
        temp_train_X,
        temp_train_y,
        temp_val_X,
        temp_val_y,
        temp_test_X,
        temp_test_y,
        aligned,
    ):
        """Save the temporal features and targets for all data splits.

        Parameters
        ----------
        temp_train_X : Vectorized
            Vectorized temporal features from the train set.
        temp_train_y : Vectorized
            Vectorized temporal targets from the train set.
        temp_val_X : Vectorized
            Vectorized temporal features from the validation set.
        temp_val_y : Vectorized
            Vectorized temporal targets from the validation set.
        temp_test_X : Vectorized
            Vectorized temporal features from the test set.
        temp_test_y : Vectorized
            Vectorized temporal targets from the test set.
        aligned : bool
            Whether data is aligned with the tabular data.

        """
        vectorized = [
            (temp_train_X, "temp_train_X"),
            (temp_train_y, "temp_train_y"),
            (temp_val_X, "temp_val_X"),
            (temp_val_y, "temp_val_y"),
            (temp_test_X, "temp_test_X"),
            (temp_test_y, "temp_test_y"),
        ]
        for vec, name in vectorized:
            if aligned:
                save_pickle(vec, self.aligned_path + name)
            else:
                save_pickle(vec, self.unaligned_path + name)

    ###################
    # Combined methods
    ###################

    def _vectorize_combined(
        self,
        temp_vectorized: Vectorized,
        tab_aggregated_vec: Vectorized,
    ) -> Vectorized:
        """Vectorize the combined data.

        Parameters
        ----------
        temp_vectorized : Vectorized
            Vectorized temporal data.
        tab_aggregated_vec : Vectorized
            Vectorized aggregated tabular data.

        Returns
        -------
        Vectorized
            Vectorized combined data.

        """
        comb_vectorized = temp_vectorized.concat_over_axis(
            self.temp_feature_params["primary_feature"],
            tab_aggregated_vec.data,
            tab_aggregated_vec.get_index(self.temp_feature_params["primary_feature"]),
        )
        comb_vectorized, _ = comb_vectorized.split_out(
            self.temp_feature_params["primary_feature"],
            self.tab_feature_params[TARGETS],
        )

        comb_vectorized.add_normalizer(
            self.temp_feature_params["primary_feature"],
            normalization_method=self.temp_norm_params["method"],
        )

        save_pickle(comb_vectorized, self.comb_vectorized_file)
        return comb_vectorized

    def _get_intersect_vec(
        self,
        tab_vectorized: Vectorized,
        temp_vectorized: Vectorized,
        comb_vectorized: Vectorized,
    ) -> Tuple:
        """Get the records that are available in all datasets.

        Parameters
        ----------
        tab_vectorized : Vectorized
            Vectorized tabular data.
        temp_vectorized : Vectorized
            Vectorized temporal data.
        comb_vectorized : Vectorized
            Vectorized combined data.

        Returns
        -------
        Tuple
            Vectorized tabular, temporal, and combined data.

        """
        tab_vectorized, temp_vectorized, comb_vectorized = intersect_vectorized(
            [tab_vectorized, temp_vectorized, comb_vectorized], axes=self.common_feature
        )

        return tab_vectorized, temp_vectorized, comb_vectorized

    def _split_combined(self, comb_vectorized: Vectorized) -> Tuple:
        """Split combined data to train, validation, and test sets.

        Parameters
        ----------
        tab_vectorized : Vectorized
            Vectorized combined data.

        Returns
        -------
        Tuple
            A tuple of datasets of splits. All splits are Vectorized objects.

        """
        fractions = self.split_fractions.copy()
        comb_train, comb_val, comb_test = split_vectorized(
            vecs=[comb_vectorized],
            fractions=fractions,
            axes=self.common_feature,
        )[0]
        return comb_train, comb_val, comb_test

    def _get_comb_train(
        self,
        comb_train: Vectorized,
        normalize: bool,
        impute: Optional[bool] = True,
    ) -> Tuple:
        """Get combined train features (normalized) and the targets.

        Parameters
        ----------
        comb_train : Vectorized
            Vectorized combined data.
        normalize : bool
            Whether to normalize the data.
        impute : bool
            Whether to impute values.

        Returns
        -------
        Tuple
            Tuple of train features and targets.

        """
        comb_train_X, comb_train_y = comb_train.split_out(
            self.temp_feature_params["primary_feature"],
            self.temp_feature_params[TARGETS],
        )
        if impute:
            comb_train_X.impute(
                self.temp_impute_params["axis"],
                self.temp_feature_params["primary_feature"],
                self.temp_impute_params["func"],
            )

        if normalize:
            comb_train_X = self._normalize(comb_train_X)

        return comb_train_X, comb_train_y

    def _get_comb_val(
        self,
        comb_val: Vectorized,
        normalize: bool,
        impute: Optional[bool] = True,
    ) -> Tuple:
        """Get combined validation features (normalized) and the targets.

        Parameters
        ----------
        comb_validation : Vectorized
            Vectorized combined data.
        normalize : bool
            Whether to normalize the data.
        impute : bool
            Whether to impute values.

        Returns
        -------
        Tuple
            Tuple of validation features and targets.

        """
        comb_val_X, comb_val_y = comb_val.split_out(
            self.temp_feature_params["primary_feature"],
            self.temp_feature_params[TARGETS],
        )
        if impute:
            comb_val_X.impute(
                self.temp_impute_params["axis"],
                self.temp_feature_params["primary_feature"],
                self.temp_impute_params["func"],
            )

        if normalize:
            comb_val_X = self._normalize(comb_val_X)

        return comb_val_X, comb_val_y

    def _get_comb_test(
        self,
        comb_test: Vectorized,
        normalize: bool,
        impute: Optional[bool] = True,
    ) -> Tuple:
        """Get combined test features (normalized) and target.

        Parameters
        ----------
        comb_test : Vectorized
            Vectorized combined data.
        normalize : bool
            Whether to normalize the data.
        impute : bool
            Whether to impute values.

        Returns
        -------
        Tuple
            Tuple of test features and targets.

        """
        comb_test_X, comb_test_y = comb_test.split_out(
            self.temp_feature_params["primary_feature"],
            self.temp_feature_params[TARGETS],
        )
        if impute:
            comb_test_X.impute(
                self.temp_impute_params["axis"],
                self.temp_feature_params["primary_feature"],
                self.temp_impute_params["func"],
            )

        if normalize:
            comb_test_X = self._normalize(comb_test_X)

        return comb_test_X, comb_test_y

    def _save_combined(
        self,
        comb_train_X,
        comb_train_y,
        comb_val_X,
        comb_val_y,
        comb_test_X,
        comb_test_y,
    ):
        """Save combined features and targets for all data splits.

        Parameters
        ----------
        comb_train_X : Vectorized
            Vectorized combined features from the train set.
        comb_train_y : Vectorized
            Vectorized combined targets from the train set.
        comb_val_X : Vectorized
            Vectorized combined features from the validation set.
        comb_val_y : Vectorized
            Vectorized combined targets from the validation set.
        comb_test_X : Vectorized
            Vectorized combined features from the test set.
        comb_test_y : Vectorized
            Vectorized combined targets from the test set.

        """
        vectorized = [
            (comb_train_X, "comb_train_X"),
            (comb_train_y, "comb_train_y"),
            (comb_val_X, "comb_val_X"),
            (comb_val_y, "comb_val_y"),
            (comb_test_X, "comb_test_X"),
            (comb_test_y, "comb_test_y"),
        ]
        for vec, name in vectorized:
            save_pickle(vec, self.aligned_path + name)

    ###################
    # Process methods
    ###################

    def process_tabular_one(self) -> Tuple:
        """First step of tabular processing.

            1. Load data.
            2. Get tabular features as an object.
            3. Slice the data if required.
            4. Vectorize.

        Returns
        -------
        Tuple
            Tuple of vectorized tabular data and tabular features object.

        """
        LOGGER.info("Loading the tabular data.")
        cohort = self._load_cohort().reset_index(drop=True)
        tab_features = self._get_tabular_features(cohort)

        if self.tab_slice_params["slice"]:
            LOGGER.info("Slicing the tabular data.")
            _ = self._slice_tabular(tab_features)

        LOGGER.info("Vectorizing the tabular data.")
        tab_vectorized = self._vectorize_tabular(
            tab_features, self.tab_norm_params["normalize"]
        )
        return tab_vectorized, tab_features

    def process_tabular_two(self, tab_vectorized: Vectorized, aligned: bool) -> None:
        """Second step of tabular processing.

            1. Split.
            2. Get the features and targets for each split.
            3. Save the finalized vectorized data.

        Parameters
        ----------
        tab_vectorized : Vectorized
            Vectorized tabular data.
        aligned : bool
            Whether data is aligned with the temporal data.

        """
        LOGGER.info("Splitting the tabular data.")
        tab_train, tab_val, tab_test = self._split_tabular(tab_vectorized)

        tab_train_X, tab_train_y = self._get_tab_train(
            tab_train, self.tab_norm_params["normalize"]
        )
        tab_val_X, tab_val_y = self._get_tab_val(
            tab_val, self.tab_norm_params["normalize"]
        )
        tab_test_X, tab_test_y = self._get_tab_test(
            tab_test, self.tab_norm_params["normalize"]
        )

        LOGGER.info("Saving the tabular features and targets for all data splits.")
        self._save_tabular(
            tab_train_X,
            tab_train_y,
            tab_val_X,
            tab_val_y,
            tab_test_X,
            tab_test_y,
            aligned,
        )

    def process_tabular(self) -> None:
        """Process tabular data."""
        tab_vectorized, _ = self.process_tabular_one()
        self.process_tabular_two(tab_vectorized, aligned=False)

    def process_temporal_one(self) -> Vectorized:
        """First step of temporal processing.

            1. Aggregate temporal data.
            2. Vectorize temporal features.
            3. Create targets.
            4. Vectorize the whole temporal data.

        Returns
        -------
        Vectorized
            Vectorized temporal data.

        """
        cleaned_generator = self._load_batches(self.cleaned_dir)
        filter_fn = None
        if (
            self.temp_params["query"] == mimic.CHARTEVENTS  # pylint: disable=no-member
            and self.temp_params["top_n_events"]
        ):
            LOGGER.info("Getting top %d events", self.temp_params["top_n_events"])
            top_events = get_top_events(
                self.cleaned_dir, self.temp_params["top_n_events"]
            )
            filter_fn = lambda events: valid_events(events, top_events)  # noqa: E731

        LOGGER.info("Aggregating the temporal features in batches.")
        self._aggregate_temporal_batches(cleaned_generator, filter_fn)

        LOGGER.info("Vectorizing the temporal features in batches.")
        agg_generator = self._load_batches(self.aggregated_dir)
        self._vectorize_temporal_batches(agg_generator)

        vec_generator = yield_pickled_files(self.vectorized_dir)
        temp_vectorized = self._vectorize_temporal_features(vec_generator)

        LOGGER.info("Creating the temporal targets.")
        timestamps = self._get_timestamps()
        targets = self._create_target(temp_vectorized, timestamps)

        LOGGER.info("Vectorizing the temporal data.")
        temp_vectorized = self._vectorize_temporal(
            temp_vectorized, targets, self.temp_norm_params["normalize"]
        )
        return temp_vectorized

    def process_temporal_two(self, temp_vectorized: Vectorized, aligned: bool) -> None:
        """Second step of temporal processing.

            1. Split.
            2. Get the features and targets for each split.
            3. Save the finalized vectorized data.

        Parameters
        ----------
        temp_vectorized : Vectorized
            Vectorized temporal data.
        aligned : bool
            Whether the data is aligned with the tabular data.

        """
        LOGGER.info("Splitting the temporal data.")
        temp_train, temp_val, temp_test = self._split_temporal(temp_vectorized)
        temp_train_X, temp_train_y = self._get_temp_train(
            temp_train, self.temp_norm_params["normalize"]
        )
        temp_val_X, temp_val_y = self._get_temp_val(
            temp_val, self.temp_norm_params["normalize"]
        )
        temp_test_X, temp_test_y = self._get_temp_test(
            temp_test, self.temp_norm_params["normalize"]
        )

        LOGGER.info("Saving the temporal features and targets for data splits.")
        self._save_temporal(
            temp_train_X,
            temp_train_y,
            temp_val_X,
            temp_val_y,
            temp_test_X,
            temp_test_y,
            aligned,
        )

    def process_temporal(self) -> None:
        """Process temporal data."""
        temp_vectorized = self.process_temporal_one()
        self.process_temporal_two(temp_vectorized, aligned=False)

    def process_combined_one(self) -> Tuple:
        """First step of combined processing.

            1. Process tabular data or load from a file.
            2. Process temporal data or load from a file.
            3. Aggregate tabular data.
            4. Vectorize aggregated tabular data.
            5. Vectorize the combined data.
            6. Get the intersection of the three datasets.

        Returns
        -------
        Tuple
            Vectorized tabular, temporal, and combined data.

        """
        LOGGER.info("Getting the vectorized tabular data.")
        if path.exists(self.tab_vectorized_file):
            tab_vectorized = load_pickle(self.tab_vectorized_file)
            tab_features = load_pickle(self.tab_features_file)
        else:
            tab_vectorized, tab_features = self.process_tabular_one()

        LOGGER.info("Getting the vectorized temporal data.")
        if path.exists(self.temp_vectorized_file):
            temp_vectorized = load_pickle(self.temp_vectorized_file)
        else:
            temp_vectorized = self.process_temporal_one()

        LOGGER.info("Combining tabular and temporal data.")
        tab_aggregated = self._aggregate_tabular(tab_features, temp_vectorized)
        tab_aggregated_vec = self._vectorize_agg_tabular(tab_aggregated)
        comb_vectorized = self._vectorize_combined(temp_vectorized, tab_aggregated_vec)
        tab_vectorized, temp_vectorized, comb_vectorized = self._get_intersect_vec(
            tab_vectorized, temp_vectorized, comb_vectorized
        )
        return tab_vectorized, temp_vectorized, comb_vectorized

    def process_combined_two(self, comb_vectorized: Vectorized) -> None:
        """Second step of combined processing.

            1. Split.
            2. Get the features and targets for each split.
            3. Save the finalized vectorized data.

        Parameters
        ----------
        comb_vectorized : Vectorized
            Vectorized combined data.

        """
        LOGGER.info("Splitting the combined data.")
        comb_train, comb_val, comb_test = self._split_combined(comb_vectorized)
        comb_train_X, comb_train_y = self._get_comb_train(
            comb_train, self.temp_norm_params["normalize"]
        )
        comb_val_X, comb_val_y = self._get_comb_val(
            comb_val, self.temp_norm_params["normalize"]
        )
        comb_test_X, comb_test_y = self._get_comb_test(
            comb_test, self.temp_norm_params["normalize"]
        )

        LOGGER.info("Saving the combined features and targets for all data splits.")
        self._save_combined(
            comb_train_X,
            comb_train_y,
            comb_val_X,
            comb_val_y,
            comb_test_X,
            comb_test_y,
        )

    def process_combined(self) -> None:
        """Process combined data."""
        tab_vectorized, temp_vectorized, comb_vectorized = self.process_combined_one()
        self.process_tabular_two(tab_vectorized, aligned=True)
        self.process_temporal_two(temp_vectorized, aligned=True)
        self.process_combined_two(comb_vectorized)
