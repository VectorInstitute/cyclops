"""Cyclops package."""

import pandas as pd


# use new copy-view behaviour using Copy-on-Write, which will be default in pandas 3.0
# see: https://pandas.pydata.org/docs/user_guide/copy_on_write.html#copy-on-write-enabling
pd.options.mode.copy_on_write = True


# whether to infer sequence of str objects as pyarrow string dtype
# this will be the default in pandas 3.0
pd.options.future.infer_string = True
