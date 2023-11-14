# Authors: Lukas Gemein <l.gemein@gmail.com>
#          Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import pandas as pd
import mne

from .base import BaseDataset, BaseConcatDataset, WindowsDataset


def create_from_mne_raw(
        raws, trial_start_offset_samples, trial_stop_offset_samples,
        window_size_samples, window_stride_samples, drop_last_window,
        descriptions=None, mapping=None, preload=False, drop_bad_windows=True,
        accepted_bads_ratio=0.0):
    """Create WindowsDatasets from mne.RawArrays

    Parameters
    ----------
    raws: array-like
        list of mne.RawArrays
    trial_start_offset_samples: int
        start offset from original trial onsets in samples
    trial_stop_offset_samples: int
        stop offset from original trial stop in samples
    window_size_samples: int
        window size
    window_stride_samples: int
        stride between windows
    drop_last_window: bool
        whether or not have a last overlapping window, when
        windows do not equally divide the continuous signal
    descriptions: array-like
        list of dicts or pandas.Series with additional information about the raws
    mapping: dict(str: int)
        mapping from event description to target value
    preload: bool
        if True, preload the data of the Epochs objects.
    drop_bad_windows: bool
        If True, call `.drop_bad()` on the resulting mne.Epochs object. This
        step allows identifying e.g., windows that fall outside of the
        continuous recording. It is suggested to run this step here as otherwise
        the BaseConcatDataset has to be updated as well.
    accepted_bads_ratio: float, optional
        Acceptable proportion of trials withinconsistent length in a raw. If
        the number of trials whose length is exceeded by the window size is
        smaller than this, then only the corresponding trials are dropped, but
        the computation continues. Otherwise, an error is raised. Defaults to
        0.0 (raise an error).

    Returns
    -------
    windows_datasets: BaseConcatDataset
        X and y transformed to a dataset format that is compatible with skorch
        and braindecode
    """
    # Prevent circular import
    from preprocessing.windowers import create_fixed_length_windows
    
    if descriptions is not None:
        if len(descriptions) != len(raws):
            raise ValueError(
                f"length of 'raws' ({len(raws)}) and 'description' "
                f"({len(descriptions)}) has to match")
        base_datasets = [BaseDataset(raw, desc) for raw, desc in
                         zip(raws, descriptions)]
    else:
        base_datasets = [BaseDataset(raw) for raw in raws]

    base_datasets = BaseConcatDataset(base_datasets)
    
    
    windows_datasets = create_fixed_length_windows(
        base_datasets,
        start_offset_samples=trial_start_offset_samples,
        stop_offset_samples=trial_stop_offset_samples,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        drop_last_window=drop_last_window,
        mapping=mapping,
        preload=preload,
    )
    # remove drop_bad_windows=drop_bad_windows, as I am using create_fixed_length_windows, not create_windows_from_events
    return windows_datasets
