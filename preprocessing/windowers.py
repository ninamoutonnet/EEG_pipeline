"""Get epochs from mne.Raw
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#          Henrik Bonsmann <henrikbons@gmail.com>
#          Ann-Kathrin Kiessner <ann-kathrin.kiessner@gmx.de>
#          Vytautas Jankauskas <vytauto.jankausko@gmail.com>
#          Dan Wilson <dan.c.wil@gmail.com>
#          Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Mohammed Fattouh <mo.fattouh@gmail.com>
#          Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

## Nind heavily edited

import warnings

import numpy as np
import mne
import pandas as pd
from joblib import Parallel, delayed

# from datasets.base import WindowsDataset, BaseConcatDataset, EEGWindowsDataset

from datasets.base import BaseConcatDataset, BaseDataset



def create_fixed_length_window_Nina( 
        concat_ds, start_offset_seconds=0, stop_offset_seconds=None,
        window_size_seconds=None, window_stride_seconds=None, 
        mapping=None, preload=False, picks=None,
        reject=None, flat=None, targets_from='metadata', last_target_only=True,
        on_missing='error', n_jobs=1, verbose='error'):
    
    '''
    # check that the sample size and offsets make sense
    start_offset_seconds, drop_last_window = _check_and_set_fixed_length_window_arguments(
        start_offset_seconds, stop_offset_seconds, window_size_seconds, window_stride_seconds,
        drop_last_window)

    
    # the window size and stride and all are now in seconds, not in samples.
    # convert everything here; 
    # note that this is a list of the sampling frequencies
    sfreq = [ds.raw.info['sfreq'] for ds in tuh_sz.datasets] 
    start_offset_samples = start_offset_seconds * sfreq
    stop_offset_samples = stop_offset_seconds * sfreq
    window_size_samples = window_size_seconds * sfreq
    window_stride_samples = window_stride_seconds * sfreq
    

    # fetch the recordings different lengths
    lengths = np.array([ds.raw.n_times for ds in concat_ds.datasets])
    if (np.diff(lengths) != 0).any() and window_size_samples is None:
        # if the recordings have different lengths in samples, which they do
        warnings.warn('Recordings have different lengths, they will not be batch-able!')
    if (window_size_samples is not None) and any(window_size_samples > lengths):
        raise ValueError(f'Window size {window_size_samples} exceeds trial '
                         f'duration {lengths.min()}.')
    '''  

    # fetch the recordings different lengths
    lengths = np.array([ds.raw.n_times for ds in concat_ds.datasets])
    if (np.diff(lengths) != 0).any() and window_size_seconds is None:
        # if the recordings have different lengths in samples, which they do
        warnings.warn('Recordings have different lengths, they will not be batch-able!')
        
    list_of_base_ds = Parallel(n_jobs=n_jobs)(
        delayed(_create_fixed_length_windows)(
            ds, start_offset_seconds, stop_offset_seconds, window_size_seconds,
            window_stride_seconds, mapping, preload,
            picks, reject, flat, targets_from, last_target_only,
            on_missing, verbose) for ds in concat_ds.datasets)
    return BaseConcatDataset(list_of_base_ds)




def _check_windowing_arguments(
        trial_start_offset_samples, trial_stop_offset_samples,
        window_size_samples, window_stride_samples):
    
    # check the validity of input arguments, ie, they are not floats or you don't have a stride without a window size
    assert isinstance(trial_start_offset_samples, (int, np.integer))
    assert isinstance(trial_stop_offset_samples, (int, np.integer, type(None)))
    assert isinstance(window_size_samples, (int, np.integer, type(None)))
    assert isinstance(window_stride_samples, (int, np.integer, type(None)))
    
    assert (window_size_samples is None) == (window_stride_samples is None)
    
    # if you specify a size, it has to be greater than 0 and you have to have a stride >0 too 
    if window_size_samples is not None:
        assert window_size_samples > 0, (
            "window size has to be larger than 0")
        assert window_stride_samples > 0, (
            "window stride has to be larger than 0")






def _create_fixed_length_windows(
        ds, start_offset_seconds, stop_offset_seconds, window_size_seconds,
        window_stride_seconds, mapping=None, preload=False,
        picks=None, reject=None, flat=None, targets_from='metadata',
        last_target_only=True, on_missing='error', verbose='error'):
   
    # the window size and stride and all are now in seconds, not in samples.
    # convert everything here; 
    # note that this is a list of the sampling frequencies
    sfreq = ds.raw.info['sfreq']  
    start_offset_samples = int(start_offset_seconds * sfreq)
    stop_offset_samples = int(stop_offset_seconds * sfreq)
    window_size_samples = int(window_size_seconds * sfreq)
    window_stride_samples = int(window_stride_seconds * sfreq)
    
    #check that the window size does not exceed the data duration
    length = np.array(ds.raw.n_times)
    if (window_size_samples is not None) and (window_size_samples > length):
        raise ValueError(f'Window size {window_size_samples} exceeds the '
                         f'duration {window_size_seconds} (sec) of the segment, samp_freq: {sfreq}.')        
    
    # check that the sample size and offsets make sense
    _check_windowing_arguments(
        start_offset_samples, stop_offset_samples,
        window_size_samples, window_stride_samples)
    
    # stop is the end sample 
    stop = ds.raw.n_times if stop_offset_samples is None else (ds.raw.n_times - stop_offset_samples)

    # assume window should be whole recording
    if window_size_samples is None:
        window_size_samples = stop - start_offset_samples
    # default stride is 0 overlap
    if window_stride_samples is None:
        window_stride_samples = window_size_samples

    last_potential_start = stop - window_size_samples

    # already includes last incomplete window start
    starts = np.arange(start_offset_samples, last_potential_start + 1, window_stride_samples)


    # get targets from dataset description if they exist
    target = -1 if ds.target_name is None else ds.description[ds.target_name]
    if mapping is not None:
        # in case of multiple targets
        if isinstance(target, pd.Series):
            target = target.replace(mapping).to_list()
        # in case of single value target
        else:
            target = mapping[target]

    
    i_window_in_trial = np.arange(len(starts))
    i_start_in_trial = starts
    i_stop_in_trial = starts + window_size_samples
    

    # Take the raw trace and crop it using the indices computes above. 
    # for every smaller segment, create a Base dataset, keep the description, transform and target_name the same, 
    list_of_ds = []
    time_start_in_trial = i_start_in_trial / sfreq
    time_stop_in_trial = i_stop_in_trial / sfreq
    
    print(sfreq)
    print(time_start_in_trial)
    print()
    print(time_stop_in_trial)

    for i in range(len(starts)): 
        ds_copy = ds.raw.copy()
        list_of_ds.append(BaseDataset(raw = ds_copy.crop(tmin=time_start_in_trial[i], tmax=time_stop_in_trial[i]),
                                      description = ds.description, 
                                      transform = ds.transform, 
                                      target_name = ds.target_name))
                         
    return list_of_ds





