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
from itertools import chain

# from datasets.base import WindowsDataset, BaseConcatDataset, EEGWindowsDataset

from datasets.base import BaseConcatDataset, BaseDataset



def create_fixed_length_window_Nina( 
        concat_ds, start_offset_seconds=0, stop_offset_seconds=None,
        window_size_seconds=None, window_stride_seconds=None, 
        mapping=None, preload=False, picks=None,
        reject=None, flat=None, targets_from='metadata', last_target_only=True,
        on_missing='error', n_jobs=1, verbose='error', remove_background_annotations=False):
    
     

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
            on_missing, verbose, remove_background_annotations) for ds in concat_ds.datasets)
    return list(chain.from_iterable(BaseConcatDataset(list_of_base_ds).datasets))




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
        last_target_only=True, on_missing='error', verbose='error', remove_background_annotations=False):
   
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
    try:
        if (window_size_samples is not None) and (window_size_samples > (length-start_offset_samples-stop_offset_samples)):
            raise ValueError(f'Window size {window_size_samples} exceeds the '
                             f'duration {window_size_seconds} (sec) of the segment including the offsets, samp_freq: {sfreq}.')    
    except: 
        return []
    
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
   
    for i in range(len(starts)): 
        ds_copy = ds.raw.copy()
        additional_description = {'start time in the original edf file': time_start_in_trial[i], 
                                 'stop time in the original edf file': time_stop_in_trial[i]}
        raw = ds_copy.crop(tmin=time_start_in_trial[i], tmax=time_stop_in_trial[i], include_tmax=False)

        # remove annotations pertaining to bckg activity
        if remove_background_annotations:
            raw = _screen_annotations_based_on_description(raw, 'bckg')
        
        list_of_ds.append(BaseDataset(raw = raw,
                                      description = pd.concat([ds.description, pd.Series(additional_description)]), 
                                      transform = ds.transform, 
                                      target_name = ds.target_name))
                         
    return list_of_ds




def _screen_annotations_based_on_description(edf_raw, keyword_to_remove):
    #remove any non-term based annotations from the ds. 
    temp = []
    if (len(edf_raw.annotations)!=0):
        for i in range(len(edf_raw.annotations)):   
            if keyword_to_remove in edf_raw.annotations[i]['description']:
                temp.append(i)
        edf_raw.annotations.delete(temp)
    return edf_raw



def create_window_from_TERM_annotations( 
        concat_ds, start_offset_seconds=0, stop_offset_seconds=None,
        window_size_seconds=None, window_stride_seconds=None, 
        mapping=None, preload=False, picks=None,
        reject=None, flat=None, targets_from='metadata', last_target_only=True,
        on_missing='error', n_jobs=1, verbose='error', remove_background_annotations=False):
    
    list_of_base_ds = Parallel(n_jobs=n_jobs)(
        delayed(_create_window_from_TERM_annotations)(
            ds, start_offset_seconds, stop_offset_seconds, window_size_seconds,
            window_stride_seconds, mapping, preload,
            picks, reject, flat, targets_from, last_target_only,
            on_missing, verbose, remove_background_annotations) for ds in concat_ds.datasets)
    
    return list(chain.from_iterable(BaseConcatDataset(list_of_base_ds).datasets))



def _create_window_from_TERM_annotations(
        ds, start_offset_seconds, stop_offset_seconds, window_size_seconds,
        window_stride_seconds, mapping=None, preload=False,
        picks=None, reject=None, flat=None, targets_from='metadata',
        last_target_only=True, on_missing='error', verbose='error', remove_background_annotations=False):
   
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
    length_seconds = ds.raw.n_times / sfreq
    try:
        if (window_size_samples is not None) and (window_size_samples > (length-start_offset_samples-stop_offset_samples)):
            raise ValueError(f'Window size {window_size_samples} exceeds the '
                             f'duration {window_size_seconds} (sec) of the segment including the offsets, samp_freq: {sfreq}.')  
    except: 
        return []

    # check that the sample size and offsets make sense
    _check_windowing_arguments(
        start_offset_samples, stop_offset_samples,
        window_size_samples, window_stride_samples)

    # remove annotations pertaining to EVENT based activity
    ds.raw = _screen_annotations_based_on_description(ds.raw, 'EVENT')
    
    # If it is a segment containing no seizure activity at all -> i.e both the description pathological session (TERM label)            
    # and pathological session (EVENT labels) == False -> the. just segment the data similarly to create_fixed_length_window
    list_of_ds = []


    if (not ds.description['pathological session (TERM label)']) and (not ds.description['pathological session (EVENT labels)']):
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
        time_start_in_trial = i_start_in_trial / sfreq
        time_stop_in_trial = i_stop_in_trial / sfreq
        
        for i in range(len(starts)): 
            ds_copy = ds.raw.copy()
            additional_description = {'start time in the original edf file': time_start_in_trial[i], 
                                     'stop time in the original edf file': time_stop_in_trial[i]}
            raw = ds_copy.crop(tmin=time_start_in_trial[i], tmax=time_stop_in_trial[i], include_tmax=False)

            # remove annotations pertaining to bckg activity
            if remove_background_annotations:
                raw = _screen_annotations_based_on_description(raw, 'bckg')

            list_of_ds.append(BaseDataset(raw = raw,
                                          description = pd.concat([ds.description, pd.Series(additional_description)]), 
                                          transform = ds.transform, 
                                          target_name = ds.target_name))

    # If you have TERM seizure in the recording    
    elif ds.description['pathological session (TERM label)']:
        #crop the extremities to remove the start and stop offsets: 
        ds.raw.crop(tmin = start_offset_seconds, tmax = length_seconds-stop_offset_seconds, include_tmax=False)
        
        #for all of the rest of the TERM time and SEIZURE annotations, crop the samples.
        temp =[]
        for i in range(len(ds.raw.annotations)):   
            if 'seiz' in ds.raw.annotations[i]['description']:
                temp.append(ds.raw.annotations[i])
                
        list_of_raw_term_sz_segments = []                  
        try :
            list_of_raw_term_sz_segments = (ds.raw.crop_by_annotations(temp))
        except:
            pass
         
        seizure_segments_full_length = []
        for i in range(len(list_of_raw_term_sz_segments)):
            try:
                seizure_segments_full_length.append(BaseDataset(raw = list_of_raw_term_sz_segments[i], 
                                              description = ds.description,  
                                              transform = ds.transform,                                     
                                              target_name = ds.target_name))
            except:
                pass
                
        # individually divide them in small fixed length windows, which you append to list_of_ds
        # note, you use the same window length and overlap as for background segmentation
        for seizure_segment_full_length in seizure_segments_full_length:                    
            length_seconds=seizure_segment_full_length.raw.n_times/seizure_segment_full_length.raw.info['sfreq']
            last_potential_start = length_seconds - window_size_seconds
            starts = np.arange(0, last_potential_start, window_stride_seconds) 
            stops = starts + window_size_seconds
                    
            for start in range(len(starts)):
                ds = seizure_segment_full_length.raw.copy()
                list_of_ds.append(BaseDataset(raw = ds.crop(tmin=starts[start], tmax=stops[start], include_tmax=False), description = seizure_segment_full_length.description, transform = seizure_segment_full_length.transform,target_name = seizure_segment_full_length.target_name))
        
    return list_of_ds






