"""
Dataset classes for the Siena scalp EEG Corpus.
"""

# Authors: Nina Moutonnet <nm2318@ic.ac.uk>
#


import re
import os
import glob
import warnings
from unittest import mock
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
import numpy as np
import mne
from joblib import Parallel, delayed

from .base import BaseDataset, BaseConcatDataset
from .subject_info_siena import subject_info_Siena

class Siena(BaseConcatDataset):
    """Siena scalp EEG Corpus
    (https://physionet.org/content/siena-scalp-eeg/1.0.0/).

    Parameters
    ----------
    path: str
        Parent directory of the dataset.
    recording_ids: list(int) | int
        A (list of) int of recording id(s) to be read (order matters and will
        overwrite default chronological order, e.g. if recording_ids=[1,0],
        then the first recording returned by this class will be chronologically
        later then the second recording. Provide recording_ids in ascending
        order to preserve chronological order.).
    target_name: str
        Can be 'gender', or 'age', or 'pathological session (TERM label)'.
    preload: bool
        If True, preload the data of the Raw objects.
    n_jobs: int
        Number of jobs to be used to read files in parallel. 
    debug: bool
        If True, only select the first 100 files in the files paths option
        Added by Nina, check that this can be adapted for other TUH repositories
    """
    def __init__(self, path, recording_ids=None, target_name=None,
                 preload=False, n_jobs=1, debug=False):
            
        # create an index of all files and gather easily accessible info
        # without actually touching the files
        file_paths = glob.glob(os.path.join(path, '**/*.edf'), recursive=True)
        # If debug is true and the number of edf files is bigger than 10, select 
        # the first 10 file_paths. This is a small number for Siena as the number 
        # of edf files is small, but they are quite long in time
        if (debug and len(file_paths)>10):
            file_paths = file_paths[0:10]
        
        descriptions = _create_description(file_paths)
                
        # limit to specified recording ids before doing slow stuff
        if recording_ids is not None:
            if not isinstance(recording_ids, Iterable):
                # Assume it is an integer specifying number
                # of recordings to load
                recording_ids = range(recording_ids)
            descriptions = descriptions[recording_ids]
            
        # this is the second loop (slow)
        # create datasets gathering more info about the files touching them
        # reading the raws and potentially preloading the data
        # disable joblib for tests. 
        if n_jobs == 1:
            base_datasets = [self._create_dataset(descriptions[i], target_name, preload) for i in descriptions.columns]
                
        else:
            base_datasets = Parallel(n_jobs)(delayed(
                self._create_dataset)(descriptions[i], target_name, preload) for i 
                                             in descriptions.columns)
        super().__init__(base_datasets)

    @staticmethod
    def _create_dataset(description, target_name, preload):
        file_path = description.loc['path']

        # parse age and gender information from the txt file called 'subject info'
        age, gender, sz_type, localisation, lateralisation, nb_eeg, nb_sz, total_rec_time_mins = _parse_subject_info(file_path)
        
        raw = mne.io.read_raw_edf(file_path, 
                                  preload=preload, 
                                  verbose='ERROR')
        
        # Get rid of useless channels in the raw file. 
        # "Each folder also includes a text file named Seizures-list-PNxx.txt containing: data sampling rate (in Hz); the list of the channels from which the EEG and EKG signals are extracted (all other channels in the edf files must be ignored); "
        bad_channels_chb_mit = _parse_bad_channel_info(file_path, raw.info['ch_names'])     
        raw.drop_channels(ch_names=bad_channels_chb_mit,
                         on_missing='ignore')
        
        # default
        recording_type_TERM = 'seizure' # the edf files of Siena all contain at least 1 seizure 
        recording_type_EVENT = None # the Siena dataset does not contain channel specific annotations, so default is None
        
        
        #################################################################
        # Create annotation 
        annotations = _parse_term_based_annotations_from_txt_file(file_path, raw.info['meas_date'])  
        raw = raw.set_annotations(annotations, on_missing='warn')
        
        # set the subject ID:
        raw.info['subject_info']['his_id'] = description['subject']
        
        d = {
                'age': age,
                'gender': gender,
                'pathological session (TERM label)': 'seizure' in recording_type_TERM,
                'pathological session (EVENT labels)': False      
        }
        
        
        additional_description = pd.Series(d)
        description = pd.concat([description, additional_description])
        base_dataset = BaseDataset(raw, description, target_name=target_name)
        
        return base_dataset


        
def _parse_term_based_annotations_from_txt_file(file_path, orig_time):
    # From a path such as : /physionet.org/files/siena-scalp-eeg/1.0.0/PN00/PN00-1.edf,
    # get:  /physionet.org/files/siena-scalp-eeg/1.0.0/PN00/Seizures-list-PN00.txt
    
    file_path = os.path.normpath(file_path)
    print(file_path)
    tokens = file_path.split(os.sep)
    # Remove last part of path
    subject_id_and_rec_nb = tokens[-1]  # PN00-1.edf
    subject_id = tokens[-2]  # PN00
    
    # Deal with path name inconsistencies
    if subject_id == 'PN01':
        summary_file_path = file_path.replace(subject_id_and_rec_nb, "Seizures-list-PN01.txt")
    elif subject_id == 'PN11':
        summary_file_path = file_path.replace(subject_id_and_rec_nb, "Seizures-list-PN11.txt")
    else:
        summary_file_path = file_path.replace(subject_id_and_rec_nb, "Seizures-list-"+subject_id+".txt")
    
    # store the summary file line by line in a list
    with open(summary_file_path) as file:
        summary_list = file.readlines()
    
    sz_onset = []
    sz_duration = []
    sz_description = []
    indices = []
    
    sz_start_time =[]
    sz_end_time = []
    
    print(subject_id_and_rec_nb)
    
    #manual extraction as structure is different for this subject - inconsistency 
    if subject_id_and_rec_nb == 'PN01-1.edf':
        sz_start_time.append('21.51.02')
        sz_end_time.append('21.51.56')
        
        sz_start_time.append('07.53.17') 
        sz_end_time.append('07.54.31')

    elif subject_id_and_rec_nb == 'PN12-1.2.edf':
        sz_start_time.append('16.13.23')
        sz_end_time.append('16.14.26')
        
        sz_start_time.append('18.31.01')
        sz_end_time.append('18.32.09')

    
    # Problem with PN06 -> some file are named with a O instead of 0.
    elif subject_id_and_rec_nb == 'PN06-1.edf':
        sz_start_time.append('05.54.25')
        sz_end_time.append('05.55.29')
        

    # Problem with PN06 -> some file are named with a O instead of 0.
    elif subject_id_and_rec_nb == 'PN06-2.edf':
        sz_start_time.append('23.39.09')
        sz_end_time.append('23.40.18')

    
    # Problem with PN06 -> some file are named with a O instead of 0.
    elif subject_id_and_rec_nb == 'PN06-4.edf':
        sz_start_time.append('12.55.08')
        sz_end_time.append('12.56.11')
        
        
    # Note: PN05 has seizure number 2,3,4 but not 1. 
      
    #get all the seizure start/end time in that edf file, the list is for edf files containing more than 1 seizure
    elif subject_id == 'PN11':  
        indices = [i for i, s in enumerate(summary_list) if "File name: PN11-.edf"in s]
        
    else: 
        indices = [i for i, s in enumerate(summary_list) if "File name: "+subject_id_and_rec_nb in s]
    
    for index in indices:
        sz_end_time.append((summary_list[index+4]).split(':')[1].split()[0])
        sz_start_time.append((summary_list[index+3]).split(':')[1].split()[0])
    
    print(sz_start_time)
    print(sz_end_time)
    print('recording start time ', orig_time.time())
    
    
    # Find the times and durations in seconds to populate mne.Annotation object
    date_format = "%H.%M.%S"
    date_format_2 = "%H:%M:%S"

    
    for seizure_index in range(len(sz_start_time)):
        seizure_duration_seconds = (datetime.strptime(sz_end_time[seizure_index], date_format) - datetime.strptime(sz_start_time[seizure_index], date_format)).seconds
        sz_duration.append(seizure_duration_seconds)

        
        orig_time_no_tz = datetime.strptime(str(orig_time.time()), date_format_2)
        sz_start_no_tz = datetime.strptime(str(datetime.strptime(sz_start_time[seizure_index], date_format).time()), date_format_2)      
        # onset array of float, shape (n_annotations,) The starting time of annotations in seconds after orig_time.
        print('rec start time: ', orig_time_no_tz)
        print('seiz start time: ', sz_start_no_tz)
        seizure_onset_seconds =  (sz_start_no_tz - orig_time_no_tz).seconds
        print('onset in seconds: ', seizure_onset_seconds) 
        sz_onset.append(seizure_onset_seconds)
        
        
        sz_description.append("Seizure "+str(seizure_index+1))
        
    
    print('onset ', sz_onset)
    print('duration ', sz_duration)
    print('description: ', sz_description)
    print()


    
    csvbi_annotations = mne.Annotations(onset = sz_onset, 
                                        duration = sz_duration, 
                                        description = sz_description)  
    return csvbi_annotations


def _create_description(file_paths):
    descriptions = [_parse_description_from_file_path(f) for f in file_paths]
    descriptions = pd.DataFrame(descriptions)
    return descriptions.T


def _parse_description_from_file_path(file_path):
    # stackoverflow.com/questions/3167154/how-to-split-a-dos-path-into-its-components-in-python  # noqa
    # file path looks life that: "../physionet.org/files/siena-scalp-eeg/1.0.0/PN00/PN00-2.edf"
    
    file_path = os.path.normpath(file_path)
    tokens = file_path.split(os.sep)
    
    # Extract version number, should always be 1.0.0 for Siena as there was only 1 release
    version = tokens[-3]  
    
    # Extract the subject id
    subject_id = tokens[-2]  # PN00
    
    # For the Siena, the  The edf files contain signals recorded on the same or different days and the seizure events are chronologically ordered. Assume different edf files are different sessions then? All dates in the .edf files are de-identified.
    session = tokens[-1].split('-')[1].split('.')[0]
    segment = 0
   
    # Note: dates in the edf files are dummy dates, so no need to store them for this dataset
    description = {
        'path': file_path,
        'version': version,
        'subject': subject_id,
        'session': session,
        'segment': segment
    }
    
    return description
    

def _parse_subject_info(file_path):
    
    file_path = os.path.normpath(file_path)
    tokens = file_path.split(os.sep)
    # Extract the subject id
    subject_id = tokens[-2]  # PN00
    # find the corresponding age, gender, seizure, localization, lateralization, eeg_channel, number_seizures, rec_time_minutes
    [age, gender, sz, localisation, lateralisation, nb_eeg, nb_sz, total_rec_time_mins] = subject_info_Siena[subject_id]
    
    return age, gender, sz, localisation, lateralisation, nb_eeg, nb_sz, total_rec_time_mins




def _parse_bad_channel_info(file_path, raw_channels):
    
    good_channels = []
    good_raw_channels = []
    
    file_path = os.path.normpath(file_path)
    tokens = file_path.split(os.sep)
    # Remove last part of path
    subject_id_and_rec_nb = tokens[-1]  # PN00-1.edf
    subject_id = tokens[-2]  # PN00


    summary_file_path = file_path.replace(subject_id_and_rec_nb, "Seizures-list-"+subject_id+".txt")
    # store the summary file line by line in a list
    with open(summary_file_path) as file:
        summary_list = file.readlines()
    
    # extract the good channels
    for line in summary_list:
        if 'Channel ' in line:
            term = line.split(':')[1].split('\n')[0].strip()# format is: Channel 26: F10   \n or Channel 29: EKG 1 \n
            # Add 'EEG' bfore single terms like Fp3. This is Ok as EKG channels are always labelled as EKG 1. etc and no other channels has a name in 1 part apart from EEG
            if len(term.split(' '))==1: 
                term = "EEG "+term # EEG Fp1
            good_channels.append(term) 
    
    # Channel 5 always show '1' but should be 'O1', change it here: 
    good_channels = ['EEG O1' if x=='EEG 1' else x for x in good_channels]
    
    for channel in good_channels:
        good_raw_channel = [raw_channel for raw_channel in raw_channels if (channel in raw_channel)]
        if good_raw_channel != []:
            good_raw_channels.append(good_raw_channel[0])
            
    # Removing elements present in other list
    bad_channels = [i for i in raw_channels if i not in good_raw_channels]
    
    return bad_channels
