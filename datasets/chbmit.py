"""
Dataset classes for the Children Hospital Boston - Massachusetts Instistute of Technology (MIT) EEG Corpus.
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
from .subject_info_chbmit import subject_info_CHBMIT

class CHBMIT(BaseConcatDataset):
    """Childrens' Hospital Boston - MIT (CHB-MIT) EEG Corpus
    (https://physionet.org/content/chbmit/1.0.0/chb01/#files-panel).

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
        # If debug is true and the number of edf files is bigger than 100, select 
        # the first 100 file_paths        
        if (debug and len(file_paths)>100):
            file_paths = file_paths[0:100]
        
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
        # disable joblib for tests. mocking seems to fail otherwise
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
        age, gender = _parse_age_and_gender_from_txt_file(file_path)
        raw = mne.io.read_raw_edf(file_path, preload=preload, verbose='ERROR')
        
    
        # if using TUSZ, extract the annotations here and add them to the raw files 
        # probably more efficient than doing it in the TUSZ class
        tokens = file_path.split(os.sep) 
        
        # default
        recording_type_TERM = 'background' 
        recording_type_EVENT = None # the chbmit dataset does not contain channel specific annotations
        
        # if the file 'file_path.seizures' exist, then that file contains seizures and annotations need to be created
        file_path_seizure = file_path + ".seizures"
        if os.path.isfile(file_path_seizure):
            print(file_path)

            recording_type_TERM = 'seizure' 
            annotations = _parse_term_based_annotations_from_txt_file(file_path)  
            raw = raw.set_annotations(annotations, on_missing='warn')
        
        
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


        
def _parse_term_based_annotations_from_txt_file(file_path):
    # From a path such as : live/chbmit/physionet.org/files/chbmit/1.0.0/chb20/chb20_12.edf,
    # get: live/chbmit/physionet.org/files/chbmit/1.0.0/chb20/chb20-summary.txt
    file_path = os.path.normpath(file_path)
    tokens = file_path.split(os.sep)
    final_token = tokens[-1] # chb20_12.edf
    if "chb17" in final_token:
        new_final_token = "chb17-summary.txt"
    else:
        new_final_token = tokens[-1].split('_')[0] + "-summary.txt" # chb20_summary.txt
    summary_text_path = file_path.replace(final_token, new_final_token)
    
    
    # store the summary file line by line in a list
    with open(summary_text_path) as file:
        summary_list = file.readlines()
    
    sz_onset = []
    sz_duration = []
    sz_description = []
    
    if new_final_token != 'chb24-summary.txt':
        for lines in summary_list:
            if lines == 'File Name: '+final_token+'\n':
                # print(lines)
                index_of_interest = summary_list.index(lines)
                # Find the number of seizure in the edf file, extract their start time and duration
                number_of_seizures = int(summary_list[index_of_interest+3].replace('Number of Seizures in File: ',''))
                for seizure_index in range(number_of_seizures): 
                    seizure_index += 1
                    sz_end_time = int((summary_list[index_of_interest+3 + 2*(seizure_index)]).split()[-2])
                    sz_start_time = int((summary_list[index_of_interest+3 + 2*(seizure_index) - 1]).split()[-2])

                    sz_onset.append(sz_start_time)
                    sz_duration.append(sz_end_time-sz_start_time)
                    sz_description.append("Seizure "+str(seizure_index))
    else: 
        for lines in summary_list:
            if lines == 'File Name: '+final_token+'\n':
                # print(lines)
                index_of_interest = summary_list.index(lines)
                # Find the number of seizure in the edf file, extract their start time and duration
                number_of_seizures = int(summary_list[index_of_interest+1].replace('Number of Seizures in File: ',''))
                for seizure_index in range(number_of_seizures): 
                    seizure_index += 1
                    sz_end_time = int((summary_list[index_of_interest+1 + 2*(seizure_index)]).split()[-2])
                    sz_start_time = int((summary_list[index_of_interest+1 + 2*(seizure_index) - 1]).split()[-2])

                    sz_onset.append(sz_start_time)
                    sz_duration.append(sz_end_time-sz_start_time)
                    sz_description.append("Seizure "+str(seizure_index))
                    

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
    # file path looks life that: "../chbmit/physionet.org/files/chbmit/1.0.0/chb01/chb01_03.edf"
    
    file_path = os.path.normpath(file_path)
    tokens = file_path.split(os.sep)
    
    # Extract version number, should always be 1.0.0 for chbmit as there was only 1 release
    version = tokens[-3]  
    
    # Extract the subject id
    subject_id = tokens[-2]  # chb01
    
    # For the CHBMIT, the data from a single patient is always obtained in a 'continuous' manner, the lag between the edf recording is ususally 10 sec, but can be longer (cf their website). I assume they are always at session 0 and have different segments. 
    session = 0
    segment = tokens[-1].split('_')[1].split('.')[0]
    
    # Note: there are paths like: '/chbmit/1.0.0/chb02/chb02_16+.edf', so unless you manually remove the +, the segment number should be a string, and cannot be casted as an int for all instances  
    
    # Note: dates in the edf files are dummy dates, so no need to store them for this dataset
    description = {
        'path': file_path,
        'version': version,
        'subject': subject_id,
        'session': session,
        'segment': segment
    }
    
    return description
    

def _parse_age_and_gender_from_txt_file(file_path):
    
    file_path = os.path.normpath(file_path)
    tokens = file_path.split(os.sep)
    # Extract the subject id
    subject_id = tokens[-2]  # chb01
    # find the corresponding age and gender
    [gender, age] = subject_info_CHBMIT[subject_id]
    
    return age, gender




