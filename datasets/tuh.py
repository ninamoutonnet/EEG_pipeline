"""
Dataset classes for the Temple University Hospital (TUH) EEG Corpus and the
TUH Abnormal EEG Corpus.
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)


########################################################
#      Adapted by Nina Moutonnet <nm2318@ic.ac.uk> 
#          1.  removed the mock classes
#  2.  added the debug option when creating TUH class
# 3. In _parse_description_from_file_path, generate a 
# error if the version is old (before december 2022)
# 4. remove the read_date functionality that creates 
#    a txt file with the date 
# 5. add_physician_reports is removed as from version 
#    Starting with v1.5.4 (we are on v2.0.0), 
#     we have not been distributing  reports
#        6. added a tusz class
#   7. added cvs_bi extraction of info for tusz
# 8. added csv file extraction of annotation for tusz, 
# however this does not work yet as channels present 
# in raw file and that of annotation do not match
# 9. added debug and remove_unknown_channels and 
#    _set_bipolar_tcp
########################################################

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
from .channel_clustering_TUH import channels_to_remove
from .montage_TUSZ import cathode, anode, bipolar_ch_names 

class TUH(BaseConcatDataset):
    """Temple University Hospital (TUH) EEG Corpus
    (www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tueg).

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
        Can be 'gender', or 'age'.
    preload: bool
        If True, preload the data of the Raw objects.
    n_jobs: int
        Number of jobs to be used to read files in parallel. 
    debug: bool
        If True, only select the first 100 files in the files paths option
        Added by Nina, check that this can be adapted for other TUH repositories
    remove_unknown_channels: bool
        If true, remove the channels that are unknown/contain no signal/are custom placement
        This is provided in the appendix of https://isip.piconepress.com/publications/reports/2020/tuh_eeg/electrodes/
    set_bipolar_tcp: bool
        If true, you set the bipolar reference to a tcp montage
    """
    def __init__(self, path, recording_ids=None, target_name=None,
                 preload=False, n_jobs=1, debug=False, remove_unknown_channels=False, set_bipolar_tcp=False):
            
        # create an index of all files and gather easily accessible info
        # without actually touching the files
        file_paths = glob.glob(os.path.join(path, '**/*.edf'), recursive=True)
        # If debug is true and the number of edf files is bigger than 100, select 
        # the first 100 file_paths
        if (debug and len(file_paths)>100):
            file_paths = file_paths[0:100]
        
        descriptions = _create_description(file_paths)
        # sort the descriptions chronologicaly
        descriptions = _sort_chronologically(descriptions)
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
            base_datasets = [self._create_dataset(descriptions[i], target_name, preload, remove_unknown_channels, 
                                                  set_bipolar_tcp)
                for i in descriptions.columns]
        else:
            base_datasets = Parallel(n_jobs)(delayed(
                self._create_dataset)(descriptions[i], target_name, preload, remove_unknown_channels, set_bipolar_tcp) for i 
                                             in descriptions.columns)
        super().__init__(base_datasets)

    @staticmethod
    def _create_dataset(description, target_name, preload, remove_unknown_channels, set_bipolar_tcp):
        file_path = description.loc['path']

        # parse age and gender information from EDF header
        age, gender = _parse_age_and_gender_from_edf_header(file_path)
        raw = mne.io.read_raw_edf(file_path, preload=preload, verbose='WARNING')
        
        # If remove_unknown_channels is set to True, remove the 'bad' channels at this step
        if remove_unknown_channels:
            raw = _remove_unknown_channels(raw, channels_to_remove)
        
        # If set_bipolar_tcp is true, re-reference. As this depends on the montage, extract that information first
        if set_bipolar_tcp: 
            raw = _set_bipolar_tcp(raw, file_path)
        
        # if using TUSZ, extract the annotations here and add them to the raw files 
        # probably more efficient than doing it in the TUSZ class
        tokens = file_path.split(os.sep) 
        recording_type_TERM = 'background' 

        if ('tuh_eeg_seizure') in tokens:
            annotation_csvbi = _parse_term_based_annotations_from_csv_bi_file(file_path)
            for annotation in annotation_csvbi:
                if 'seiz' in annotation['description']:
                    recording_type_TERM = 'seizure'
                    
            # annotation_csv does not work if the ch_names do not match the raw file channels
            # so it needs to have a tcp re-reference the raw file 
            if set_bipolar_tcp: 
                annotation_csv = _parse_term_based_annotations_from_csv_file(file_path)
                annotations = annotation_csvbi.append(onset=annotation_csv.onset, 
                                                 duration=annotation_csv.duration, 
                                                 description=annotation_csv.description,
                                                 ch_names=annotation_csv.ch_names)
                raw = raw.set_annotations(annotations, on_missing='warn')
            else:
                raw = raw.set_annotations(annotation_csvbi, on_missing='warn')
            
            
            
        meas_date = datetime(1, 1, 1, tzinfo=timezone.utc) \
            if raw.info['meas_date'] is None else raw.info['meas_date']
        # if this is old version of the data and the year could be parsed from
        # file paths, use this instead as before
        if 'year' in description:
            meas_date = meas_date.replace(*description[['year', 'month', 'day']])
        raw.set_meas_date(meas_date)

        d = {
            'age': int(age),
            'gender': gender,
            'pathological': recording_type_TERM
        }
        # if year exists in description = old version
        # if not, get it from meas_date in raw.info and add to description
        # if meas_date is None, create fake one
        if 'year' not in description:
            d['year'] = raw.info['meas_date'].year
            d['month'] = raw.info['meas_date'].month
            d['day'] = raw.info['meas_date'].day
            
        additional_description = pd.Series(d)
        description = pd.concat([description, additional_description])
        base_dataset = BaseDataset(raw, description,
                                   target_name=target_name)
        return base_dataset

def _set_bipolar_tcp(raw, file_path):
    # find the montage from the name of the file
    if '/01_tcp_ar/' in file_path: 
        index = 0
    elif '/02_tcp_le/' in file_path: 
        index = 1
    elif '/03_tcp_ar_a/' in file_path: 
        index = 2
    elif '/04_tcp_le_a/' in file_path: 
        index = 3
    
    sample_anode = anode[index]
    sample_cathode = cathode[index]
    sample_bipolar_ch_names = bipolar_ch_names[index]
    
    raw = mne.set_bipolar_reference(raw, anode=sample_anode, cathode=sample_cathode, ch_name=sample_bipolar_ch_names, 
                                    verbose='WARNING')
    
    return raw

def _remove_unknown_channels(raw, channels_to_remove):
    return raw.drop_channels(ch_names = channels_to_remove, on_missing='ignore')  
   
        
def _parse_term_based_annotations_from_csv_bi_file(file_path):
    csv_bi_path = file_path.replace('.edf', '.csv_bi')
    csvbi_file = pd.read_csv(csv_bi_path, header=5)   
    # at the top of every file, there is a header that should not be read, example of header below
    ###########################################################
    # version = csv_v1.0.0
    # bname = aaaaaajy_s001_t000
    # duration = 1750.00 secs
    # montage_file = $NEDC_NFC/lib/nedc_eas_default_montage.txt
    #
    ###########################################################
    csvbi_file['duration'] =  csvbi_file.stop_time - csvbi_file.start_time
    csvbi_file['description'] = csvbi_file.label + ',' + csvbi_file.confidence.astype(str) + ',' +  'TERM'

    csvbi_annotations = mne.Annotations(onset = csvbi_file['start_time'].tolist(), 
                                        duration = csvbi_file['duration'].tolist(), 
                                        description = csvbi_file['description'].tolist())  
    return csvbi_annotations


def _parse_term_based_annotations_from_csv_file(file_path): 
    csv_path = file_path.replace('.edf', '.csv')
    csv_file =  pd.read_csv(csv_path, header=5) 
    # at the top of every file, there is a header that should not be read, example of header below
    ###########################################################
    # version = csv_v1.0.0
    # bname = aaaaaajy_s001_t000
    # duration = 1750.00 secs
    # montage_file = $NEDC_NFC/lib/nedc_eas_default_montage.txt
    #
    ###########################################################
    
    csv_file['duration'] =  csv_file.stop_time - csv_file.start_time
    csv_file['description'] = csv_file.label + ',' + csv_file.confidence.astype(str) + ',' +  'EVENT'
    csv_file = csv_file.sort_values(by = ['start_time', 'stop_time'], axis = 0) 
    csv_file.reset_index(inplace=True, drop=True)

    onsets = []
    durations = []
    descriptions = []
    ch_names = []

    while len(csv_file) > 0:
        temp_dataframe = pd.DataFrame()
        temp_dataframe = pd.concat([temp_dataframe, csv_file.iloc[[0]]], ignore_index=True)
        csv_file.drop([0], inplace=True)

        # get the values and check they match 
        start_time = str(temp_dataframe['start_time'][0])
        duration = str(temp_dataframe['duration'][0])

        #iterate through rows looking for identical start time and duration
        for index, row in csv_file.iterrows():        
            if( (str(row['start_time'])==start_time) and (str(row['duration'])==duration)   ):
                # if you have found a match, add it to the temporary dataframe and remove it from the csv_file dataframe 
                temp_dataframe = pd.concat([temp_dataframe, csv_file.loc[[index]]], ignore_index=True)
                csv_file.drop([index], inplace=True)

        onsets.append(temp_dataframe['start_time'][0]) 
        durations.append(temp_dataframe['duration'][0])
        descriptions.append(temp_dataframe['description'][0])
        ch_names.append(temp_dataframe['channel'].tolist())
        csv_file.reset_index(drop=True, inplace=True)
    
    csv_annotations = mne.Annotations(onset = onsets, 
                                    duration = durations, 
                                    description = descriptions,
                                    ch_names=ch_names)  
    return csv_annotations

    

def _create_description(file_paths):
    descriptions = [_parse_description_from_file_path(f) for f in file_paths]
    descriptions = pd.DataFrame(descriptions)
    return descriptions.T


def _sort_chronologically(descriptions):
    descriptions.sort_values(
        ["year", "month", "day", "subject", "session", "segment"],
        axis=1, inplace=True)
    return descriptions


def _read_date(file_path):
    date_path = file_path.replace('.edf', '_date.txt')
    # if date file exists, read it
    if os.path.exists(date_path):
        description = pd.read_json(date_path, typ='series').to_dict()
    # otherwise read edf file, extract date and store to file
    else:
        raw = mne.io.read_raw_edf(file_path, preload=False, verbose='error')
        description = {
            'year': raw.info['meas_date'].year,
            'month': raw.info['meas_date'].month,
            'day': raw.info['meas_date'].day,
        }
        # if the txt file storing the recording date does not exist, create it
        # Nina removed that, no need to write a txt file if you can extract it in header file
        # on top of that, that creates issues when using the _parse_description_from_file_path
        # as there is more than 0 or 1 txt file in the directory
        #try:
            #pd.Series(description).to_json(date_path)
        #except OSError:
            #warnings.warn(f'Cannot save date file to {date_path}. '
                          #f'This might slow down creation of the dataset.')
    return description


def _parse_description_from_file_path(file_path):
    # stackoverflow.com/questions/3167154/how-to-split-a-dos-path-into-its-components-in-python  # noqa
    file_path = os.path.normpath(file_path)
    tokens = file_path.split(os.sep)
    # Extract version number and tuh_eeg_abnormal/tuh_eeg from file path
    if ('tuh_eeg_abnormal' in tokens): #
        abnormal = True
        # Tokens[-2] is channel configuration (always 01_tcp_ar in abnormal)
        # on new versions, or session (e.g. s004_2013_08_15) on old versions
        if tokens[-2].split('_')[0][0] == 's':  # s denoting session number
            version = tokens[-9]  # Before dec 2022 updata
        else:
            version = tokens[-6]  # After the dec 2022 update
        
    else:  # tuh_eeg or tuh_eeg_sz
        abnormal = False
        version = tokens[-7]
    v_number = int(version[1])

    if (abnormal and v_number >= 3) or ((not abnormal) and v_number >= 2):
        # New file path structure for versions after december 2022,
        # expect file paths as
        # tuh_eeg/v2.0.0/edf/000/aaaaaaaa/s001_2015_12_30/01_tcp_ar/aaaaaaaa_s001_t000.edf
        # or for abnormal:
        # tuh_eeg_abnormal/v3.0.0/edf/train/normal/01_tcp_ar/aaaaaaav_s004_t000.edf
        # or for seizure: 
        # tuh_eeg_seizure/v2.0.0/dev/aaaaaajy/s001_2003_07_16/02_tcp_le/aaaaaajy_s001_t000.edf
        subject_id = tokens[-1].split('_')[0]
        session = tokens[-1].split('_')[1]
        segment = tokens[-1].split('_')[2].split('.')[0]
        description = _read_date(file_path)
        description.update({
            'path': file_path,
            'version': version,
            'subject': subject_id,
            'session': int(session[1:]),
            'segment': int(segment[1:]),
            'channel_config': tokens[-2],
        })
        if not abnormal:
            year, month, day = tokens[-3].split('_')[1:]
            description['year'] = int(year)
            description['month'] = int(month)
            description['day'] = int(day)

        return description
    else:        
        warnings.warn('The version of the tuh dataset is too old, please update it for data loading to work')


def _read_edf_header(file_path):
    f = open(file_path, "rb")
    header = f.read(88)
    f.close()
    return header


def _parse_age_and_gender_from_edf_header(file_path):
    header = _read_edf_header(file_path)
    # bytes 8 to 88 contain ascii local patient identification
    # see https://www.teuniz.net/edfbrowser/edf%20format%20description.html
    patient_id = header[8:].decode("ascii")
    age = -1
    found_age = re.findall(r"Age:(\d+)", patient_id)
    if len(found_age) == 1:
        age = int(found_age[0])
    gender = "X"
    found_gender = re.findall(r"\s([F|M])\s", patient_id)
    if len(found_gender) == 1:
        gender = found_gender[0]
    return age, gender


class TUHAbnormal(TUH):
    """Temple University Hospital (TUH) Abnormal EEG Corpus.
    see www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tuab

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
        Can be 'pathological', 'gender', or 'age'.
    preload: bool
        If True, preload the data of the Raw objects.
    debug: bool
        If True, only select the first 100 files in the files paths option
        Added by Nina
    """
    def __init__(self, path, recording_ids=None, target_name='pathological',
                 preload=False, n_jobs=1, debug=False, remove_unknown_channels=False, set_bipolar_tcp=False):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*not in description. '__getitem__'")
            super().__init__(path=path, recording_ids=recording_ids,
                             preload=preload, target_name=target_name,
                             n_jobs=n_jobs, debug=debug, remove_unknown_channels=remove_unknown_channels, 
                             set_bipolar_tcp=set_bipolar_tcp)
        additional_descriptions = []
        for file_path in self.description.path:
            additional_description = (self._parse_additional_description_from_file_path(file_path))
            additional_descriptions.append(additional_description)
        additional_descriptions = pd.DataFrame(additional_descriptions)
        self.set_description(additional_descriptions, overwrite=True)

    @staticmethod
    def _parse_additional_description_from_file_path(file_path):
        file_path = os.path.normpath(file_path)
        tokens = file_path.split(os.sep)
        # expect paths as version/file type/data_split/pathology status/reference/subset/subject/recording session/file
        # e.g.            v2.0.0/edf/train/normal/01_tcp_ar/000/00000021/s004_2013_08_15/00000021_s004_t000.edf
        
        # Nina changed this because the naming system changed
        # path we use:  ~/projects/scott_data_tuh/live/tuh_eeg_abnormal/v3.0.0/edf/train/abnormal/01_tcp_ar/aaaaapve_s001_t000.edf 
        
        assert ('abnormal' in tokens or 'normal' in tokens), ('No pathology labels found.')
        assert ('train' in tokens or 'eval' in tokens), ('No train or eval set information found.')
        return {
            'version': tokens[-6],
            'set': tokens[-4],
            'pathological': 'abnormal' in tokens,
            'channel_config': tokens[-2]
        }


    

class TUHSeizure(TUH):
    """Temple University Hospital (TUH) Seizure EEG Corpus.
    see https://isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_seizure/
    
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
        Can be 'seizure', 'gender', or 'age'.  # target name(s) need to be read from description
    preload: bool
        If True, preload the data of the Raw objects.
    debug: bool
        If True, only select the first 100 files in the files paths option
        Added by Nina
   
    """
    def __init__(self, path, recording_ids = None, target_name=None, preload = False, n_jobs=1, debug = False, 
                 remove_unknown_channels=False, set_bipolar_tcp=False):
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*not in description. '__getitem__'")
            super().__init__(path=path, recording_ids=recording_ids,
                             preload=preload, target_name=target_name,
                             n_jobs=n_jobs, debug=debug, remove_unknown_channels=remove_unknown_channels, 
                             set_bipolar_tcp=set_bipolar_tcp)
            
        additional_descriptions = []
        for file_path in self.description.path:
            additional_description = (self._parse_additional_description_from_file_path(file_path))
            additional_descriptions.append(additional_description)
           
        additional_descriptions = pd.DataFrame(additional_descriptions)
        self.set_description(additional_descriptions, overwrite=True)

    @staticmethod
    def _parse_additional_description_from_file_path(file_path):
        file_path = os.path.normpath(file_path)
        tokens = file_path.split(os.sep)
        # expect paths to look like: 
        # Nina changed this because the naming system changed
        # path we use:  ~/projects/scott_data_tuh/live/tuh_eeg_seizure/v2.0.0/edf/dev/aaaaaajy/s001_2003_07_16/02_tcp_le/aaaaaajy_s001_t000.edf
        
        assert ('train' in tokens or 'eval' in tokens or 'dev' in tokens), ('No train or eval or dev set information found in the path name.')
        return {
            'set': tokens[-5],
            'channel_config': tokens[-2]
        }

    

