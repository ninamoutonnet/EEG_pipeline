'''
This file contains the information about the subjects of the Siena dataset. Contains patient_id, age_years, gender, seizure, localization, lateralization, eeg_channel, number_seizures, rec_time_minutes

See: https://physionet.org/content/siena-scalp-eeg/1.0.0/

'The file subject_info.csv contains, for each subject, the gender and age, the seizure classification according to the criteria of the International League Against Epilepsy, the number of EEG channels, the number of seizures and the total recording time in minutes. 

- IAS is focal onset impaired awareness; 
- WIAS is focal onset without impaired awareness; 
- FBTC is focal to bilateral tonic-clonic; 
- T is temporal; 
- R is right; 
- L is left. 

In total, the database contains 47 seizures on about 128 recording hours.'

'''


subject_info_Siena = {'PN00': [55,'M','IAS','T','R',29,5,198],
                    'PN01': [46,'M','IAS','T','L',29,2,809],
                    'PN03': [54,'M','IAS','T','R',29,2,752],
                    'PN05': [51,'F','IAS','T','L',29,3,359],
                    'PN06': [36,'M','IAS','T','L',29,5,722],
                    'PN07': [20,'F','IAS','T','L',29,1,523],
                    'PN09': [27,'F','IAS','T','L',29,3,410],
                    'PN10': [25,'M','FBTC','F','Bilateral',20,10,1002],
                    'PN11': [58,'F','IAS','T','R',29,1,145],
                    'PN12': [71,'M','IAS','T','L',29,4,246],
                    'PN13': [34,'F','IAS','T','L',29,3,519],
                    'PN14': [49,'M','WIAS','T','L',29,4,1408],
                    'PN16': [41,'F','IAS','T','L',29,2,303],
                    'PN17': [42,'M','IAS','T','R',29,2,308]}

