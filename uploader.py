import wfdb
import pandas as pd
import numpy as np
from streamlit_utils.firestore_utils import init_firestore
from streamlit_utils.app_utils import label_finder
import random

db = init_firestore()
df = pd.read_csv('./data/5_wtd_10seconds.csv')

ecg_base_path = 'C:/Data/ECG/will_two_do_2021/data'

patient_id_list = df[(df['dataset'] == 'cpsc_2018') & (df['g2_164931005'] == 1)]['patient_id'].to_list()
patient_id_list_rand = random.sample(patient_id_list, 40)

for patient_id in patient_id_list_rand:
    patient_record_path = ecg_base_path + df[df['patient_id']==patient_id]['record_path'].iloc[0]
    record = wfdb.rdrecord(patient_record_path)
    
    # Process ECG signals
    ecg = []
    for ix, _ in enumerate(record.sig_name[:12]): # type: ignore
        num_signals = record.fs * 5 # type: ignore
        signals = record.p_signal[:num_signals, ix] # type: ignore
        signals = signals.reshape(500, (record.fs//100)).mean(axis=1) # type: ignore
        ecg.append(signals)
    ecg = np.array(ecg)
    
    label_list = label_finder(record.comments[2]) # type: ignore
    actual_labels = '\n- '.join(label_list)
    
    # Save to Firestore
    doc_ref = db.collection("ecg_data_ste").document(patient_id)
    doc_ref.set({
        "signals_flat": ecg.flatten().tolist(),  # 1D list
        "shape": list(ecg.shape),
        "label_list": label_list,
        "eval": False
    })