import os
import streamlit as st
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from streamlit_utils.app_utils import label_finder

ecg_base_path = 'C:/Data/ECG/will_two_do_2021/data'
df = pd.read_csv('./data/5_wtd_10seconds.csv')

import numpy as np
import random

def random_scale(ecg, scale_range=(0.9, 1.1)):
    """Amplitude scaling for numpy ECG [T, 12]."""
    scale = np.random.uniform(*scale_range)
    return ecg * scale

def random_shift(ecg, shift_range=(-0.1, 0.1)):
    """Baseline shift."""
    shift = np.random.uniform(*shift_range)
    return ecg + shift

def random_gaussian_noise(ecg, sigma_range=(0.0, 0.01)):
    """Add Gaussian noise."""
    sigma = np.random.uniform(*sigma_range)
    noise = np.random.randn(*ecg.shape) * sigma
    return ecg + noise

def random_crop(ecg, crop_size):
    """Random crop along time axis."""
    T, _ = ecg.shape
    if crop_size >= T:
        return ecg
    start = np.random.randint(0, T - crop_size)
    return ecg[start:start + crop_size, :]

def random_time_warp(ecg, max_stretch=0.1):
    """Random time stretch without changing length."""
    T, C = ecg.shape
    stretch = np.random.uniform(1 - max_stretch, 1 + max_stretch)
    new_T = int(T * stretch)

    # Resample to new length (stretch)
    warped = np.zeros((new_T, C))
    for lead in range(C):
        warped[:, lead] = np.interp(
            np.linspace(0, T - 1, new_T),
            np.arange(T),
            ecg[:, lead]
        )

    # Resize back to original length
    out = np.zeros_like(ecg)
    for lead in range(C):
        out[:, lead] = np.interp(
            np.linspace(0, new_T - 1, T),
            np.arange(new_T),
            warped[:, lead]
        )

    return out

def random_lead_dropout(ecg, p=0.1):
    """Randomly zero out some leads."""
    mask = np.random.rand(ecg.shape[1]) > p
    return ecg * mask[None, :]

def load_ecg(record, ecg_name):
    p_signal = record.p_signal
    
    if st.session_state.random_scale:
        p_signal = random_scale(p_signal, (0.85, 1.15))
    if st.session_state.random_shift:
        p_signal = random_shift(p_signal, (-0.5, 0.5))
    if st.session_state.random_gaussian_noise:
        p_signal = random_gaussian_noise(p_signal, (0.0, 0.2))
    if st.session_state.random_time_warp:
        p_signal = random_time_warp(p_signal, max_stretch=0.8)
    if st.session_state.random_lead_dropout:
        p_signal = random_lead_dropout(p_signal, p=0.1)
    
    ecg = []
    for ix, _ in enumerate(record.sig_name[:12]): # type: ignore
        num_signals = record.fs * 5 # type: ignore
        signals = p_signal[:num_signals, ix] # type: ignore
        signals = signals.reshape(500, (record.fs//100)).mean(axis=1) # type: ignore
        ecg.append(signals)
    ecg = np.array(ecg)
    
    label_list = label_finder(record.comments[2]) # type: ignore
    actual_labels = '\n- '.join(label_list)
    
    st.success(f'🎉 ECG "{ecg_name}" successfully processed!')
        
    st.info(f"""
            **Actual** interpretation:\n- {actual_labels}
            """)
    
    #### Plot settings
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    lead_order = [0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11]  # I, aVR, II, aVL, III, aVF, V1, V4, V2, V5, V3, V6

    # Parameters
    n_samples = 500  # Number of samples per lead
    sampling_rate = 100  # Hz
    duration = n_samples / sampling_rate
    time = np.linspace(0, duration, n_samples)  # Time array with 500 samples
    
    # Create a 6x2 grid of subplots
    fig, axes = plt.subplots(6, 2, figsize=(10, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    # Plot each lead
    for subplot_idx, lead_idx in enumerate(lead_order):
        ax = axes[subplot_idx]
        # Set light pink background to mimic ECG paper
        ax.set_facecolor('#ffe6e6')
        
        # Plot ECG signal
        ax.plot(time, ecg[lead_idx], color='red', linewidth=1)
        
        # Set title to lead name
        ax.set_title(lead_names[lead_idx], loc='left', fontsize=10, fontweight='bold')
        
        # Major grid: 0.5 mV (5 mm) and 0.2 s (5 mm at 25 mm/s)
        ax.grid(True, which='major', linestyle='-', linewidth=0.8, color='gray', alpha=0.7)
        ax.set_yticks(np.arange(-2, 2.1, 0.5))  # 0.5 mV steps
        ax.set_xticks(np.arange(0, duration + 0.2, 0.2))  # 0.2 s steps
        
        # Minor grid: 0.1 mV (1 mm) and 0.04 s (1 mm at 25 mm/s)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.4, color='gray', alpha=0.4)
        ax.set_yticks(np.arange(-2, 2.1, 0.1), minor=True)
        ax.set_xticks(np.arange(0, duration + 0.04, 0.04), minor=True)
        
        # Hide tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Set axis limits
        ax.set_ylim(-2, 2)  # Typical ECG range in mV
        ax.set_xlim(0, duration)
        
        # Remove spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="ECGMaster Viewer", page_icon="👨‍⚕️")
    st.title("👨‍⚕️ ECGMaster Viewer")
    st.header("Upload your ECG to see what is going on!")
    
    if 'ecg_loaded' not in st.session_state:
        st.session_state.ecg_loaded = False
    if 'random_scale' not in st.session_state:
        st.session_state.random_scale = False
    if 'random_shift' not in st.session_state:
        st.session_state.random_shift = False
    if 'random_gaussian_noise' not in st.session_state:
        st.session_state.random_gaussian_noise = False
    if 'random_crop' not in st.session_state:
        st.session_state.random_crop = False
    if 'random_time_warp' not in st.session_state:
        st.session_state.random_time_warp = False
    if 'random_lead_dropout' not in st.session_state:
        st.session_state.random_lead_dropout = False
    
    # Sidebar widgets
    st.sidebar.header("Load ECG")
    ecg_name = st.sidebar.text_input('ECG Name')
    if st.sidebar.button('Load') and ecg_name != "":
        st.session_state.ecg_loaded = True

    if st.session_state.ecg_loaded:
        patient_record_path = ecg_base_path + df[df['patient_id']==ecg_name]['record_path'].iloc[0]
        record = wfdb.rdrecord(patient_record_path)
        
        st.sidebar.success("Files loaded successfully!")
        st.sidebar.checkbox('random scale', key='random_scale')
        st.sidebar.checkbox('random shift', key='random_shift')
        st.sidebar.checkbox('random gaussian noise', key='random_gaussian_noise')
        st.sidebar.checkbox('random crop', key='random_crop')
        st.sidebar.checkbox('random time warp', key='random_time_warp')
        st.sidebar.checkbox('random lead dropout', key='random_lead_dropout')
        
        load_ecg(record=record, ecg_name=ecg_name)
    else:
        st.info("👈 Please specify your Patient ID in sidebar to load ECG.")

    # Expander with additional info
    with st.expander("About the app"):
        st.write("""
            This is an interface to view ECG data.\n
            Design and development by Dr. Alireza Soheilipour
        """)

if __name__ == "__main__":
    main()
    