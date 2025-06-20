import os
import streamlit as st
import wfdb
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from streamlit_utils.firestore_utils import init_firestore
from streamlit_utils.app_utils import label_finder

db = init_firestore()

def main():
    st.set_page_config(page_title="ECGMaster Uploader", page_icon="👨‍⚕️")
    st.title("👨‍⚕️ ECGMaster Uploader")

    st.header("Upload your ECG to see what is going on!")

    # Sidebar widgets
    st.sidebar.header("Load files")
    st.sidebar.write('Please load the files with same names:')
    ecg_mat = st.sidebar.file_uploader(
        '.mat file', type='mat',
        accept_multiple_files=False,
        help="Load MAT file",
        
    )   
    ecg_hea = st.sidebar.file_uploader(
        '.hea file', type='hea',
        accept_multiple_files=False,
        help="Load HEA file"
    )

    if ecg_mat is not None and ecg_hea is not None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save BOTH files to the SAME temporary directory
            mat_path = os.path.join(tmp_dir, ecg_mat.name)
            hea_path = os.path.join(tmp_dir, ecg_hea.name)
            
            with open(mat_path, "wb") as f_mat, open(hea_path, "wb") as f_hea:
                f_mat.write(ecg_mat.getbuffer())
                f_hea.write(ecg_hea.getbuffer())
            
            # Verify filenames match (excluding extensions)
            if ecg_mat.name[:-4] == ecg_hea.name[:-4]:
                ecg_name = os.path.join(tmp_dir, ecg_mat.name[:-4])
                st.sidebar.success("Files loaded successfully!")
                
                record = wfdb.rdrecord(ecg_name)
                
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
                doc_ref = db.collection("ecg_data").document(ecg_mat.name[:-4])
                doc_ref.set({
                    "signals_flat": ecg.flatten().tolist(),  # 1D list
                    "shape": list(ecg.shape),
                    "label_list": label_list,
                    "eval": False
                })
                st.success('🎉 Files successfully processed and added to database!')
                    
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

                # Adjust layout to prevent overlap
                plt.tight_layout()

                # Display in Streamlit
                st.pyplot(fig)
            else:
                st.sidebar.error('File names must match (e.g., "00001.hea" and "00001.mat")')
    else:
        st.info("👈 Both **MAT** and **HEA** files are needed. Please load them in sidebar.")

    # Expander with additional info
    with st.expander("About the app"):
        st.write("""
            This is an interface for uploading ECG data.\n
            Design and development by Dr. Alireza Soheilipour
        """)

if __name__ == "__main__":
    main()
    