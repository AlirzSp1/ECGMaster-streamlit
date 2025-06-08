import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from streamlit_utils.firestore_utils import init_firestore

db = init_firestore()
docs = db.collection('ecg_data_1').stream()
ecg_id_list = ['--'] + [doc.id for doc in docs]

def main():
    st.set_page_config(page_title="ECGMaster Uploader", page_icon="👨‍⚕️")
    st.title("👨‍⚕️ ECGMaster Uploader")

    st.header("Upload your ECG to see what is going on!")

    # Sidebar widgets
    st.sidebar.header("Select patient")
    ecg_name = st.sidebar.selectbox('Select a patient', ecg_id_list)
    btn_load = st.sidebar.button('Load', type='secondary')
    
    if btn_load:
        ecg_dict = db.collection('ecg_data_1').document(ecg_name).get().to_dict()
        st.sidebar.success("🎉 Patient loaded successfully!")
        
        st.sidebar.text('your feedback:')
        fb_thumb = st.sidebar.feedback("thumbs")
        st.sidebar.text('your comment if needed:')
        comment_text = st.sidebar.text_area('optional')
        btn_submit = st.sidebar.button('Submit!', type='secondary')
        
        ecg = np.array(ecg_dict['signals_flat'])
        ecg = ecg.reshape(12, 500)
        
        actual_labels = ecg_dict['actual_labels']            
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
        st.info("👈 Select a patient in sidebar.")

    # Expander with additional info
    with st.expander("About the app"):
        st.write("""
            This is a sample app to analyze ECG papers.\n
            Design and development by Dr.Alireza Soheilipour
        """)

if __name__ == "__main__":
    main()
    