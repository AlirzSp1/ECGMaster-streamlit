import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from streamlit_utils.firestore_utils import init_firestore
import datetime

# Initialize Firestore
db = init_firestore()
docs = db.collection('ecg_data').stream()
ecg_id_list = [None] + [doc.id for doc in docs]

def main():
    st.set_page_config(page_title="ECGMaster Evaluator", page_icon="👨‍⚕️")
    st.title("👨‍⚕️ ECGMaster Evaluator")

    st.header("Select your patient to see what is going on!")
    
    st.session_state.username = ""

    # Initialize session state
    if "ecg_select" not in st.session_state:
        st.session_state.ecg_select = False
    if "ecg_dict" not in st.session_state:
        st.session_state.ecg_dict = None
    if "ecg_loaded" not in st.session_state:
        st.session_state.ecg_loaded = False
    if "fb_thumb" not in st.session_state:
        st.session_state.fb_thumb = None
    if "fb_comment" not in st.session_state:
        st.session_state.fb_comment = ""

    # Sidebar widgets
    # Define Users
    def update_username():
        st.session_state.username = st.session_state.temp_username
    st.sidebar.text_input('Username:', value=st.session_state.username, key="temp_username", on_change=update_username)
        
    st.sidebar.header("Select patient")
    st.sidebar.selectbox('Select a patient', ecg_id_list, key="ecg_select")
    btn_load = st.sidebar.button('Load', type='secondary')

    # Load ECG data when button is clicked
    if btn_load and st.session_state.ecg_select:
        act_ecg_dict = db.collection('ecg_data').document(st.session_state.ecg_select)
        st.session_state.ecg_dict = act_ecg_dict.get().to_dict()
        st.session_state.ecg_loaded = True
        st.session_state.fb_thumb = None  # Reset feedback
        st.session_state.fb_comment = ""   # Reset comment
        st.sidebar.success("🎉 Loaded successfully!")

    # Display ECG and feedback widgets if data is loaded
    if st.session_state.ecg_loaded and st.session_state.ecg_dict:
        ecg_dict = st.session_state.ecg_dict
        if ecg_dict['eval']:
            st.sidebar.warning('This patient was evaluated before.')
            
        # Feedback widgets
        st.sidebar.text('Your feedback:')
        def update_feedback():
            st.session_state.fb_thumb = st.session_state.temp_fb_thumb
        st.sidebar.feedback("stars", key="temp_fb_thumb", on_change=update_feedback, )
        
        def update_comment():
            st.session_state.fb_comment = st.session_state.temp_fb_comment
        st.sidebar.text_area('Your comment if needed:', value=st.session_state.fb_comment, key="temp_fb_comment", on_change=update_comment)
        
        btn_submit = st.sidebar.button('Submit!', type='secondary')

        # ECG data processing
        ecg = np.array(ecg_dict['signals_flat'])
        ecg = ecg.reshape(12, 500)
        
        label_list = ecg_dict['label_list']
        actual_labels = '\n- '.join(label_list)
        st.info(f"""
                **Actual** interpretation:\n- {actual_labels}
                """)

        # Plot settings
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_order = [0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11]  # I, aVR, II, aVL, III, aVF, V1, V4, V2, V5, V3, V6

        # Parameters
        n_samples = 500
        sampling_rate = 100
        duration = n_samples / sampling_rate
        time = np.linspace(0, duration, n_samples)

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

        # Handle submission
        if btn_submit:
            act_ecg_dict = db.collection('ecg_data').document(st.session_state.ecg_select)
            act_ecg_dict.update({
                'eval': True
            })
            eval_data = db.collection("eval_data").document(st.session_state.ecg_select)
            eval_data.set({
                "username": st.session_state.username,
                "fb_thumb": st.session_state.fb_thumb,
                "fb_comment": st.session_state.fb_comment,
                "submit_datetime": datetime.datetime.now()
            })
            st.sidebar.success('Your feedback was submitted!')
            
            if st.sidebar.button("Reset"):
                st.session_state.ecg_select = ecg_id_list[0]
                st.session_state.ecg_dict = None
                st.session_state.ecg_loaded = False
                st.session_state.fb_thumb = None
                st.session_state.fb_comment = ""
    else:
        st.info("👈 Select a patient in sidebar.")

    # Expander with additional info
    with st.expander("About the app"):
        st.write("""
            This is an interface for evaluating ECG data.\n
            Design and development by Dr. Alireza Soheilipour
        """)

if __name__ == "__main__":
    main()
    