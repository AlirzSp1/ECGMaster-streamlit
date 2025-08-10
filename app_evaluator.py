import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from streamlit_utils.firestore_utils import init_firestore
import datetime

# Initialize Firestore
@st.cache_resource
def init_st():
    db = init_firestore()
    docs = db.collection('ecg_data_ste').stream()
    ecg_id_list = [""] + [doc.id for doc in docs]
    return db, ecg_id_list

def main():
    st.set_page_config(page_title="ECG Evaluator", page_icon="👨‍⚕️")
    st.title("👨‍⚕️ ECG MI Evaluator")
    st.text("استاد عزیز، لطفا بررسی کنید آیا نوارها همگی به نفع \nMI\n هستند یا خیر.")
    
    db, ecg_id_list = init_st()

    # Initialize session state
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "ecg_select" not in st.session_state:
        st.session_state.ecg_select = ""
    if "ecg_dict" not in st.session_state:
        st.session_state.ecg_dict = None
    if "ecg_loaded" not in st.session_state:
        st.session_state.ecg_loaded = False
    if "fb_stars" not in st.session_state:
        st.session_state.fb_stars = None
    if "fb_comment" not in st.session_state:
        st.session_state.fb_comment = ""
    if "fb_submit" not in st.session_state:
        st.session_state.fb_submit = False
    if "select_change" not in st.session_state:    
        st.session_state.select_change = ""
        
    @st.cache_data
    def load_ecg():
        ecg = np.array(st.session_state.ecg_dict['signals_flat'])
        ecg = ecg.reshape(12, 500)
        
        actual_labels = '\n- '.join(st.session_state.ecg_dict['label_list'])
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
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)
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

    # Sidebar widgets
    st.sidebar.header("Config")
    st.sidebar.text_input('Your Name:', value=st.session_state.username, key="username")
    st.sidebar.selectbox('Select Patient:', ecg_id_list, key="ecg_select")
    
    if (st.session_state.username != "") and (st.session_state.ecg_select != "") and (st.session_state.select_change != st.session_state.ecg_select):
        load_ecg.clear() # type: ignore
        act_ecg_dict = db.collection('ecg_data_ste').document(st.session_state.ecg_select)
        st.session_state.ecg_dict = act_ecg_dict.get().to_dict()
        st.session_state.ecg_loaded = True
        st.session_state.select_change = st.session_state.ecg_select
        st.sidebar.success("🎉 Loaded successfully!")
    else:
        st.sidebar.warning('Please fill both fields.')
    
            
    # Display ECG and feedback widgets if data is loaded
    if st.session_state.ecg_loaded and st.session_state.ecg_dict:
        if st.session_state.ecg_dict['eval']:
            st.sidebar.warning('This patient was evaluated before.')
            
        load_ecg()
            
        # Feedback widgets
        st.sidebar.text('Your feedback:')
        st.sidebar.feedback("stars", key="fb_stars")
        st.sidebar.text_area('Your comment if needed:', value=st.session_state.fb_comment, key="fb_comment")
    
        # Handle submission
        if st.sidebar.button('Submit!', type='secondary'):
            act_ecg_dict = db.collection('ecg_data_ste').document(st.session_state.ecg_select)
            act_ecg_dict.update({
                'eval': True
            })
            eval_data = db.collection("eval_data").document(st.session_state.ecg_select)
            eval_data.set({
                "username": st.session_state.username,
                "fb_stars": st.session_state.fb_stars,
                "fb_comment": st.session_state.fb_comment,
                "submit_datetime": datetime.datetime.now()
            })
            
            st.session_state.ecg_loaded = False
            st.session_state.fb_submit = True
            load_ecg.clear() # type: ignore
            st.rerun()
    else:
        st.info("👈 Select a patient in sidebar.")
        
    if st.session_state.fb_submit:
        st.sidebar.success('Your feedback was submitted!')
        st.session_state.fb_submit = False
        
    # Expander with additional info
    with st.expander("About the app"):
        st.write("""
            This is an interface for evaluating ECG data.\n
            Design and development by Dr. Alireza Soheilipour
        """)

if __name__ == "__main__":
    main()
    