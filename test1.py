from streamlit_utils.firestore_utils import init_firestore

db = init_firestore()

docs = db.collection('ecg_data_1').stream()

ecg_id_list = [doc.id for doc in docs]
print(ecg_id_list)