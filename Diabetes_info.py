# diabetes_info.py
import streamlit as st


def show_diabetes_info():
    st.subheader("About Diabetes")
    st.markdown("""
    Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy. 
    There are two main types of diabetes:

    - **Type 1 Diabetes:** The body does not produce insulin.
    - **Type 2 Diabetes:** The body does not use insulin properly.

    Managing diabetes involves lifestyle changes, monitoring blood sugar levels, and sometimes medication.

    **Key Points:**
    - Regular monitoring is essential.
    - Diet, exercise, and medications can help manage the condition.
    - Early detection and treatment are critical to reduce the risk of complications.
    """)
