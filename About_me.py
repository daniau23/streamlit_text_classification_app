import streamlit as st

st.set_page_config(
    page_title="About me",
    page_icon="👋",
)

sidebar = st.sidebar
st.write("# Welcome to Text Classification App by")
st.markdown('## Daniel Chiebuka Ihenacho👋!')

# st.sidebar.success("About ME!")
sidebar.success("About ME!")

st.markdown('### Summary')
st.info('''
- Aspiring Data scientist, Analyst and Teacher.
- TEFL (Teaching English as a Foreign Language) certified.
- Mandarin proficient; HSK 4 (Hanyu Shuiping Kaoshi) certified.
- Skilled, disciplined & a team player
- Motivated by problems/challenges and intrigued in finding solutions.
I am thrilled and look forward to hearing from you about potential vacancies and opportunities. Please do feel free to message me.
''')
