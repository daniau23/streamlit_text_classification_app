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
# Custom function for printing text
def txt(a, b):
    col1, col2 = st.columns([4,1])
    with col1:
        st.markdown(a)
    with col2:
        st.markdown(b)

def txt2(a, b):
    col1, col2 = st.columns([1,3])
    with col1:
        st.markdown(a)
    with col2:
        st.markdown(b)

def txt3(a, b):
    col1, col2 = st.columns([1,2])
    with col1:
        st.markdown(a)
    with col2:
        st.markdown(b)

def txt4(a, b, c):
    col1, col2, col3 = st.columns([1.5,2,2])
    with col1:
        st.markdown(f'`{a}`')
    with col2:
        st.markdown(b)
    with col3:
        st.markdown(c)

tab1, tab2, tab3,tab4 = st.tabs(["Education", "Work Experience", "Skills","Social Media"])
with tab1:
    st.markdown('''
    ### Education
    ''')

    txt("**M.Eng.** (Software Engineering), *Liaoning Technical University*, China",
    "2019-2022")
    st.markdown('''
    - GPA: `3.98`
    - Research thesis entitled `Design of Gas Cyclone Using Hybrid Particle Swarm Optimization Algorithm`.[`Link`](https://www.mdpi.com/2076-3417/11/20/9772/htm)
    ''')

    txt("**B.Eng.** (Electrical & Electronics Engineering), *Landmark University*, Nigeria",
    "2014-2019")
    st.markdown('''
    - GPA: `4.77`
    - Thesis entitled `Design of Pyramidal Horn Antenna for WLAN Communication in Landmark Farm at 5.8GHz`
    - Graduated with First Class Honors.
    ''')
with tab2:
    st.markdown('''
    ### Work Experience
    ''')

    txt("**Industrial Training (IT) Student**,*Nigerian Communications Satellite Limited (Nigcomsat)*, Nigeria",
    "01-01-2018 To 30-06-2018")
    st.markdown('''
    - Contributed to the successful NNPC (Nigerian National Petroleum Corporation) elections by setting up computers and ensuring a safe zone.
    - Collaborated in a team/group work to design and implement a temperature cooling system for the department of Innovation & Development department (I&D)
    ''')

    txt("**Intern**, *Hotspot Network Limited*, Nigeria",
    "01-07-2017 to 31-07-2017")
    st.markdown("""
    - Contributed in Radio Planning and Deployment of Rural telephony BTS project under the auspices of Universal Service Provision Fund (USPF).
    - Praised for effective writing of executive summaries for the company as an intern in the company, which helped in business negotiations.
    """)
with tab3:
    st.markdown('''
    ### Skills
    ''')
    txt3("Programming", "`Python`")
    txt3("Data processing/wrangling", "`pandas`, `numpy`")
    txt3("Data visualization", "`matplotlib`, `seaborn`, `plotly`,`yellowbrick`")
    txt3("Data analysis", "`excel`,`powerbi`")
    txt3("Machine Learning", "`scikit-learn`,`gensim`,`spacy`")
    txt3("Deep Learning", '``')
    txt3("Model deployment", "`streamlit`")

#####################
with tab4:
    st.markdown('''
    ### Social Media
    ''')
    txt2("#### [`LinkedIn`](https://www.linkedin.com/in/daniel-ihenacho-637467223)", "#### [`Indeed`](https://my.indeed.com/p/danielchiebukai-hz1szfb)")
    txt2("#### [`Gmail`](mailto:danihenacho95@gmail.com)", "#### [`GitHub`](https://github.com/daniau23)")
    txt2("#### [`ORCID`](https://orcid.org/0000-0003-3043-9201)", "#### [`Kaggle`](https://www.kaggle.com/danielihenacho)")
    txt2("#### [`Twitter`](https://twitter.com/Danny_MLE)","#### [`Medium`](https://medium.com/@danihenacho95)")