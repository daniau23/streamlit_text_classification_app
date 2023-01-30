import streamlit as st
import pandas as pd
import numpy as np 
import warnings
import pickle
import re
import string
import spacy
import time
import tensorflow as tf
# import spacy_streamlit

st.set_page_config(page_title="Bi-GRU Classifier", page_icon="😶‍🌫️")

sidebar = st.sidebar

st.markdown("# Text Classification")
sidebar.info("Text Classification with Bi-GRU Classifier")
# st.write(
#     """### *Paragraph Vector (Doc2Vec) is supposed to be an extension to Word2Vec such that Word2Vec learns to project words into a latent d-dimensional space whereas Doc2Vec aims at learning how to project a document into a latent d-dimensional space.*"""
# )

# st.write(
#     "##### `Kindly read Practical Natural Language Processing A Comprehensive Guide to Building Real-World NLP Systems by(Sowmya Vajjala, Bodhisattwa Majumder, Anuj Gupta and Harshit Surana) pages 105-106`"
# )

# https://stackoverflow.com/questions/12851791/removing-numbers-from-string
# Preprocessing the text data
@st.cache(persist=True,show_spinner=False,allow_output_mutation=True)
def preprocess(sent):
    '''Cleans text data up, leaving only 2 or
        more char long non-stepwords composed of A-Z & a-z only
        in lowercase'''
    # lowercase
    sentence = sent.lower()

    # Remove RT
    sentence = re.sub('RT @\w+: '," ",sentence)

    # Remove special characters
    sentence = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", sentence)

    # Removing digits
    sentence = sentence.translate(str.maketrans('', '', string.digits))

    # Removing puntuactions
    # sentence = sentence.translate(str.maketrans('', '', string.punctuation))

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)  
    # When we remove apostrophe from the word "Mark's", 
    # the apostrophe is replaced by an empty space. 
    # Hence, we are left with single character "s" that we are removing here.

    # Remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)  
    # Next, we remove all the single characters and replace it by a space 
    # which creates multiple spaces in our text. 
    # Finally, we remove the multiple spaces from our text as well.

    return sentence


# Removing several stop words
all_stopwords = {"'d","'ll","'m","'re","'s","'ve",'a','about',
'above','across','after','afterwards','again','all','almost','alone','along',
'already','also','although','always','am','among','amongst','amount','an','and',
'another','any','anyhow','anyone','anything','anyway','anywhere','are','around',
'as','at','back','be','became','because','become','becomes','becoming','been','before',
'beforehand','behind','being','below','beside','besides','between','both','bottom',
'but','by','ca','call','can','could','did','do','does','doing','done','down','due','during','each',
'eight','either','eleven','else','elsewhere','empty','even','everyone','everything',
'everywhere','except','few','fifteen','fifty','first','five','for','former','formerly','forty','four','from','front',
'full','further','go','had','has','have','he','hence','her','here','hereafter','hereby','herein','hereupon','hers',
'herself','him','himself','his','how','however','hundred','i','if','in','indeed','into','is','it','its','itself','just','keep','last',
'latter','latterly','made','make','many','may','me','meanwhile','might','mine','more','moreover','move','much',
'must','my','myself','name','namely','neither','nevertheless','next','nine','nobody','noone','nothing','now','nowhere','of','often',
'on','once','one','only','onto','or','other','others','otherwise','our','ours','ourselves','out','own','part','per','perhaps','please','put',
'rather','re','regarding','same','say','see','several','she','should','show','side',
'since','six','sixty','so','some','somehow','someone','something','sometime','sometimes','somewhere','still','such','take','ten','than','that','the','their',
'them','themselves','then','thence','there','thereafter','thereby','therefore','therein','thereupon','these','they','third','this','those','though','three',
'through','throughout','thru','thus','to','together','top','toward','towards','twelve','twenty','two','under','unless','until','up','upon','us','used','using',
'various','via','was','we','well','were','what','whatever','when','whence','whenever','where','whereafter','whereas','whereby','wherein','whereupon',
'wherever','whether','which','while','whither','who','whoever','whole','whom','whose','why','will','with','within','would','yet','you','your','yours','yourself',
'yourselves','‘d','‘ll','‘m','‘re','‘s','‘ve','’d','’ll','’m','’re','’s','’ve'}

my_stop_words = set(all_stopwords) # My own stop words

SPACY_MODEL_NAMES = "en_core_web_sm"
# nlp = spacy.load("en_core_web_sm")

@st.cache(allow_output_mutation=True,persist=True,show_spinner=False)
def load_spacy_model(name):
    return spacy.load(name)

nlp = load_spacy_model(SPACY_MODEL_NAMES)

@st.cache(persist=True,show_spinner=False,suppress_st_warning=True,hash_funcs={"preshed.maps.PreshMap": hash,
"cymem.cymem.Pool": hash,"thinc.model.Model": hash,"spacy.pipeline.tok2vec.Tok2VecListener": hash})
def spacy_tokeniser(sent):
    sent = sent.strip().lower()
    doc = nlp(sent)
    mytokens = [token.lemma_ for token in doc if token.text not in my_stop_words]
    return mytokens

# This might be rewritten again using decorators to check if the model has been ran once then it will use the streamlit
# decorator to recursively call it using st.cache
# @st.cache(persist=True,show_spinner=False,allow_output_mutation=True)
# def loaders():
#     model = tf.keras.models.load_model("models/deeplearning/BiGRU/saved_model/saved_bi_gru_model")
#     return model
# model = loaders()
    
loaded_tokenizer = pickle.load(open("models/deeplearning/BiGRU/word_tokenizer/saved_tokeniser.pkl",'rb'))
model = tf.keras.models.load_model("models/deeplearning/BiGRU/saved_model/saved_bi_gru_model")

uncleaned_text =  '''
    I love Streamlit
    '''
txt = st.text_input('Text to analyze',uncleaned_text,placeholder="Awaiting your text")

# Calling all predefined functions 
preprocessed_text  = preprocess(txt)
lemmatised_text = spacy_tokeniser(preprocessed_text)
X = [lemmatised_text]

# Reshaping the vector for prediction
X = loaded_tokenizer.texts_to_sequences(X)
max_len = 300
X = tf.keras.preprocessing.sequence.pad_sequences(X,padding="post",maxlen=max_len)

if txt:
    y_pred = model.predict(X)
    prediction = np.argmax(y_pred, axis = 1)
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1)
        if percent_complete == 50:
            st.write("**Almost there**")
        my_bar.progress(percent_complete + 1)

    time.sleep(0.01)
    st.markdown("**Done predicting**")
    # st.snow()
    st.balloons()
    time.sleep(0.01)
    if prediction == 2:
        st.success(f"Positive sentiment")
    elif prediction == 1:
        st.warning(f"Neutral sentiment")
    elif prediction == 0:
        st.error(f"Negative sentiment")