import streamlit as st 
import pandas as pd
import joblib as jb
import sklearn

df = ''

st.set_page_config(
    page_title='Drug Expectation'
)

model = jb.load('model.h5')
sex_enc = jb.load('sex_enc.h5')
target_enc = jb.load('target_enc.h5')
bp_d = jb.load('bp_d.h5')
cho_d = jb.load('cho_d.h5')

st.write('<h1 style="text-align: center; color: GoldenRod;">Drug Prediction Form</h1>', unsafe_allow_html=True)
st.write('-'*100)

sex = st.selectbox('Select Patient Gender', ['Male', 'Female'])
BP = st.selectbox('Blood Pressure Status', ['HIGH', 'LOW', 'NORMAL'])

chol = st.radio('Select Cholesterol Status', ['HIGH', 'NORMAL'], horizontal=True)

Age = st.slider('Pick Patient Age', 15, 74, 25, step=1)
st.write('Age Picked : ', Age)

na_to_k = st.number_input('Insert Value',max_value=38.247, min_value=6.269)
st.write('Na : K is -->', na_to_k)

def predict():
    df = pd.DataFrame(columns=jb.load('columns_names.h5'))
    df.loc[0, 'Age'] = Age
    df .loc[0, 'Sex'] = sex[0]
    df.loc[0, 'BP'] = BP
    df.loc[0, 'Cholesterol'] = chol 
    df.loc[0, 'Na_to_K'] = na_to_k
    df['BP'] = df['BP'].map(bp_d)
    df['Cholesterol'] = df['Cholesterol'].map(cho_d)
    encoded_sex = pd.DataFrame(sex_enc.transform(df[['Sex']]), columns=sex_enc.get_feature_names_out())
    df = pd.concat([df.drop('Sex', axis=1), encoded_sex],axis=1)
    predtiction = model.predict(df)[0]
    return df, predtiction


col1, col2, col3 = st.columns([4, 3, 4])

if col2.button('Recommend Drug'):
    df, predtiction = predict()


if len(df) > 0:    
    
    drug = target_enc.inverse_transform([predtiction])[0]
    color = ''
    if drug == 'DrugY':
        color = 'Ivory'
    elif drug == 'drugC':
        color = 'LavenderBlush'
    elif drug == 'drugX':
        color = 'Lavender'
    elif drug == 'drugA':
        color = 'LightCyan'
    else :
        color = 'LightSeaGreen'
    st.write(f'<h3 style = "text-align:center; color: {color};">Patient needs :{drug.title()}</h6>', unsafe_allow_html=True)
    st.dataframe(df, width=750)
