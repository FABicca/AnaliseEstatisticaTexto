import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import pdfplumber
import docx2txt
from itertools import islice
import re
import nltk
from collections import Counter
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

def readFile(uploaded_file):
    texto_a_ser_analisado = ''
    df = None

    if uploaded_file:
        st.write('Arquivo: ' + uploaded_file.name)
        if uploaded_file.type == 'text/plain':
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            texto_a_ser_analisado = stringio.read()

        elif uploaded_file.type == 'text/csv':
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            string_data = stringio.read()

            if string_data.count(',') > string_data.count(';'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, sep=';')

            texto_a_ser_analisado = df.to_csv(sep='\t', index= False, header = False)

        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            texto_a_ser_analisado = docx2txt.process(uploaded_file)

        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            df = pd.read_excel(uploaded_file)
            
            texto_a_ser_analisado = df.to_csv(sep='\t', index=False, header = False)

        elif uploaded_file.type == 'application/pdf':
            pdf = pdfplumber.open(uploaded_file)

            for page in pdf.pages:
                content = page.extract_text()
                texto_a_ser_analisado += content + ' '

            texto_a_ser_analisado = str.strip(texto_a_ser_analisado)
    return texto_a_ser_analisado, df

st.title('Análise Estatística de um Texto\n')

texto_a_ser_analisado = ''
df = None

with st.sidebar.form("my-form", clear_on_submit=True):
    uploaded_file = st.file_uploader("Escolha o arquivo que se deseja analisar:", 
                                  type = ['txt', 'csv', 'docx', 'xlsx', 'pdf'])
    submitted = st.form_submit_button("Enviar")

    if submitted and uploaded_file is not None:
        texto_a_ser_analisado, df = readFile(uploaded_file)

if texto_a_ser_analisado != '':
    if isinstance(df, pd.DataFrame):
        st.write(df)
    
    # Transformação do conteúdo em lowercase
    texto_em_analise=texto_a_ser_analisado.lower()

    # Elminiação dos números utilizando
    texto_em_analise=re.sub(r'\d','', texto_em_analise)

    # expressão que indica a presença de 1 ou mais caracteres alfanuméricos consecutivos
    regex_token = r'\w+'  

    tokens = re.findall(regex_token, texto_em_analise)
    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('portuguese')

    tokens_limpos=[]
    for item in tokens:
        if (item not in stopwords) & (len(item) > 2) :
            tokens_limpos.append(item)

    palavras_frequentes_ordenadas = Counter(tokens_limpos).most_common()
    words_tokens = [palavra[0] for palavra in palavras_frequentes_ordenadas[:20]]
    freq_tokens = [palavra[1] for palavra in palavras_frequentes_ordenadas[:20]]

    fig=go.Figure(go.Bar(x=words_tokens,
                        y=freq_tokens, text=freq_tokens, textposition='outside'))
    fig.update_layout(
        autosize=False,
        width=1000,
        height=500,
        title_text='20 palavras mais utilizadas no relatório')
    fig.update_xaxes(tickangle = -45)

    st.plotly_chart(fig, use_container_width=True)

    #Nuvem de Palavras
    all_tokens = " ".join(s for s in tokens_limpos)
    wordcloud = wordcloud = WordCloud(width=1600, height=800, background_color="#f5f5f5").generate(all_tokens)

    # mostrar a imagem final
    fig2, ax = plt.subplots(figsize=(10,6))
    ax.imshow(wordcloud, interpolation='bilinear')
    st.pyplot(fig2)

