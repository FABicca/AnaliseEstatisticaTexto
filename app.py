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
from bs4 import BeautifulSoup
from seleniumbase import SB
import os
import shutil

path_projeto = os.getcwd()
path_exemplos = path_projeto + '/exemplo'
texto_a_ser_analisado = ''
df = None
idioma = 'pt-br'
gerar_bigrama = False

@st.cache_resource(show_spinner=False)
def get_chrome_driver_path() -> str:
    return shutil.which('chromedriver')

def get_chrome_path() -> str:
    return shutil.which('chromium-browser')

st.write(get_chrome_driver_path())
st.write(get_chrome_path())

@st.cache_resource(show_spinner=False)
def getSite(link):
    with SB(uc=True, incognito= True, headless=True,
            #binary_location= f'{path_projeto}/chrome-linux64/chrome'
            #binary_location=get_chrome_path()
            ) as sb:
        sb.driver.uc_open_with_tab(link)
        sb.sleep(3)
        site = BeautifulSoup(sb.driver.page_source, 'html.parser') 
    return site

def readFile(uploaded_file):
    texto_a_ser_analisado = ''
    df = None

    if uploaded_file:
        st.write('Arquivo: ' + uploaded_file.name)
        if uploaded_file.type == 'text/plain':
            gerar_bigrama = True

            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            texto_a_ser_analisado = stringio.read()

        elif uploaded_file.type == 'text/csv':
            gerar_bigrama = False

            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            string_data = stringio.read()

            if string_data.count(',') > string_data.count(';'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, sep=';')

            texto_a_ser_analisado = df.to_csv(sep='\t', index= False, header = False)

        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            gerar_bigrama = True
            
            texto_a_ser_analisado = docx2txt.process(uploaded_file)

        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            gerar_bigrama = False

            df = pd.read_excel(uploaded_file)
            
            texto_a_ser_analisado = df.to_csv(sep='\t', index=False, header = False)

        elif uploaded_file.type == 'application/pdf':
            gerar_bigrama = True

            pdf = pdfplumber.open(uploaded_file)

            for page in pdf.pages:
                content = page.extract_text()
                texto_a_ser_analisado += content + ' '

            texto_a_ser_analisado = str.strip(texto_a_ser_analisado)
    return texto_a_ser_analisado, df

st.title('Análise Estatística de um Texto\n')

escolha = st.sidebar.radio(
    "Escolha como será feito a análise:",
    ["Texto",
     "Link", 
     "Arquivo"])

if escolha == 'Texto':
    with st.sidebar.form("my-form", clear_on_submit=True):
        texto_a_ser_lido = st.text_area(label = 'Texto a ser analisado:', 
                                        height = 400)
        submitted = st.form_submit_button("Enviar")

        if submitted and texto_a_ser_analisado is not None:
            gerar_bigrama = True
            texto_a_ser_analisado = texto_a_ser_lido
elif escolha == "Link":
    with st.sidebar.form("my-form", clear_on_submit=True):
        link = st.text_input(label = 'Site a ser analisado:')
        submitted = st.form_submit_button("Enviar")

        if submitted and link is not None:
            gerar_bigrama = True

            if re.search('pt', link):
                idioma = 'pt-br'
            elif re.search('en', link):
                idioma = 'en-us'
            elif not re.search('.br', link):
                idioma = 'en-us'    

            site = getSite(link) 

            site_body = site.find('body')
            texto_a_ser_analisado = site_body.get_text()
else:
    with st.sidebar.form("my-form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Escolha o arquivo que se deseja analisar:", 
                                    type = ['txt', 'csv', 'docx', 'xlsx', 'pdf'])
        submitted = st.form_submit_button("Enviar")

        if submitted and uploaded_file is not None:
            texto_a_ser_analisado, df = readFile(uploaded_file)

if texto_a_ser_analisado == '':
    st.write('Digite um texto, procure por um link ou \
             faça o upload de um arquivo para ser analisado.')
    
    st.write('Exemplo:')

    st.image(f'{path_exemplos}/wikipedia_img_1.png')
    st.image(f'{path_exemplos}/wikipedia_img_2.png')
    st.image(f'{path_exemplos}/wikipedia_img_3.png')
    st.image(f'{path_exemplos}/wikipedia_img_4.png')
else:
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
    if idioma == 'pt-br':
        stopwords = nltk.corpus.stopwords.words('portuguese')
    else:
        stopwords = nltk.corpus.stopwords.words('english')    

    tokens_limpos=[]
    for item in tokens:
        if (item not in stopwords) & (len(item) > 2) :
            tokens_limpos.append(item)

    if len(tokens_limpos) > 0:
        palavras_frequentes_ordenadas = Counter(tokens_limpos).most_common()
        words_tokens = [palavra[0] for palavra in palavras_frequentes_ordenadas[:20]]
        freq_tokens = [palavra[1] for palavra in palavras_frequentes_ordenadas[:20]]

        fig=go.Figure(go.Bar(x=words_tokens,
                            y=freq_tokens, text=freq_tokens, textposition='outside'))
        fig.update_layout(
            autosize=False,
            width=1000,
            height=500,
            title_text='20 palavras mais utilizadas')
        fig.update_xaxes(tickangle = -45)

        st.plotly_chart(fig, use_container_width=True)

        #Nuvem de Palavras
        all_tokens = " ".join(s for s in tokens_limpos)
        wordcloud = wordcloud = WordCloud(width=1600, height=800, background_color="#f5f5f5").generate(all_tokens)

        # mostrar a imagem final
        fig2, ax = plt.subplots(figsize=(10,6))
        ax.imshow(wordcloud, interpolation='bilinear')
        st.pyplot(fig2)

        if gerar_bigrama:
            bigrams= [*map(' '.join, zip(tokens_limpos, islice(tokens_limpos, 1, None)))]
            if len(bigrams) > 0:
                bigramas_mais_frequentes = Counter(bigrams).most_common()
                words_bigrams = [bigram[0] for bigram in bigramas_mais_frequentes[:20]]
                freq_bigrams = [bigram[1] for bigram in bigramas_mais_frequentes[:20]]

                fig=go.Figure(go.Bar(x=words_bigrams,
                        y=freq_bigrams, text=freq_bigrams, textposition='outside'))
                fig.update_layout(
                    autosize=False,
                    width=1000,
                    height=500,
                    title_text='20 bigramas mais frequentes')
                fig.update_xaxes(tickangle = -45)

                st.plotly_chart(fig, use_container_width=True)

            trigrams = [*map(' '.join, zip(tokens_limpos, islice(tokens_limpos, 1, None), islice(tokens_limpos, 2, None)))]
            if len(trigrams) > 0:
                trigramas_mais_frequentes = Counter(trigrams).most_common()

                words_trigramas = [trigrama[0] for trigrama in trigramas_mais_frequentes[:20]]
                freq_trigramas = [trigrama[1] for trigrama in trigramas_mais_frequentes[:20]]

                fig=go.Figure(go.Bar(x=words_trigramas,
                                    y=freq_trigramas, text=freq_trigramas, textposition='outside'))
                fig.update_layout(
                    autosize=False,
                    width=1000,
                    height=600,
                    title_text='20 trigramas mais frequentes')
                fig.update_xaxes(tickangle = -45)

                st.plotly_chart(fig, use_container_width=True)
    else:
        st.write('''Não foram encontradas palavras válidas para serem analisadas!
                 (Números ou palavras com 2 ou menos letras não seram analisadas.''')

