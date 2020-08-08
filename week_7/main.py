#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[2]:


# Algumas configurações para o matplotlib.
# %matplotlib inline

# from IPython.core.pylabtools import figsize


# figsize(12, 8)

# sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


print(countries.shape)
countries.info()


# In[6]:


# modify the dataset converting them to floats and removing the unnecessary spaces between the columns


cols = ['Pop_density', 'Coastline_ratio', 'Net_migration', 'Infant_mortality', 'Literacy', 'Phones_per_1000', 'Arable','Climate', 'Crops', 'Other', 'Birthrate', 'Deathrate', 'Agriculture', 'Industry', 'Service']
countries_modified = countries.copy()
for column in cols:
    countries_modified[column] = countries_modified[column].str.replace(",", ".").astype("float")
countries_modified.Region = countries_modified.Region.str.strip()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[43]:


def q1():
    return np.sort(countries_modified.Region.unique()).tolist() #return a list of unique regions of the dataset


# In[44]:


q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[9]:


pop_density = countries_modified.Pop_density 


# In[10]:


dens = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy="quantile") #discerize the column in quantiles 
quantiles = dens.fit_transform(np.array(pop_density).reshape(-1,1)) #transform the pop_density column in numbers representing quartiles


# In[11]:


def q2():
    return len(quantiles[quantiles == 9])


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[32]:


def q3():
    encoded = OneHotEncoder().fit_transform(countries_modified[["Region", "Climate"]].dropna()) #use Onehotencode to transform the categorical columns in logical
    return encoded.shape[1] + 1


# In[33]:


q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[14]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[15]:


model = Pipeline(steps=[("input", SimpleImputer(strategy="median")),
                        ("std_scaler", StandardScaler())])


# In[16]:


columns = countries.columns.to_list()


# In[17]:


n = model.fit_transform(countries_modified[columns[2:]])


# In[18]:


arable = pd.DataFrame(model.transform(np.array(test_country).reshape(1,20)[:,2:]),
             columns=columns[2:])["Arable"].round(3)


# In[19]:


def q4():
    return float(arable)


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[20]:


migration = countries_modified.Net_migration
# sns.boxplot(migration);


# In[21]:


quantile = migration.quantile([.25, .75])
q01 = quantile[quantile.index[0]]
q03 = quantile[quantile.index[1]]
IQR = abs(q01 - q03)
low_out = q01 - (1.5 * IQR)
high_out = q03 + (1.5 * IQR)


# In[22]:


def q5():
    return (len(migration[migration < low_out]), len(migration[migration > high_out]), False) #calculate the length of low_outliers and high_outliers


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[23]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[24]:


vectorizer = CountVectorizer()
count = vectorizer.fit_transform(newsgroup.data) #


# In[25]:


w_id = vectorizer.vocabulary_["phone"] #get the id of phone variable


# In[26]:


def q6():
    return int(count[:, w_id].toarray().sum()) #calculate the length of phone (number of occurrences)


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[27]:


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroup.data)


# In[28]:


def q7():
    return float(X[:, w_id].toarray().sum().round(3)) #calculate the TF-IDF

