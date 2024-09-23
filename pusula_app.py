import streamlit as st
import streamlit as st
import base64
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="Pusula Academy Case Study",
    page_icon="💊",
    initial_sidebar_state="expanded",
)

def get_raw_data():
    """
    This function returns a pandas DataFrame with the raw data.
    """
    raw_df = pd.read_csv('datasets/side_effect_data.csv')
    return raw_df

def get_cleaned_data():
    """
    This function return a pandas DataFrame with the cleaned data.
    """
    clean_data = pd.read_csv('datasets/clean.csv')
    return clean_data

df = get_cleaned_data()

def summary_table(df):

    summary = {
    "Değişken Sayısı": [len(df.columns)],
    "Gözlem Sayısı": [df.shape[0]],
    "Eksik Hücreler": [df.isnull().sum().sum()],
    #"Missing Cells (%)": [round(df.isnull().sum().sum() / df.shape[0] * 100, 2)],
    "Yinelenen Satırlar": [df.duplicated().sum()],
    "Yinelenen Satırlar (%)": [round(df.duplicated().sum() / df.shape[0] * 100, 2)],
    "Kategorik Değişkenler": [len([i for i in df.columns if df[i].dtype==object])],
    "Sayısal Değişkenler": [len([i for i in df.columns if df[i].dtype!=object])],
    }
    return pd.DataFrame(summary).T.rename(columns={0: 'Values'})


def grab_col_names(dataframe, cat_th=10, car_th=35):  # kategorik, nümerik değişkenleri ayıklamak için
  ###cat_cols
  cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
  num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                 dataframe[col].dtypes != "O"]
  cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                 dataframe[col].dtypes == "O"]
  cat_cols = cat_cols + num_but_cat
  cat_cols = [col for col in cat_cols if col not in cat_but_car]
  ###num_cols
  num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
  num_cols = [col for col in num_cols if col not in num_but_cat]
  print(f"observations: {dataframe.shape[0]}")
  print(f"variables: {dataframe.shape[1]}")
  print(f"cat_cols: {len(cat_cols)}")
  print(f"num_cols: {len(num_cols)}")
  print(f"cat_but_car: {len(cat_but_car)}", f"cat_but_car name: {cat_but_car}")
  print(f"num_but_cat: {len(num_but_cat)}", f"num_but_cat name: {num_but_cat}")
  return cat_cols, num_cols, cat_but_car

def num_summary(dataframe, col_name, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write(dataframe[col_name].describe(quantiles).T)
    with col2:
        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            dataframe[col_name].hist(bins=20, color="#559e83")
            plt.xlabel("Value", fontsize=14)
            plt.ylabel("Frequency", fontsize=14)
            plt.title(col_name, fontsize=14)
            st.pyplot(fig)


def plot_categorical_variables(dataframe, col_name, plot=False):
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write(dataframe[col_name].value_counts())
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=col_name, data=dataframe, palette='Set2', ax=ax)
        ax.set_xlabel(col_name, fontsize=14)
        ax.set_ylabel("Frekans", fontsize=14)
        ax.set_title(f'{col_name} Değişkeni Dağılımı', fontsize=14)
        plt.xticks(rotation=45)
        st.pyplot(fig)


################################EDA#####################################
st.title("Pusula Academy Case Study")
st.markdown("""
    Bu çalışma, side_effect_data veri seti üzerinde Exploratory Data Analysis (EDA) ve veri ön işleme işlemleri gerçekleştirilmiştir. 
    Veri seti; demografik bilgiler (cinsiyet, yaş, kilo, boy, vb.), yan etki başlangıç tarihleri ve kronik hastalık bilgilerini içermektedir.
     Bu proje kapsamında hem kategorik hem de sayısal veriler üzerinde işlemler uygulanmış ve gerekli grafiksel görselleştirmeler ile desteklenmiştir.
        """)
df_c = get_cleaned_data()
df_raw = get_raw_data()
st.header("Veri Seti İncelemesi")

dataset_choice = st.radio("İncelemek İçin Veri Seti Seçiniz", ("İşlenmemiş Veri Seti", "İşlenmiş Veri Seti"))

if dataset_choice == "İşlenmemiş Veri Seti":
    if st.button("Head"):
        st.write(df_raw.head())

    if st.button("Tail"):
        st.write(df_raw.tail())

    if st.button("Tüm veriyi göster"):
        st.dataframe(df_raw)

elif dataset_choice == "İşlenmiş Veri Seti":
    if st.button("Head"):
        st.write(df_c.head())
    if st.button("Tail"):
        st.write(df_c.tail())
    if st.button("Tüm veriyi göster"):
        st.dataframe(df_c)

st.header("Veri Seti Özeti")
if st.button("İşlenmemiş veri seti"):
    st.write(summary_table(df_raw))
    cat_cols, num_cols, cat_but_car = grab_col_names(df_raw)
    df_cat_cols = pd.DataFrame({"Kategorik Sütunlar": cat_cols})
    df_num_cols = pd.DataFrame({"Sayısal Sütunlar": num_cols})
    df_car_cols = pd.DataFrame({"Kardinal Sütunlar": cat_but_car})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(df_cat_cols)

    with col2:
        st.write(df_num_cols)

    with col3:
        st.write(df_car_cols)

if st.button("İşlenmiş veri seti"):
    st.write(summary_table(df_c))
    cat_cols, num_cols, cat_but_car = grab_col_names(df_c)
    df_cat_cols = pd.DataFrame({"Kategorik Sütunlar": cat_cols})
    df_num_cols = pd.DataFrame({"Sayısal Sütunlar": num_cols})
    df_car_cols = pd.DataFrame({"Kardinal Sütunlar": cat_but_car})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(df_cat_cols)

    with col2:
        st.write(df_num_cols)

    with col3:
        st.write(df_car_cols)

st.header("Scatter Plot")
cat_cols, num_cols, cat_but_car = grab_col_names(df_c)
x_axis = st.selectbox('X', num_cols)
y_axis = st.selectbox('Y', num_cols)
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=df_c[x_axis], y=df_c[y_axis], ax=ax, color="#559e83")
plt.title(f'Scatter Plot: {x_axis} vs. {y_axis}', fontsize=15)
plt.xlabel(x_axis, fontsize=14)
plt.ylabel(y_axis, fontsize=14)
plt.grid(True)
plt.tight_layout()
st.pyplot(fig)

st.header("Sayısal Sütun Analizi")
selected_col = st.selectbox('Bir sütun seçiniz', num_cols)
num_summary(df_c, selected_col, plot=True)

st.header("Kategorik Sütun Analizi")
selected_col = st.selectbox('Bir sütun seçiniz', cat_cols)
plot_categorical_variables(df_c, selected_col, plot=True)

st.header("Kan Grubu ve Kronik Hastalık Analizi")
def plot_kan_grubu_kronik(dataframe):
    unique_groups = dataframe['Kan Grubu'].unique()  # Her kan grubunu alır

    for group in unique_groups:
        if st.button(f"{group} Kan Grubu için Kronik Hastalık Dağılımı"):
            filtered_df = dataframe[dataframe['Kan Grubu'] == group]  # O kan grubuna ait verileri filtreler

            if not filtered_df.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(x='Kan Grubu', hue='Kronik Hastaliklarim', data=filtered_df, palette='Set3', ax=ax)
                ax.set_title(f'{group} Kan Grubuna Göre Kronik Hastalık Dağılımı')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.write(f"{group} kan grubu için veri bulunamadı.")

plot_kan_grubu_kronik(df)

st.header("Yaş Gruplarına Göre Yan Etki Analizi")


def plot_yas_gruplari_yan_etki(dataframe):
    bins = [0, 18, 30, 45, 60, 100]
    labels = ['0-18', '19-30', '31-45', '46-60', '60+']
    dataframe['yas_gruplari'] = pd.cut(dataframe['Yaş'], bins=bins, labels=labels)

    unique_yan_etki = dataframe['Yan_Etki'].unique()
    palette = sns.color_palette("husl", len(unique_yan_etki))
    yan_etki_palette = dict(zip(unique_yan_etki, palette))

    fig, ax = plt.subplots(figsize=(10, 6))  # Streamlit ile uyumlu hale getirildi
    sns.countplot(x='yas_gruplari', hue='Yan_Etki', data=dataframe, palette=yan_etki_palette, ax=ax)
    ax.set_title('Yaş Gruplarına Göre Yan Etki Dağılımı')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)

    # Legend grafiğin dışına taşınması
    ax.legend(title='Yan_Etki', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    st.pyplot(fig)
plot_yas_gruplari_yan_etki(df)

st.header("Cinsiyete Göre İlaç Kullanım Süresi Analizi")


def plot_cinsiyet_ilac_suresi_histogram(dataframe):
    fig, ax = plt.subplots(figsize=(10, 6))  # Streamlit ile uyumlu hale getirildi
    sns.histplot(data=dataframe, x='toplam_ilac_kullanilan_gun', hue='Cinsiyet', multiple='stack', palette='Set2',
                 ax=ax)
    ax.set_title('Cinsiyete Göre İlaç Kullanım Süresi Dağılımı')
    ax.set_xlabel("İlaç Kullanım Süresi (Gün)")
    ax.set_ylabel("Frekans")

    st.pyplot(fig)
plot_cinsiyet_ilac_suresi_histogram(df)

st.header("Kronik Hastalık ve Yan Etki Analizi")
def plot_kronik_yan_etki(dataframe):
    dataframe['Kronik_Hastalik_Varmi'] = dataframe['Kronik Hastaliklarim'].apply(
        lambda x: 'Evet' if x != 'Yok' else 'Hayır')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Kronik_Hastalik_Varmi', hue='Yan_Etki', data=dataframe, palette='Set3', ax=ax)
    ax.set_title('Kronik Hastalığı Olanlarda Yan Etki Dağılımı')
    plt.xticks(rotation=45)
    st.pyplot(fig)

plot_kronik_yan_etki(df)

def plot_kronik_yas_yan_etki(dataframe):
    dataframe['Kronik_Hastalik_Varmi'] = dataframe['Kronik Hastaliklarim'].apply(
        lambda x: 'Evet' if x != 'Yok' else 'Hayır')

    bins = [0, 18, 30, 45, 60, 100]  # Yaş aralıkları
    labels = ['0-18', '19-30', '31-45', '46-60', '60+']
    dataframe['yas_gruplari'] = pd.cut(dataframe['Yaş'], bins=bins, labels=labels)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x='Kronik_Hastalik_Varmi', hue='yas_gruplari', data=dataframe, palette='Set2', ax=ax)
    ax.set_title('Kronik Hastalık ve Yaş Gruplarına Göre Yan Etki Dağılımı')
    plt.xticks(rotation=45)
    st.pyplot(fig)

plot_kronik_yas_yan_etki(df)

st.header("İlaçların Yan Etkisi ve Cinsiyet Dağılımı")
def plot_ilac_yan_etki_cinsiyet(dataframe):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x='Yan_Etki', hue='Cinsiyet', data=dataframe, palette='Set2', ax=ax)
    ax.set_title('İlaçların Yan Etkisi ve Cinsiyet Dağılımı')
    plt.xticks(rotation=90)
    st.pyplot(fig)

plot_ilac_yan_etki_cinsiyet(df)


st.header("İllere Göre Yan Etki Dağılımı")


def plot_il_yan_etki(dataframe):
    yan_etki_list = dataframe['Yan_Etki'].unique()  # Tüm yan etkileri al
    palette = sns.color_palette("hsv", len(yan_etki_list))  # Her yan etki için farklı renk ayarla

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Il', hue='Yan_Etki', data=dataframe, palette=palette, hue_order=yan_etki_list, ax=ax)
    ax.set_title('İllere Göre Yan Etki Dağılımı')
    plt.xticks(rotation=90)

    ax.legend(title='Yan_Etki', bbox_to_anchor=(1.05, 1), loc='upper left')

    st.pyplot(fig)


plot_il_yan_etki(df)

numeric_df = df.select_dtypes(include=['float64', 'int64'])

st.header("Korelasyon Haritası")
def corr_map(df, width=14, height=6, annot_kws=15):
    mtx = np.triu(df.corr())  # Üst üçgeni maskeleme
    fig, ax = plt.subplots(figsize=(width, height))  # Streamlit için fig nesnesi oluşturma
    sns.heatmap(df.corr(),
                annot=True,
                fmt=".2f",
                ax=ax,
                vmin=-1,
                vmax=1,
                cmap="RdBu",
                mask=mtx,
                linewidth=0.4,
                linecolor="black",
                cbar=False,
                annot_kws={"size": annot_kws})
    plt.yticks(rotation=0, size=15)
    plt.xticks(rotation=75, size=15)
    plt.title('\nCorrelation Map\n', size=20)
    st.pyplot(fig)  # Streamlit ile fig nesnesini gösterme

if st.button("Korelasyon Haritasını Göster"):
    corr_map(numeric_df, width=20, height=10, annot_kws=8)


