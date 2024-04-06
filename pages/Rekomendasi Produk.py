import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
import pickle as pickle
import streamlit as st
import os
from datetime import datetime
import squarify as squarify
import base64
import csv
from sklearn.preprocessing import StandardScaler

import re
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from pathlib import Path
import streamlit_authenticator as sa
from streamlit_extras.switch_page_button import switch_page
from st_pages import Page, show_pages, add_page_title

no_sidebar_style = """
    <style>
        div[data-testid="stSidebarNav"] {display: none;}
    </style>
    """
st.markdown(no_sidebar_style, unsafe_allow_html=True) 
st.sidebar.title("Rekomendasi Produk")

pp=[ "Data Overview", "Pre Processing Data","Modelling FP-Growth"]
choice=st.sidebar.radio("Pilih Menu Dibawah",pp)



def load_data(uploaded_file):
    if uploaded_file is not None:
        st.sidebar.success("File uploaded successfully!")
        columns = ['id_transaksi','id_customer', 'tanggal_transaksi', 'jumlah_transaksi', 'total_transaksi', 'item_transaksi']
        df = pd.read_csv(uploaded_file, encoding='Latin-1', sep=',', header=0, engine='python', on_bad_lines='skip', quoting=csv.QUOTE_NONE, names=columns)
        st.session_state['df'] = df
        return df
    else:
        st.write("Please upload a data file to proceed.")
        return None

    # Berfungsi untuk menghasilkan tautan unduhan CSV
def csv_download_link(df, csv_file_name, download_link_csv):
    csv_data = df.to_csv(index=True)
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{csv_file_name}">{download_link_csv}</a>'
    st.markdown(href, unsafe_allow_html=True)    
    # Initializing session state variables
    if 'df' not in st.session_state:
        st.session_state['df'] = None

    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None

        
        
if choice=='Data Overview':
        st.title("Data Overview")
        st.header("", divider='rainbow')
        # Daftar semua file di direktori 'sample_data'
        sample_files = os.listdir('data')
        
        # Buat tombol radio untuk memungkinkan pengguna memilih antara menggunakan file sampel atau mengunggah file baru
        data_source = st.sidebar.radio('Data source', ['Use a sample file', 'Upload a new file'])
        
        if data_source == 'Use a sample file':
            # Memungkinkan pengguna memilih file dari daftar
            selected_file = st.sidebar.selectbox('Choose a sample file', sample_files)
            
            # Baca file yang dipilih (Anda memerlukan logika tambahan untuk membaca file di sini)
            file_path = os.path.join('data', selected_file)
            st.session_state['uploaded_file'] = open(file_path, 'r')
            load_data(st.session_state['uploaded_file'])

        else:
            # Memungkinkan pengguna mengunggah file baru
            st.session_state['uploaded_file'] = st.sidebar.file_uploader("Choose a file", type=['csv'])
            
            if st.session_state['uploaded_file'] is not None:
                load_data(st.session_state['uploaded_file'])

        # st.session_state['uploaded_file'] = st.sidebar.file_uploader("Choose a file", type=['txt'])
        # load_data(st.session_state['uploaded_file'])
        
        if st.session_state['df'] is not None:
            st.write("### Data Overview")
            st.write("Number of rows:", st.session_state['df'].shape[0])
            st.write("Number of columns:", st.session_state['df'].shape[1])
            st.write("Data Transaksi")
            st.write(st.session_state['df'])
        else:
            st.write("No data available. Please upload a file in the 'Data Understanding' section.")
if choice=='Pre Processing Data':
        st.title("Pre Processing Data")
        st.header("", divider='rainbow')
        st.write("### Data Cleaning")
        if st.session_state['df'] is not None:
            st.session_state['df']['id_customer'] = st.session_state['df']['id_customer'].str.strip().str.lower()
            # 1. Handling missing, null, and duplicate values
            st.write("Number of missing values:")
            st.write(st.session_state['df'].isnull().sum())
            st.write("Number of NA values:")
            st.write((st.session_state['df'] == 'NA').sum())
            st.write("Number of duplicate rows:", st.session_state['df'].duplicated().sum())
        
            # Providing options for handling missing and duplicate values
            if st.checkbox('Remove duplicate rows'):
                st.session_state['df'].drop_duplicates(inplace=True)
                st.write("Duplicate rows removed.")
            
            if st.checkbox('Remove rows with NA values'):
                st.session_state['df'].replace('NA', pd.NA, inplace=True)
                st.session_state['df'].dropna(inplace=True)
                st.write("Rows with NA values removed.")
            # 2. Display number of unique values for each column
            unique_customer_ids = st.session_state['df']['id_customer'].unique()
            st.write("Number of unique values for each column:")
            st.write(st.session_state['df'].nunique())

            # Additional Data Overview
            st.write("Transactions timeframe from {} to {}".format(st.session_state['df']['tanggal_transaksi'].min(), st.session_state['df']['tanggal_transaksi'].max()))
            st.write("{:,} transactions don't have a customer id".format(st.session_state['df'][st.session_state['df'].id_customer.isnull()].shape[0]))
            st.write("{:,} unique id_customer".format(len(st.session_state['df'].id_customer.unique())))
            
            
            st.write("### Data Transformation")
            #memilih kolom yang akan digunakan
            # st.write("#### Kolom Yang Digunakan")
            header_list = ["id_customer", "item_transaksi"]
            # st.dataframe(st.session_state['df'][header_list])
            #menghapus ; menjadi ,
            
            st.session_state['df']['item_transaksi'] = st.session_state['df']['item_transaksi'].str.replace(';','')
            #menghapus jumlah item barang transaksi
            # remove_count =  r'\(\d+\)'
            st.session_state['df'].reset_index(drop=True, inplace=True)
            st.session_state['df']['item_transaksi'] = st.session_state['df']['item_transaksi'].apply(lambda x: [re.sub(r'\(\d+\)', ',', item).strip().rstrip(',') for item in x.split(',') ])
            st.dataframe(st.session_state['df'][header_list])  
        
        else:
            st.write("No data available. Please upload a file in the 'Data Understanding' section.")

    # Assuming st.session_state['df'] is your DataFrame with columns 'id_customer' and 'item_transaksi'



if choice == "Modelling FP-Growth":
        st.title("Modelling FP-Growth")
        st.header("", divider='rainbow')
        if st.session_state['df'] is not None:
            data = st.session_state['df']
            data = data.explode('item_transaksi')

            # Assuming 'item_transaksi' is a Pandas Series
            data['item_transaksi'] = data['item_transaksi'].apply(lambda x: x.strip('()').split(', ') if isinstance(x, str) else x)
            item_list = data['item_transaksi'].tolist()
            
            # Display the modified DataFram
            # st.dataframe(item_list)
            
            te = TransactionEncoder()
            te_ary = te.fit(item_list).transform(item_list)
            
            df = pd.DataFrame(te_ary, columns=te.columns_)
            st.write("#### Item Per Transaksi Dalam Setiap Transaksi")
            st.dataframe(df)
            rank_item = df.sum().sort_values(ascending=False).reset_index()
            rank_item.columns = ['item', 'frequency']
            
            st.write("#### Daftar Produk Terlaris")
            st.dataframe(rank_item)
            fig_bar = px.bar(
                rank_item,
                x='item',
                y='frequency',
                title='Diagram Produk Terlaris',
                labels={'item': 'Produk', 'frequency': 'Frekuensi'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.write("#### Frequent Itemsets")
            frequent_item = fpgrowth(df, min_support=0.03, use_colnames=True)
            frequent_item['itemsets'] = frequent_item['itemsets'].apply(lambda x: list(x))
            st.dataframe(frequent_item)
            
            st.write("### Aturan Asosiasi")
            rules = association_rules(frequent_item, metric="confidence", min_threshold=0.05)
            rules = rules.drop_duplicates(subset=['antecedents', 'consequents']).sort_values(by='confidence', ascending=False).reset_index(drop=True)
            rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
            rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
            rules['rule_description'] = rules.apply(lambda x: f"Jika Membeli {x['antecedents']} maka membeli {x['consequents']}", axis=1)
            
            st.dataframe(rules[['rule_description', 'support', 'confidence', 'lift']])
            
            
        else:
            st.write("No data available. Please upload a file in the 'Data Understanding' section.")

if st.sidebar.button(":derelict_house_building: Back to Home"):
    st.switch_page("app.py")