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
import matplotlib as plt

no_sidebar_style = """
    <style>
        div[data-testid="stSidebarNav"] {display: none;}
    </style>
    """
st.markdown(no_sidebar_style, unsafe_allow_html=True)


### USER LOGIN
names = ["arya", "benti"]
usernames = ["arya", "benti"]
##load password
file_path=Path(__file__).parent/"users.pkl"
with file_path.open("rb") as file:
    hashed_password = pickle.load(file)

credentials = {"usernames":{}}
for un, name, pw in zip(usernames, names, hashed_password):
    user_dict = {"name":name,"password":pw}
    credentials["usernames"].update({un:user_dict})

authenticator = sa.Authenticate(credentials, "app_home", "auth", cookie_expiry_days=30)
name, authentication_status, username = authenticator.login("main")

if authentication_status == False:
    st.error("Username/Password salah")
if authentication_status == None:
    st.warning("Silahkan login terlebih dahulu")

if authentication_status == True:
    st.sidebar.image("img/logo.png", width=250)
    st.sidebar.write("<h5 style='text-align: center; color: black;'>Sistem Segmentasi dan Penentuan Rekomendasi Produk</h5>", unsafe_allow_html=True)
    st.sidebar.header("Hi, :blue[{}] :man-raising-hand:  ".format(name), )  
    if st.sidebar.button(":desktop_computer: Tentang Aplikasi", use_container_width=True, key="tentang"):
        st.title("Tentang Aplikasi")
        st.subheader("", divider='rainbow')
        st.write("Aplikasi ini merupakan aplikasi yang digunakan untuk melakukan segmentasi pelanggan dan rekomendasi produk.")
        st.title("Segmentasi Pelanggan")
        st.subheader("", divider='rainbow')
        st.write("Segmentasi Customer merupakan proses pengelompokan pelanggan berdasarkan karakteristik yang dimiliki oleh pelanggan tersebut. Segmentasi pelanggan ini dilakukan untuk mengetahui karakteristik pelanggan yang ada, sehingga dapat memberikan pelayanan yang lebih baik kepada pelanggan.")
        st.title("Rekomendasi Produk")
        st.subheader("", divider='rainbow')
        st.write("Rekomendasi produk merupakan proses memberikan rekomendasi produk kepada pelanggan berdasarkan karakteristik pelanggan tersebut. Rekomendasi produk ini dilakukan untuk memberikan rekomendasi produk yang sesuai dengan karakteristik pelanggan.")
        st.title("Download Template File")
        st.subheader("", divider='rainbow')
        st.download_button(label="Download Template File", data="data/template.csv", file_name="template.csv", mime="text/csv")
    if st.sidebar.button(":bar_chart: Dashboard", use_container_width=True, key="dashboard"):
        st.empty()
        st.header("Dashboard")
        st.subheader("", divider='rainbow')
        df = pd.read_csv("data/segmentasicustomerwsh.csv")
        user_grouped = df.groupby('id_customer').agg({'jumlah_transaksi': 'sum', 'total_transaksi': 'sum'})
        fig_bar = px.bar(user_grouped, x=user_grouped.index, y='total_transaksi', color='jumlah_transaksi', title='Total Transaksi per Customer')
        st.plotly_chart(fig_bar, use_container_width=True)
            # Create a new column for the month
        date_format = "%d/%m/%Y"
        df['tanggal_transaksi'] = pd.to_datetime(df['tanggal_transaksi'], format=date_format)
        df['month'] = df['tanggal_transaksi'].dt.month
            
        monthly_summary = df.groupby(['month','id_customer']).agg({
                'jumlah_transaksi': 'sum',
                'total_transaksi': 'sum'
            }).reset_index()

            # Display the monthly summary
        st.subheader('Data Total User Transaksi Per Bulan')
        st.dataframe(monthly_summary)

            # Plot the total Sales per month
        dfm = df.groupby('month')['total_transaksi'].sum()
        fig, ax = plt.subplots()
        ax.plot(dfm.index.astype(str), dfm, label='Total Penjualan', marker='o')


            # Plot the total Quantity per month
        st.write("### Total Quantity per Month")
        dfpc = df.groupby('month')['jumlah_transaksi'].sum()
        fig,axis = plt.subplots()
        axis.plot(dfpc.index.astype(str), dfpc, label='Total Barang Terjual', marker='o')
        ax.fill_between(dfpc.index.astype(str), dfpc, alpha=0.9)
        axis.set_xlabel('Bulan ke-')
        axis.set_ylabel('Total Barang Terjual')
        axis.set_title('Total Barang Terjual per Bulan')
        axis.legend()
        st.pyplot(fig)
        
        
    st.sidebar.subheader("Menu")
    
    if st.sidebar.button(":sparkle: Segmentasi Customer", use_container_width=True, key="segmentasi"):
        st.switch_page("pages/Segmentasi Pelanggan.py")
    if st.sidebar.button(":seedling: Rekomendasi Produk", use_container_width=True, key="rekomendasi"):
        st.switch_page("pages/Rekomendasi Produk.py")
        
    st.sidebar.subheader("Logout")
    authenticator.logout(" :walking: Logout", "sidebar")
    