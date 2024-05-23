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
from sklearn.preprocessing import MinMaxScaler
import io
import re
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from pathlib import Path
import streamlit_authenticator as sa
from streamlit_extras.switch_page_button import switch_page
from st_pages import Page, show_pages, add_page_title
from sentry_sdk import capture_exception



no_sidebar_style = """
    <style>
        div[data-testid="stSidebarNav"] {display: none;}
    </style>
    """
st.markdown(no_sidebar_style, unsafe_allow_html=True)   



st.sidebar.title("Segmentasi Pelanggan")



sc=[ "Data Understanding","Data preparation","Modeling & Evaluation"]
choice= st.sidebar.radio("Pilih Menu Dibawah",sc)


def load_data(uploaded_file):
    try:
        # Expected columns template
        expected_columns = ['id_transaksi', 'id_customer', 'tanggal_transaksi', 'jumlah_transaksi', 'total_transaksi', 'item_transaksi']
        
        # Read the uploaded file
        df = pd.read_csv(uploaded_file, encoding='Latin-1', sep=',', header=0, engine='python', on_bad_lines='skip', quoting=csv.QUOTE_NONE)

        # Check if the columns in the uploaded file match the expected columns
        if list(df.columns) != expected_columns:
            raise ValueError("Kolom File Tidak Sesuai Dengan Template")
        
        # Store the DataFrame in the session state if columns are correct
        st.success("File berhasil dibaca.")
        st.session_state['df'] = df
        return df
    
    except ValueError as ve:
        st.error(str(ve))
        st.session_state['df'] = None
        return None
    
    except Exception as e:
        st.error("Terjadi kesalahan saat membaca file.")
        st.write(e)
        st.session_state['df'] = None
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


        empty = st.empty()
        # Assuming csv_path is the path to your CSV file
        csv_path = "data/template.csv"

        # Load CSV data into a DataFrame
        df = pd.read_csv(csv_path)

        # Encode the CSV data to base64
        csv_data = df.to_csv(index=False).encode('utf-8')
        b64_csv = base64.b64encode(csv_data).decode('utf-8')

        # Create a downloadable link for the CSV file
def data_understanding():
    try:    
        st.title("Data Understanding")
        st.header("", divider='rainbow')
        # Daftar semua file di direktori 'sample_data'
        sample_files = os.listdir('data')
        
        # Buat tombol radio untuk memungkinkan pengguna memilih antara menggunakan file sampel atau mengunggah file baru
        data_source = st.sidebar.radio('Data source', ['Use a sample file', 'Upload a new file'])
        
        if data_source == 'Use a sample file':
            st.sidebar.warning("Pilih File Sesuai Dengan Template")
            # Memungkinkan pengguna memilih file dari daftar
            selected_file = st.sidebar.selectbox('Choose a sample file', sample_files)
            
            # Baca file yang dipilih (Anda memerlukan logika tambahan untuk membaca file di sini)
            file_path = os.path.join('data', selected_file)
            st.session_state['uploaded_file'] = open(file_path, 'r')
            load_data(st.session_state['uploaded_file'])
        else:
            # Memungkinkan pengguna mengunggah file baru
            st.session_state.clear()
            st.sidebar.warning("Upload File Sesuai Dengan Template")
            st.session_state['uploaded_file'] = st.sidebar.file_uploader("Choose a file", type=['csv'])
            template_file = pd.read_csv('data/template.csv')
            def to_csv(df):
                output = io.StringIO()
                df.to_csv(output, index=False)
                processed_data = output.getvalue()
                return processed_data
            template_file_download = to_csv(template_file)
            

            st.sidebar.download_button(label="Download Template File",data=template_file_download, file_name="template.csv", mime="text/csv")
            
            if st.session_state['uploaded_file'] is not None:
                load_data(st.session_state['uploaded_file'])
                st.sidebar.success("File Berhasil Diupload")
            else:
                st.error("Silahkan Upload File Terlebih Dahulu")
        # st.session_state['uploaded_file'] = st.sidebar.file_uploader("Choose a file", type=['txt'])
        # load_data(st.session_state['uploaded_file']) 
        if st.session_state['df'] is not None:
            st.write("### Data Overview")
            st.write("Number of rows:", st.session_state['df'].shape[0])
            st.write("Number of columns:", st.session_state['df'].shape[1])
            st.write("Data Transaksi")
            st.write(st.session_state['df'])
        
            user_grouped = st.session_state['df'].groupby('id_customer').agg({'jumlah_transaksi': 'sum', 'total_transaksi': 'sum'})
            # st.write("### Total Data Transaksi Keseluruhan Berdasarkan Customer ID")
            # st.write(user_grouped)
            fig_bar = px.bar(
            user_grouped.reset_index(),  # Reset index to make 'id_customer' a regular column
            x='id_customer',
            y=['jumlah_transaksi', 'total_transaksi'],
            labels={'jumlah_transaksi': 'Total Transactions', 'total_transaksi': 'Total Sales'},
            title='Diagram Batang Total Transaksi',
            )

            # Display the bar chart
            # st.plotly_chart(fig_bar, use_container_width=True)
            # Create a new column for the month
            date_format = "%d/%m/%Y"
            st.session_state['df']['tanggal_transaksi'] = pd.to_datetime(st.session_state['df']['tanggal_transaksi'], format=date_format)
            st.session_state['df']['month'] = st.session_state['df']['tanggal_transaksi'].dt.month
            
            monthly_summary = st.session_state['df'].groupby(['month','id_customer']).agg({
                'jumlah_transaksi': 'sum',
                'total_transaksi': 'sum'
            }).reset_index()

            # Display the monthly summary
            # st.subheader('Data Total User Transaksi Per Bulan')
            # st.dataframe(monthly_summary)

            # Plot the total Sales per month
            dfm = st.session_state['df'].groupby('month')['total_transaksi'].sum()
            fig, ax = plt.subplots()
            ax.plot(dfm.index.astype(str), dfm, label='Total Penjualan', marker='o')
            st.session_state['data_understanding_done'] = True

            # Plot the total Quantity per month
            #  
            # st.pyplot(fig)
        else:
            st.session_state['df'] = None
    except Exception as e:
        capture_exception(e)
        # st.error("Silahkan Upload File Terlebih Dahulu")        
def data_preparation():
    try:
        st.title("Data Pre-Processing")
        st.header("", divider='rainbow')
        
        if st.session_state['df'] is not None:
            st.write("### Data Cleaning")
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
            #Change Tanggal Transaksi
            
            st.session_state['df']['tanggal_transaksi'] = pd.to_datetime(st.session_state['df']['tanggal_transaksi'])

            st.session_state['data_preparation_done'] = True
            # ... (rest of your code, don't forget to modify scatter plots too)
        else:
            st.error("No data available. Please upload a file in the 'Data Understanding' section.")
    except Exception as e:
        capture_exception(e)
        st.error("Silahkan Upload File Terlebih Dahulu")
def modelling():
    try:
        st.title("Modelling K-Means")
        st.header("", divider='rainbow')
        if st.session_state['df'] is not None and st.session_state['df'].shape[0] > 0 :
            # RFM Analysis
            recent_date = st.session_state['df']['tanggal_transaksi'].max()
            # Calculate Recency, Frequency, and Monetary value for each customer
            df_RFM = st.session_state['df'].groupby('id_customer').agg({
                'tanggal_transaksi': lambda x: (recent_date - x.max()).days, # Recency
                'id_customer': 'count', # Frequency
                'total_transaksi': lambda x: (x.sum()) # Monetary
            }).rename(columns={'tanggal_transaksi': 'Recency', 'id_customer': 'Frequency', 'total_transaksi': 'Monetary'})
            st.subheader('RFM Analysis DataFrame')
            st.dataframe(df_RFM)
            # Assuming df_RFM is your original DataFrame
            df_RFM_normalized = df_RFM.copy()  # Create a copy to avoid modifying the original DataFrame
            
            # Initialize the MinMaxScaler
            scaler = StandardScaler()

            # Normalize the 'Recency', 'Frequency', and 'Monetary' columns
            df_RFM_normalized[['Recency', 'Frequency', 'Monetary']] = scaler.fit_transform(df_RFM[['Recency', 'Frequency', 'Monetary']])

            # Display the normalized DataFrame
            st.subheader('Normalized RFM Analysis DataFrame')
            st.dataframe(df_RFM_normalized)
            
            
            st.title('Analisis KMeans menggunakan Metode Elbow')

            # Membangun dan menampilkan grafik Elbow Method
            sse = {}
            for k in range(1, 10):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(df_RFM_normalized)
                sse[k] = kmeans.inertia_

            fig, ax = plt.subplots()
            ax.set_title('Metode Elbow')
            ax.set_xlabel('Jumlah kluster (K)')
            ax.set_ylabel('Jumlah kuadrat jarak')
            sns.pointplot(x=list(sse.keys()), y=list(sse.values()), ax=ax)
            st.pyplot(fig)

            # Memungkinkan pengguna untuk memilih jumlah cluster k
            n_clusters = st.number_input('Pilih jumlah cluster k dari 1 hingga 20:', min_value=1, max_value=20, value=3, step=1, key="cluster_value")
            st.write(f'Anda telah memilih kluster {n_clusters} kluster.')

            # Terapkan model KMeans ke jumlah cluster yang dipilih
            model = KMeans(n_clusters=n_clusters, random_state= 42, n_init=10)
            model.fit( df_RFM_normalized)
            kmeans =  df_RFM_normalized.copy()
            kmeans['Segmen'] = (model.labels_ +1)
            st.subheader('Hasil Clustering')
            st.dataframe(kmeans)
            kmeans_data = kmeans.groupby('Segmen').agg({'Segmen':'count'})
            st.write('### Jumlah Pelanggan Setiap Segmen')
            kmeans_data.columns = ['Jumlah Pelanggan']
            st.dataframe(kmeans_data)
            def func(row):
                    if row['Segmen'] == 3:
                        return 'Pelanggan Tidak Aktif'
                    elif row['Segmen'] == 2:
                        return 'Pelanggan Aktif'
                    elif row['Segmen'] == 1:
                        return 'Pelanggan Baru'
                
            kmeans['Segmen'] = kmeans.apply(func, axis=1)
            
            def labelbaru(row, cluster_labels):
                for i, label in enumerate(cluster_labels):
                    if row['Segmen'] == i+1:
                        return label if label else f'Segmen {i+1}'
            st.warning("Jika ingin mengubah label, silahkan isi form dibawah ini")
            with st.form(key='label_form'):
                st.write('### Form Ubah Label Segmen')
                cluster_labels = []
                for i in range(n_clusters):
                    label = st.text_input(f'Label untuk Segmen {i+1}', key=f'label_{i}')
                    cluster_labels.append(label)
                
                submit = st.form_submit_button('Submit')
                
            if submit:
                kmeans =  df_RFM_normalized.copy()
                kmeans['Segmen'] = (model.labels_ +1)
                kmeans['Segmen'] = kmeans.apply(labelbaru, args=(cluster_labels,), axis=1)
                st.success('Label telah diubah. ')
            if st.button('Reset Label'):
                kmeans =  df_RFM_normalized.copy()
                kmeans['Segmen'] = (model.labels_ +1)
                kmeans['Segmen'] = kmeans.apply(func, axis=1)
                st.success('Label telah direset. ')
            # if resetlabel:
            #     # kmeans =  df_RFM_normalized.copy()
            #     # kmeans['Segmen'] = (model.labels_ +1)
            #     kmeans['Segmen'] = kmeans.apply(func, axis=1)
            #     st.success('Label telah direset. ')               

                
            st.subheader('Hasil Clustering Setelah Diberi Label')
            st.dataframe(kmeans, width=1500, height=500)

            
            kmeans['id_customer'] = df_RFM.index
            top_loyal_customers = kmeans.nlargest(5, 'Frequency')
            st.subheader('Top 5 Loyal Customers ')
            st.dataframe(top_loyal_customers[['Segmen', 'id_customer', 'Recency', 'Frequency', 'Monetary']])
            fig_bar = px.bar(
                top_loyal_customers,
                x='id_customer',
                y='Frequency',
                labels={'id_customer': 'Customer ID', 'Frequency': 'Frequency'},
                title='Diagram Top 5 Loyal Customers'
            )

            st.plotly_chart(fig_bar, use_container_width=True)
            top_recency_customers = kmeans.nlargest(5, 'Recency')
            st.subheader('Top 5 Customers with Highest Recency')
            st.dataframe(top_recency_customers[['Segmen', 'id_customer', 'Recency', 'Frequency', 'Monetary']])
            fig_bar = px.bar(
                top_recency_customers,
                x='id_customer',
                y='Recency',
                labels={'id_customer': 'Customer ID', 'Recency': 'Recency'},
                title='Diagram Top 5 Customers with Highest Recency'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            top_monetary_customers = kmeans.nlargest(5, 'Monetary')
            st.subheader('Top 5 Customers with Highest Monetary')
            st.dataframe(top_monetary_customers[['Segmen', 'id_customer', 'Recency', 'Frequency', 'Monetary']])
            fig_bar = px.bar(
                top_monetary_customers,
                x='id_customer',
                y='Monetary',
                labels={'id_customer': 'Customer ID', 'Monetary': 'Monetary'},
                title='Diagram Top 5 Customers with Highest Monetary'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Informasi Penjualan Barang
            item_transaksi = st.session_state['df'].copy()
            
            item_transaksi['item_transaksi'] = item_transaksi['item_transaksi'].str.replace(';','', regex=False)
            item_transaksi.reset_index(drop=True, inplace=True)
           
            item_transaksi['item_transaksi'] = item_transaksi['item_transaksi'].apply(lambda x: [re.sub(r'\(\d+\)', ',', item).strip().rstrip(',') for item in x.split(',') ])
    
        
            data_item_transaksi = item_transaksi.explode('item_transaksi')
    
            data_item_transaksi['item_transaksi'] = data_item_transaksi['item_transaksi'].apply(lambda x: x.strip('()').split(', ') if isinstance(x, str) else x)
            item_list = data_item_transaksi['item_transaksi'].tolist()
            # st.dataframe(item_list)
            te = TransactionEncoder()
            te_ary = te.fit(item_list).transform(item_list)
            te_df = pd.DataFrame(te_ary, columns=te.columns_)
            data_item_tanggal = pd.concat([item_transaksi['month'], te_df], axis=1)
            data_item_tanggal = data_item_tanggal.groupby('month').agg('sum')
            st.write('### Informasi Produk')
            produk = 'data/produk.csv'
            nama_produk = pd.read_csv(produk, sep=',')
            st.dataframe(nama_produk, width=1500, height=500)
            st.write("#### Data Item Transaksi Per Bulan")
            st.dataframe(data_item_tanggal)
            st.plotly_chart(px.bar(data_item_tanggal, barmode='group', title='Diagram Data Item Transaksi Per Bulan'), use_container_width=True)
            
            data_item_customer = pd.concat([item_transaksi['id_customer'], te_df], axis=1)
            
            data_item_customer = data_item_customer.groupby('id_customer').agg('sum')
            st.write("#### Data Item Transaksi Per Customer")
            # st.dataframe(data_item_customer)
            data_item_customer_segmen = pd.concat([kmeans['Segmen'], data_item_customer], axis=1)
            st.dataframe(data_item_customer_segmen)
            st.plotly_chart(px.bar(data_item_customer, barmode='group', title='Diagram Data Item Transaksi Per Customer', width=800, height=600),use_container_width=True)

            # data_item_transaksi['item_transaksi'] = data_item_transaksi['item_transaksi'].str.lower()
            # # data_item_transaksi = data_item_transaksi.groupby('item_transaksi').agg({'item_transaksi':'count'})
            # data_item_transaksi = data_item_transaksi.apply(lambda x: x.strip('()').split(', ') if isinstance(x, str) else x)
            # item_list = data_item_transaksi['item_transaksi'].tolist()
            # te = TransactionEncoder()
            # te_ary = te.fit(item_list).transform(item_list)
            # df = pd.DataFrame(te_ary, columns=te.columns_)
            # st.write("#### Item Per Transaksi Dalam Setiap Transaksi")
            # st.dataframe(df)
            # st.dataframe(data_item_transaksi)
         
        else:
            st.error("No data available. Please upload a file in the 'Data Understanding' section.")
    except Exception as e:
        capture_exception(e)
        st.error("Silahkan Upload File Terlebih Dahulu dan Lakukan Data Pre-Processing")
            
        
if choice == 'Data Understanding':    
    data_understanding()
elif choice == 'Data preparation':
    if 'data_understanding_done' in st.session_state and st.session_state['data_understanding_done']:
        data_preparation()
    else:
        st.warning("Selesaikan Data Understanding terlebih dahulu.") 
if choice == 'Modeling & Evaluation':
    if 'data_preparation_done' in st.session_state and st.session_state['data_preparation_done']:
        modelling()
    else:
        st.warning("Selesaikan Data Preparation terlebih dahulu.")
    
if st.sidebar.button(":derelict_house_building: Back to Home"):
    st.switch_page("app.py")