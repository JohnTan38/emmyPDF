import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
import glob, os
import re
from llmsherpa.readers import LayoutPDFReader
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config('EM Chem Onshore', page_icon="🏛️", layout='wide')

def title_main(url):
     st.markdown(f'<h1 style="color:#230c6e;font-size:42px;border-radius:2%;"><br>{url}</h1>', unsafe_allow_html=True)

def success_df(html_str):
    html_str = f"""
        <p style='background-color:#baffc9;
        color: #313131;
        font-size: 15px;
        border-radius:5px;
        padding-left: 12px;
        padding-top: 10px;
        padding-bottom: 12px;
        line-height: 18px;
        border-color: #03396c;
        text-align: left;'>
        {html_str}</style>
        <br></p>"""
    st.markdown(html_str, unsafe_allow_html=True)

title_main('EM Chem Onshore Upcoming Orders')

def extract_info(doc):
    extracted_info = {}
    for row in doc['table_rows']:
        if 'block_idx' in row and 'cell_value' in row:
            extracted_info[row['block_idx']] = row['cell_value']
        elif 'block_idx' in row and 'sentences' in row:
            extracted_info[row['block_idx']] = row['cell_value']
        elif 'cells' in row:
            for cell in row['cells']:
                if 'cell_value' in cell and cell['cell_value']:
                    extracted_info[cell['cell_value']] = cell['cell_value']
    return extracted_info

def get_address_parts(address_string):
    # Split the address string by commas
    address_parts = address_string.split(',')

    # Extract the desired parts
    first_part = address_parts[0].strip()
    second_from_last_part = address_parts[-2].strip()
    third_from_last_part = address_parts[-3].strip()

    # Return the extracted parts as a list
    return [first_part, second_from_last_part, third_from_last_part]

#get_address_parts(customerName_address)

def get_order_details(order_details_str):
    # Split the input list by white space
    elements = order_details_str.split(' ')

    # Extract the desired elements
    second_element = elements[1]
    third_element = elements[2]
    fourth_element = elements[3]
    tenth_element = elements[9]
    fourteenth_element = elements[13]

    # Return the extracted elements as a list of strings
    return [second_element, third_element, fourth_element, tenth_element, fourteenth_element]

def prepend_url_to_files(file_list, url):
    return [url + file for file in file_list]

llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
pdf_reader = LayoutPDFReader(llmsherpa_api_url)

list_pdf_order_upload = st.file_uploader("Upload pdf files", type=['pdf'], accept_multiple_files=True)
list_pdf_order_name = [file.name for file in list_pdf_order_upload]

github_path = "https://raw.githubusercontent.com/JohnTan38/emmyPDF/main/"
pdf_order = prepend_url_to_files(list_pdf_order_name, github_path)
list_customer_order_details = []

if pdf_order is None:
    st.write('Please upload pdf file')
elif len(pdf_order) != 0:
    for uploaded_file in pdf_order:
        #st.write(uploaded_file)
        #st.write(f"File url: {uploaded_file._file_urls.upload_url}")
        llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
        pdf_reader = LayoutPDFReader(llmsherpa_api_url)
       
      
        try:
                doc = pdf_reader.read_pdf(uploaded_file)                
        except Exception as e:
                print(f"Error reading PDF: {e}")

        order_num = extract_info(doc.json[0]).get(1)              
        order_num_digit = re.findall("\d+", order_num)[0]
        customerName_address = get_address_parts(extract_info(doc.json[0]).get(4))
        order_details = get_order_details(extract_info(doc.json[0]).get(8))
        #order_details = (doc.json[1]).get('sentences') #list

        customerDetails_orderDetails = customerName_address + order_details
        customerDetails_orderDetails.insert(0, order_num_digit)
        list_customer_order_details.append(customerDetails_orderDetails)
    #st.write(list_customer_order_details)
df_orders = pd.DataFrame((list_customer_order_details), columns=['CustomerRef', 'OnshoreCustomer', 
                                                                 'State', 'DestinationPort', 
                                                                 'RequestedETA', 'ShippingNumber', 
                                                                 'MaterialCode', 'Weight', 'Plant'])

d_hsCode = {}
hsCode_xlsx = pd.read_excel(github_path+ "HS_Code.xlsx", engine='openpyxl')
xl_hsCode = pd.ExcelFile(github_path+ "HS_Code.xlsx", engine='openpyxl')

for sheet in xl_hsCode.sheet_names:
    d_hsCode[f'{sheet}']= pd.read_excel(xl_hsCode, sheet_name=sheet)

def merge_dataframes(df_1, df_2):
    df_1['MaterialCode'] = pd.to_numeric(df_1['MaterialCode'], errors='coerce')
    df_2['MaterialCode'] = pd.to_numeric(df_2['MaterialCode'], errors='coerce')
    merged_df = pd.merge(df_1, df_2, on='MaterialCode', how='left')
    return merged_df

createOrders = merge_dataframes(df_orders, d_hsCode['HSCode_BL'])
try:
    createOrders['TruckCapacity_(MT)'] = np.where(createOrders.Weight.astype('float') >20, '26', '20')
except Exception as e:
    createOrders['TruckCapacity_(MT)'] = createOrders['Weight']
    print(e)

def add_prefix(df, column_name='Order No', prefix='00'):
    df[column_name] = prefix + df[column_name].astype(str)
    return df

def replace_comma_in_columns(df, columns: list):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '')     
    return df

def format_date_in_column(df, columns: list):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('00:00:00', '')
    return df

freightOrders = st.file_uploader("Upload Upcoming Shipment Report", type=['xlsx'], accept_multiple_files=False)
if freightOrders is None:
    st.write('Please upload Upcoming Shipment Report')
else:
    freightOrders_xlsx = pd.read_excel(freightOrders, engine='openpyxl')

st.divider()
if st.button('Get Upcoming Orders'):
    freightOrders_1 = add_prefix(freightOrders_xlsx, 'Order No', '00')

    dict_TMFO = pd.Series(freightOrders_1['TM FO No'].values, index=freightOrders_1['Order No']).to_dict()
    createOrders['FreightOrder'] = createOrders['CustomerRef'].map(dict_TMFO)

    createOrders['RequestedETA'] = pd.to_datetime(createOrders['RequestedETA'])
    createOrders['PlannedLoad'] = createOrders['RequestedETA'] -pd.Timedelta(days=1)

    createOrders_1 = replace_comma_in_columns(createOrders, ['MaterialCode', 'FreightOrder'])
    createOrders_2 = format_date_in_column(createOrders_1, ['RequestedETA', 'PlannedLoad'])
    
    st.dataframe(createOrders_2)
