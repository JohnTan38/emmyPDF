import streamlit as st
import pandas as pd
#import polars as pl
import numpy as np
import openpyxl
import warnings
warnings.filterwarnings("ignore")

st.set_page_config('CompareSAP', page_icon="🏛️", layout='wide')
def title(url):
     st.markdown(f'<p style="color:#2f0d86;font-size:22px;border-radius:2%;"><br><br><br>{url}</p>', unsafe_allow_html=True)
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

#sidebar = st.sidebar
#with sidebar:
    #st.title("FIS2 Container Status")
    #title("DMS Inventory")
    #st.write('## Current Status')
    #status_inventory = st.radio(
        #label='Select one',
        #options=['AA', 'AV', 'AAP'],
        #index=0
    #)

def sort_dataframe_col(df, col_name):
    df = df.sort_values(by=col_name)
    return df

def compare_intersect(x, y):
    return bool((len(frozenset(x).intersection(y))==len(x))) #compare 2 lists

title_main('Compare differences in Upcoming Shipments')

# llama-parse is async-first, running the async code in a notebook requires the use of nest_asyncio
import nest_asyncio

nest_asyncio.apply()

import os

# API access to llama-cloud
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-QG645NFpO26Yggea6kJkFKHhBFc8TpcN4DvdosrrXV5tYptd"

# Using OpenAI API for embeddings/llms
os.environ["OPENAI_API_KEY"] = "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"

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings

embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-4-turbo")

Settings.llm = llm
Settings.embed_model = embed_model

#uploaded_file_1 = st.file_uploader("Upload SAP_1", type=['pdf'])
uploaded_file_1 = 'CompareSAP_1.pdf'
#uploaded_file_2 = st.file_uploader("Upload SAP_2", type=['pdf'])
uploaded_file_2 = 'CompareSAP_2.pdf'
filepath_sap = r'C:/Users/john.tan/OneDrive - Cogent Holdings Pte. Ltd/Documents/CrossBorder/CompareSAP/'
#"C:\Users\john.tan\OneDrive - Cogent Holdings Pte. Ltd\Documents\CrossBorder\CompareSAP\CompareSAP_1.pdf"

from llama_parse import LlamaParse
docs_2021 = LlamaParse(result_type="markdown").load_data(filepath_sap+ uploaded_file_1)#'/content/drive/MyDrive/CompareSAP_1.pdf'
docs_2020 = LlamaParse(result_type="markdown").load_data(filepath_sap+ uploaded_file_2)#'/content/drive/MyDrive/CompareSAP_2.pdf'

from llama_index.core.node_parser import MarkdownElementNodeParser

node_parser = MarkdownElementNodeParser(
    llm=OpenAI(model="gpt-4-turbo"), num_workers=8
)

import pickle
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)

reranker = FlagEmbeddingReranker(
    top_n=5,
    model="BAAI/bge-reranker-large",
)


def create_query_engine_over_doc(docs, nodes_save_path=None):
    """Big function to go from document path -> recursive retriever."""
    if nodes_save_path is not None and os.path.exists(nodes_save_path):
        raw_nodes = pickle.load(open(nodes_save_path, "rb"))
    else:
        raw_nodes = node_parser.get_nodes_from_documents(docs)
        if nodes_save_path is not None:
            pickle.dump(raw_nodes, open(nodes_save_path, "wb"))

    base_nodes, objects = node_parser.get_nodes_and_objects(raw_nodes)

    ### Construct Retrievers
    # construct top-level vector index + query engine
    vector_index = VectorStoreIndex(nodes=base_nodes + objects)
    query_engine = vector_index.as_query_engine(
        similarity_top_k=15, node_postprocessors=[reranker]
    )
    return query_engine, base_nodes

query_engine_2021, nodes_2021 = create_query_engine_over_doc(
    docs_2021, nodes_save_path="C:/Users/john.tan/Downloads/2021_nodes.pkl"#2021_nodes.pkl
)
query_engine_2020, nodes_2020 = create_query_engine_over_doc(
    docs_2020, nodes_save_path="C:/Users/john.tan/Downloads/2020_nodes.pkl"#2020_nodes.pkl
)

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine


# setup base query engine as tool
query_engine_tools = [
    QueryEngineTool(
        query_engine=query_engine_2021,
        metadata=ToolMetadata(
            name="sap_1",
            description=("Provides information about original orders data"),
        ),
    ),
    QueryEngineTool(
        query_engine=query_engine_2020,
        metadata=ToolMetadata(
            name="sap_2",
            description=("Provides information about amended orders data"),
        ),
    ),
]

sub_query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    llm=llm,
    use_async=True,
)

response_user= st.text_input('Input your query')
if response_user is not None:
    response = sub_query_engine.query(
        response_user
    )
else:
    st.write('Please input a query')

#st.write("Can you Select Order Numbers where Delivery Dates in sap_1 is different from Delivery Dates in sap_2")
if st.button('Get response'):
    st.write(str(response))
