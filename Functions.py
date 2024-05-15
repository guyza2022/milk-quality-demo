import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

data:pd.DataFrame = None

def update_data(new_data):
    global data
    data = new_data

@st.cache_data
def split_data_corr_y(data,_y_names):
    dataset_dict = {}
    for name in _y_names:
        temp_data = data
        y_data = data[name]
        temp_data = temp_data.drop(columns=_y_names)
        temp_data[name] = y_data
        dataset_dict[name] = temp_data
    return dataset_dict

@st.cache_data
def perform_savgol(x_names, pol, wl, dvt):
    global data
    print(1)
    for index in data.index:
        data.loc[index,x_names] = savgol_filter(data.loc[index,x_names], polyorder=pol, window_length=wl, deriv=dvt, mode='nearest')
    return data

@st.cache_data
def perform_savgol2(x_names, pol, wl, dvt):
    for index in data.index:
        data.loc[index,x_names] = savgol_filter(data.loc[index,x_names], polyorder=pol, window_length=wl, deriv=dvt, mode='nearest')
    return data

@st.cache_data
def plot_spectrum(data,_x_names):
    fig = go.Figure()
    for index in data.index:
        fig.add_trace(go.Scatter(x=_x_names, y=data.loc[index,_x_names],
                        mode='lines',
                        name=index))
    fig.update_layout(title='Spectrum Plot',
                    xaxis_title='Spectrum',
                    yaxis_title='Value',showlegend=False)
    return fig

@st.cache_data
def msc():
    """
        :msc: Scatter Correction technique performed with mean of the sample data as the reference.
        :param input_data: Array of spectral data
        :type input_data: DataFrame
        :returns: data_msc (ndarray): Scatter corrected spectra data
    """
    global data
    input_data = data
    eps = np.finfo(np.float32).eps
    input_data = np.array(input_data, dtype=np.float64)
    ref = []
    sampleCount = int(len(input_data))

    # mean centre correction
    for i in range(input_data.shape[0]):
        input_data[i,:] -= input_data[i,:].mean()
    
    # Get the reference spectrum. If not given, estimate it from the mean
    # Define a new array and populate it with the corrected data    
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        for j in range(0, sampleCount, 10):
            ref.append(np.mean(input_data[j:j+10], axis=0))
            # Run regression
            fit = np.polyfit(ref[i], input_data[i,:], 1, full=True)
            # Apply correction
            data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0]
    
    return (data_msc)

#@st.cache_data
def tt_split(data,_x_names,_to_predict_label):
    #data_to_train = data_dict.get(to_predict_label) #use specific feature to train
    y = data.loc[:,_to_predict_label].to_numpy()
    X = data.loc[:,_x_names].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return x_train,x_test,y_train,y_test

@st.cache_data
def normalize_y(y_train,y_test,_scaler):
    y_train = _scaler.fit_transform(y_train.reshape(-1,1))
    y_test = _scaler.transform(y_test.reshape(-1,1))
    return y_train,y_test,_scaler

@st.cache_data(experimental_allow_widgets=True)
def sav_tuning_1():
    with st.container():
        col1,col2= st.columns(2)
        with col1:
            first_input_ponm = st.selectbox(label='Polynomial Order',options=list(range(1,36)),key='p_first',index=0,on_change=clear_cache())
        with col2:
            first_input_dev = st.selectbox(label='Derivative',options=[x for x in range(1,first_input_ponm+1)],key='d_first',index=0,on_change=clear_cache())
        first_input_smp = st.slider(label='Smoothing Points',min_value=3,max_value=41,step=2,value=15,key='s_first',on_change=clear_cache)
    return first_input_dev,first_input_ponm,first_input_smp

@st.cache_data(experimental_allow_widgets=True)
def sav_tuning_2():
    with st.container():
        col1,col2 = st.columns(2)
        with col1:
            second_input_ponm = st.selectbox(label='Polynomial Order',options=list(range(1,36)),key='p_second',index=0,on_change=clear_cache())
        with col2:
            second_input_dev = st.selectbox(label='Derivative',options=[x for x in range(1,second_input_ponm+1)],key='d_second',index=0,on_change=clear_cache())
        second_input_smp = st.slider(label='Smoothing Points',min_value=3,max_value=41,step=2,value=21,key='s_second',on_change=clear_cache())
    return second_input_dev,second_input_ponm,second_input_smp

def clear_cache():
    st.cache_data.clear()