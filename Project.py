import streamlit as st
import graphviz as graph
import matplotlib.pyplot as plt
import altair as alt
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hiplot as hip
from mpl_toolkits.mplot3d import Axes3D
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern, ExpSineSquared,DotProduct
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from yellowbrick.features import ParallelCoordinates
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.multioutput import MultiOutputRegressor
##################################################################################################################################################################

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")

##################################################################################################################################################################

st.markdown(""" <style> .font_title {
font-size:50px ; font-family: 'times'; color: black;text-align: center;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_header {
font-size:50px ; font-family: 'times'; color: black;text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subheader {
font-size:35px ; font-family: 'times'; color: black;text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subsubheader {
font-size:28px ; font-family: 'times'; color: black;text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_text {
font-size:22px ; font-family: 'times'; color: black;text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subtext {
font-size:18px ; font-family: 'times'; color: black;text-align: center;} 
</style> """, unsafe_allow_html=True)

####################################################################################################################################################################

st.markdown('<p class="font_title">Indoor Plant Growth</p>', unsafe_allow_html=True)

####################################################################################################################################################################

st.image("https://www.springwise.com/wp-content/uploads/2022/03/innovationsustainabilitycaptured-CO2-used-in-greenhouses.png")
cols=st.columns(6,gap='medium')
with cols[0].expander("Calming Video:"):
    st.video("https://www.youtube.com/watch?v=SA8lu-m2IZY",start_time=0)
with cols[1].expander("Calming Video:"):
    st.video("https://www.youtube.com/watch?v=lFcSrYw-ARY",start_time=0)
with cols[2].expander("Calming Video:"):
    st.video("https://www.youtube.com/watch?v=_kT38XB1YHo",start_time=0)
with cols[3].expander("Calming Video:"):
    st.video("https://www.youtube.com/watch?v=_jvyrwwOFys",start_time=0)
with cols[4].expander("Calming Video:"):
    st.video("https://www.youtube.com/watch?v=vybkZeU22bQ",start_time=0)
with cols[5].expander("Calming Video:"):
    st.video("https://www.youtube.com/watch?v=1ZYbU82GVz4",start_time=0)

####################################################################################################################################################################

st.markdown('<p class="font_text">Numerous factors regulate plant growth and cultivation yield including temperature, carbon dioxide concentration, relative humidity, and the intensity and quality of light. </p>', unsafe_allow_html=True)
st.markdown('<p class="font_text">The goal for this project is to investigate the available data for indoor lettuce cultivar for a possible trend between the investigated features and the cultivation yield, and trained a regression model to investigate its performance. </p>', unsafe_allow_html=True)

####################################################################################################################################################################
tab1, tab2 , tab3 , tab4 ,tab5 , tab6 , tab7 , tab8 , tab9 , tab10 , tab11 , tab12 , tab13 , tab14 , tab15 , tab16 = st.tabs(["Light-treatment Data",
 "Plant Growth Data", "Plant Growth Distribution", "Interactive Visualization","Linear Regression",
 "Neural Network Visualization","Neural Network Regression","","","","","","","","",""])

####################################################################################################################################################################
# Light-Treatment Data
with tab1: 
    st.markdown('<p class="font_header">Light-treatment Data: </p>', unsafe_allow_html=True)

    ####################################################################################################################################################################
    
    st.markdown('<p class="font_text">One of the cofactors that its impact has been investigated comprehensively, is the quality and the intensity with respect to its energy and photon density. While it is impossible to investigate the impact of every wavelengths on plant growth, studies suggest that impact of incoming light on plant growth could be measured by dividing the spectra into several wavebands. A common classification of light spectra is based on creating 100-nm wavebands and measure suggested parameters for specific wavebands. The following classification is used for the division spectral wavelength : Blue (B) 400 to 500 nm, Green (B) 500 to 600 nm, Red (R) 600 to 700 nm, Far-Red (FR) 700 to 800 nm. </p>',unsafe_allow_html=True)
    st.markdown('<p class="font_text">Energy and number of photons associated with the incoming spectra is often investigated in plant growth studies. Therefore, we analyzed the photon flux density and energy for each of the proposed wavebands. The following figure shows the photon flux and energy distribution for each of the investigated light treatment conditions. While intensity for photon flux density or energy refers to its value, the quality referes to the fraction ratio of the photon flux density or energy for those wavebands. </p>', unsafe_allow_html=True)

    Light_Treatments = pd.read_csv("Light Data All.csv")
    col1,col2=st.columns(2,gap='small')
    light_stat = col1.checkbox('Show statistical properties of Light Treatments')
    if light_stat==True:
        st.table(Light_Treatments.describe())
        st.markdown('<p class="font_subtext">Table 1: Statistical properties of various light treatment.</p>', unsafe_allow_html=True)

    light_show = col2.checkbox('Show Light Treatments Data')
    if light_show==True:
        st.table(Light_Treatments)
        st.markdown('<p class="font_subtext">Table 2: Light treatment information including energy, photon density, and wavelength.</p>', unsafe_allow_html=True)

    ####################################################################################################################################################################

    st.sidebar.markdown('<p class="font_text">Fig. 1: Spectral Visualization:</p>', unsafe_allow_html=True)

    Light_Treatment_Name = st.sidebar.selectbox(
    "Fig. 1: Light Treatment:",
    ['B30R150' , 'B30R150FR30' , 'R180FR30' , 'B90R90' , 'B90R90FR30' , 'B180FR30' , 'B90R90FR75' , 'B180R180' , 'B180R180FR30' , 'B180R180FR75' , 'B60R120' , 'B40G20R120' , 'B20G40R120' , 'G60R120' , 'B40R120FR20' , 'B20R120FR40' , 'R120FR60' , 'B20G20R120FR20' , 'R180' , 'B20R160' , 'B20G60R100' , 'B60G60R60' , 'B100R80' , 'B100G60R20' , 'EQW180' , 'EQW100B10R70' , 'EQW100B50R30' , 'WW180'])
    All_ratio = pd.read_csv("Overall Ratio.csv")
    col1,col3,col2=st.columns([5,1,5],gap='small')
    x_min = col1.number_input('Insert a minimum value for x-axis',value=0.00)
    x_max = col2.number_input('Insert a maximum value for x-axis',value=1000.00)

    if Light_Treatment_Name != "Greenhouse":
        Figure=plt.figure(figsize=(12,2))
        plt.subplot(1,2,1)
        plt.plot(Light_Treatments['Wavelength'] , Light_Treatments[Light_Treatment_Name], linewidth=1,color='black')
        plt.title(Light_Treatment_Name)
        plt.fill_between(Light_Treatments['Wavelength'].loc[(Light_Treatments['Wavelength']>=400) & (Light_Treatments['Wavelength']<=500) ], Light_Treatments[Light_Treatment_Name].loc[(Light_Treatments['Wavelength']>=400) & (Light_Treatments['Wavelength']<=500) ], color='blue')
        plt.fill_between(Light_Treatments['Wavelength'].loc[(Light_Treatments['Wavelength']>=500) & (Light_Treatments['Wavelength']<=600) ], Light_Treatments[Light_Treatment_Name].loc[(Light_Treatments['Wavelength']>=500) & (Light_Treatments['Wavelength']<=600) ], color='green')
        plt.fill_between(Light_Treatments['Wavelength'].loc[(Light_Treatments['Wavelength']>=600) & (Light_Treatments['Wavelength']<=700) ], Light_Treatments[Light_Treatment_Name].loc[(Light_Treatments['Wavelength']>=600) & (Light_Treatments['Wavelength']<=700) ], color='red')
        plt.fill_between(Light_Treatments['Wavelength'].loc[(Light_Treatments['Wavelength']>=700) & (Light_Treatments['Wavelength']<=800) ], Light_Treatments[Light_Treatment_Name].loc[(Light_Treatments['Wavelength']>=700) & (Light_Treatments['Wavelength']<=800) ], color='darkred')
        plt.axvline(x = 400, linewidth=1,linestyle='--',color='black')
        plt.axvline(x = 500, linewidth=1,linestyle='--',color='black')
        plt.axvline(x = 600, linewidth=1,linestyle='--',color='black')
        plt.axvline(x = 700, linewidth=1,linestyle='--',color='black')
        plt.axvline(x = 800, linewidth=1,linestyle='--',color='black')
        plt.xticks(ticks=[400,500,600,700,800])
        plt.xlim(x_min,x_max)
        plt.grid(which='both',axis='both',color='grey', linestyle='--', linewidth=.3)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Photon Flux Density')
        plt.subplot(1,2,2)
        plt.plot(Light_Treatments['Wavelength'] , Light_Treatments['Energy '+Light_Treatment_Name], linewidth=1,color='black')
        plt.title(Light_Treatment_Name)
        plt.fill_between(Light_Treatments['Wavelength'].loc[(Light_Treatments['Wavelength']>=400) & (Light_Treatments['Wavelength']<=500) ], Light_Treatments['Energy '+Light_Treatment_Name].loc[(Light_Treatments['Wavelength']>=400) & (Light_Treatments['Wavelength']<=500) ], color='blue')
        plt.fill_between(Light_Treatments['Wavelength'].loc[(Light_Treatments['Wavelength']>=500) & (Light_Treatments['Wavelength']<=600) ], Light_Treatments['Energy '+Light_Treatment_Name].loc[(Light_Treatments['Wavelength']>=500) & (Light_Treatments['Wavelength']<=600) ], color='green')
        plt.fill_between(Light_Treatments['Wavelength'].loc[(Light_Treatments['Wavelength']>=600) & (Light_Treatments['Wavelength']<=700) ], Light_Treatments['Energy '+Light_Treatment_Name].loc[(Light_Treatments['Wavelength']>=600) & (Light_Treatments['Wavelength']<=700) ], color='red')
        plt.fill_between(Light_Treatments['Wavelength'].loc[(Light_Treatments['Wavelength']>=700) & (Light_Treatments['Wavelength']<=800) ], Light_Treatments['Energy '+Light_Treatment_Name].loc[(Light_Treatments['Wavelength']>=700) & (Light_Treatments['Wavelength']<=800) ], color='darkred')
        plt.axvline(x = 400, linewidth=1,linestyle='--',color='black')
        plt.axvline(x = 500, linewidth=1,linestyle='--',color='black')
        plt.axvline(x = 600, linewidth=1,linestyle='--',color='black')
        plt.axvline(x = 700, linewidth=1,linestyle='--',color='black')
        plt.axvline(x = 800, linewidth=1,linestyle='--',color='black')
        # plt.xticks(ticks=[400,500,600,700,800])
        plt.xlim(x_min,x_max)
        plt.grid(which='both',axis='both',color='grey', linestyle='--', linewidth=.3)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Spectral Energy')
    else:
        Figure=plt.figure(figsize=(12,2))
        plt.subplot(1,2,1)
        plt.plot(Light_Treatments['Greenhouse Wavelength'] , Light_Treatments[Light_Treatment_Name], linewidth=1,color='black')
        plt.title(Light_Treatment_Name)
        plt.fill_between(Light_Treatments['Greenhouse Wavelength'].loc[(Light_Treatments['Greenhouse Wavelength']>=400) & (Light_Treatments['Greenhouse Wavelength']<=500) ], Light_Treatments[Light_Treatment_Name].loc[(Light_Treatments['Greenhouse Wavelength']>=400) & (Light_Treatments['Greenhouse Wavelength']<=500) ], color='blue')
        plt.fill_between(Light_Treatments['Greenhouse Wavelength'].loc[(Light_Treatments['Greenhouse Wavelength']>=500) & (Light_Treatments['Greenhouse Wavelength']<=600) ], Light_Treatments[Light_Treatment_Name].loc[(Light_Treatments['Greenhouse Wavelength']>=500) & (Light_Treatments['Greenhouse Wavelength']<=600) ], color='green')
        plt.fill_between(Light_Treatments['Greenhouse Wavelength'].loc[(Light_Treatments['Greenhouse Wavelength']>=600) & (Light_Treatments['Greenhouse Wavelength']<=700) ], Light_Treatments[Light_Treatment_Name].loc[(Light_Treatments['Greenhouse Wavelength']>=600) & (Light_Treatments['Greenhouse Wavelength']<=700) ], color='red')
        plt.fill_between(Light_Treatments['Greenhouse Wavelength'].loc[(Light_Treatments['Greenhouse Wavelength']>=700) & (Light_Treatments['Greenhouse Wavelength']<=800) ], Light_Treatments[Light_Treatment_Name].loc[(Light_Treatments['Greenhouse Wavelength']>=700) & (Light_Treatments['Greenhouse Wavelength']<=800) ], color='darkred')
        plt.axvline(x = 400, linewidth=1,linestyle='--',color='black')
        plt.axvline(x = 500, linewidth=1,linestyle='--',color='black')
        plt.axvline(x = 600, linewidth=1,linestyle='--',color='black')
        plt.axvline(x = 700, linewidth=1,linestyle='--',color='black')
        plt.axvline(x = 800, linewidth=1,linestyle='--',color='black')
        # plt.xticks(ticks=[400,500,600,700,800])
        plt.xlim(x_min,x_max)
        plt.grid(which='both',axis='both',color='grey', linestyle='--', linewidth=.3)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Photon Flux Density')
        plt.subplot(1,2,2)
        plt.plot(Light_Treatments['Greenhouse Wavelength'] , Light_Treatments['Energy '+Light_Treatment_Name], linewidth=1,color='black')
        plt.title(Light_Treatment_Name)
        plt.fill_between(Light_Treatments['Greenhouse Wavelength'].loc[(Light_Treatments['Greenhouse Wavelength']>=400) & (Light_Treatments['Greenhouse Wavelength']<=500) ], Light_Treatments['Energy '+Light_Treatment_Name].loc[(Light_Treatments['Greenhouse Wavelength']>=400) & (Light_Treatments['Greenhouse Wavelength']<=500) ], color='blue')
        plt.fill_between(Light_Treatments['Greenhouse Wavelength'].loc[(Light_Treatments['Greenhouse Wavelength']>=500) & (Light_Treatments['Greenhouse Wavelength']<=600) ], Light_Treatments['Energy '+Light_Treatment_Name].loc[(Light_Treatments['Greenhouse Wavelength']>=500) & (Light_Treatments['Greenhouse Wavelength']<=600) ], color='green')
        plt.fill_between(Light_Treatments['Greenhouse Wavelength'].loc[(Light_Treatments['Greenhouse Wavelength']>=600) & (Light_Treatments['Greenhouse Wavelength']<=700) ], Light_Treatments['Energy '+Light_Treatment_Name].loc[(Light_Treatments['Greenhouse Wavelength']>=600) & (Light_Treatments['Greenhouse Wavelength']<=700) ], color='red')
        plt.fill_between(Light_Treatments['Greenhouse Wavelength'].loc[(Light_Treatments['Greenhouse Wavelength']>=700) & (Light_Treatments['Greenhouse Wavelength']<=800) ], Light_Treatments['Energy '+Light_Treatment_Name].loc[(Light_Treatments['Greenhouse Wavelength']>=700) & (Light_Treatments['Greenhouse Wavelength']<=800) ], color='darkred')
        plt.axvline(x = 400, linewidth=1,linestyle='--',color='black')
        plt.axvline(x = 500, linewidth=1,linestyle='--',color='black')
        plt.axvline(x = 600, linewidth=1,linestyle='--',color='black')
        plt.axvline(x = 700, linewidth=1,linestyle='--',color='black')
        plt.axvline(x = 800, linewidth=1,linestyle='--',color='black')
        # plt.xticks(ticks=[400,500,600,700,800])
        plt.xlim(x_min,x_max)
        plt.grid(which='both',axis='both',color='grey', linestyle='--', linewidth=.3)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Spectral Energy')
    st.pyplot(Figure)
    
    
    st.markdown('<p class="font_subtext">Fig. 1: Photon and energy distribution for various investigated light treatment.</p>', unsafe_allow_html=True)
    st.write('For Blue wavebands (400-500 nm) of',Light_Treatment_Name, ' light treatment, photon flux density ratio is ',np.round(All_ratio[Light_Treatment_Name].iloc[0],2),', and energy ratio is ',np.round(All_ratio['Energy '+Light_Treatment_Name].iloc[0],2),'.')
    st.write('For Green wavebands (500-600 nm) of',Light_Treatment_Name, ' light treatment, photon flux density ratio is ',np.round(All_ratio[Light_Treatment_Name].iloc[1],2),', and energy ratio is ',np.round(All_ratio['Energy '+Light_Treatment_Name].iloc[1],2),'.')
    st.write('For Red wavebands (600-700 nm) of',Light_Treatment_Name, ' light treatment, photon flux density ratio is ',np.round(All_ratio[Light_Treatment_Name].iloc[2],2),', and energy ratio is ',np.round(All_ratio['Energy '+Light_Treatment_Name].iloc[2],2),'.')
    st.write('For Far-Red wavebands (700-800 nm) of',Light_Treatment_Name, ' light treatment, photon flux density ratio is ',np.round(All_ratio[Light_Treatment_Name].iloc[3],2),', and energy ratio is ',np.round(All_ratio['Energy '+Light_Treatment_Name].iloc[3],2),'.')

####################################################################################################################################################################
# Plant Growth Data
with tab2:

    st.markdown('<p class="font_header">Plant Growth Data: </p>', unsafe_allow_html=True)

    ####################################################################################################################################################################
    
    st.markdown('<p class="font_text"> In the previous section, we analyzed the spectral properties for different light treatments. Using obtained values for photon flux density and energy ration for the suggested wavebands, A dataset comprised of enery and photon flux density (PFD) ratio (Blue: 400-500, Green: 500-600, Red: 600-700, and Far-Red: 700-800 nm), temperature (T), relative humidity (RH), and carbon dioxide (CO2) concentation (both the average value and standard deviation), photoperiod hours and day where lettuce is harvested, in addition to fresh and dry mass of the harvested lettuce which is the average of 10 to 20 samples for each light treatment. One concern which may affect the developement of surrogate model (next stage of the project), is the fact the data itself is biased. Out of 231 observations, more than 73% of the data is for natural light in the greenhouse, while for the other 34 light treatment conditions have 60 observations. Moreover, we can see that while the impact of red, blue and far-red wavebands have been throughly investigated, there is small variation for green wavebands in the data. </p>', unsafe_allow_html=True)

    Plant_Data=pd.read_csv("Project Data.csv")

    col1,col2,col3=st.columns(3,gap='small')
    lettuce_stat = col1.checkbox('Show statistical properties of Lettuce dataset')

    if lettuce_stat==True:
        st.table(Plant_Data.describe())
        st.markdown('<p class="font_subtext">Table 3: Statistical properties of lettuce cultivated under different light treatment and environmental conidtions.</p>', unsafe_allow_html=True)

    lettuce_show = col2.checkbox('Show Lettuce dataset')

    if lettuce_show==True:
        st.table(Plant_Data)
        st.markdown('<p class="font_subtext">Table 4: Experimental observations for lettuce cultivated under different light treatments and various environmental conidtions.</p>', unsafe_allow_html=True)


    lettuce_hip = col3.checkbox('Show Lettuce hiplot')

    if lettuce_hip==True:
        xp = hip.Experiment.from_dataframe(Plant_Data)
        ret_val = xp.to_streamlit(ret="selected_uids", key="hip").display()
        st.markdown('<p class="font_subtext">Fig. 2: Hiplot for lettuce data.</p>', unsafe_allow_html=True)

####################################################################################################################################################################
# Plant Growth Distribution
with tab3:

    st.markdown('<p class="font_header">Plant Growth Distribution: </p>', unsafe_allow_html=True)
    
    st.markdown('<p class="font_text"> Next, distribution of plant growth data are presented based on the species or the light-treatments used for indoor-cultivation of lettuce.</p>',unsafe_allow_html=True)
    
    col3,col4=st.columns(2,gap='small')

    source = pd.DataFrame({"category": ['Rouxai','Rex','Cherokee'], "value": [Plant_Data.loc[Plant_Data["Species"]=="Rouxai"].shape[0],
                                                    Plant_Data.loc[Plant_Data["Species"]=="Rex"].shape[0],
                                                    Plant_Data.loc[Plant_Data["Species"]=="Cherokee"].shape[0]]})

    d=alt.Chart(source).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field="value", type="quantitative"),
        color=alt.Color(field="category", type="nominal",scale=alt.Scale(scheme='rainbow')),
        tooltip=("category","value")
    ).interactive()

    col3.altair_chart(d, use_container_width=True)

    Cat=[]
    for i in Light_Treatments.columns:
        if ('(' not in i) & ("Energy" not in i) & ("Wave" not in i):
            Cat=np.append(Cat,i)
            
    value=[];
    for i in Cat:
        value=np.append(value,Plant_Data.loc[Plant_Data["Treatment"]==i].shape[0])
    source1 = pd.DataFrame({"category": Cat, "value": value})

    g=alt.Chart(source1).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field="value", type="quantitative"),
        color=alt.Color(field="category", type="nominal",scale=alt.Scale(scheme='rainbow')),
        tooltip=("category","value")
    ).interactive()

    col4.altair_chart(g, use_container_width=True)

    st.markdown('<p class="font_subtext">Fig. 3: Categorical distribution of experimental observations.</p>', unsafe_allow_html=True)

####################################################################################################################################################################
# Interactive Visualization
with tab4:

    st.markdown('<p class="font_header">Interactive Visualization: </p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text">Several visualization are developed to study possible existing trend between different features of the dataset. Moreover, some of the figures are based on the target variable of the dataset which is the dry mass for the lettuce at the cultivation day. </p>', unsafe_allow_html=True)
    
    st.sidebar.markdown('<p class="font_text">Fig. 4: Matrix plot configuration:</p>', unsafe_allow_html=True)
    col1,col2=st.columns(2,gap='small')
    pairplot_options_x = col1.multiselect(
        'Select features for x-axis of matrixplot:',
        ['Energy', 'Energy (400-500)','Energy (500-600)', 'Energy (600-700)', 'Energy (700-800)', 'PFD','PFD (400-500)', 'PFD (500-600)', 'PFD (600-700)', 'PFD (700-800)',
        'CO2 ave', 'CO2 std', 'T ave', 'T std', 'RH ave', 'RH std','Photoperiod (h)', 'Day', 'Fresh Mass (g)', 'Dry Mass (g)'],default = "Energy")

    pairplot_options_y = col2.multiselect(
        'Select features for y-axis of matrixplot:',
        ['Energy', 'Energy (400-500)','Energy (500-600)', 'Energy (600-700)', 'Energy (700-800)', 'PFD','PFD (400-500)', 'PFD (500-600)', 'PFD (600-700)', 'PFD (700-800)',
        'CO2 ave', 'CO2 std', 'T ave', 'T std', 'RH ave', 'RH std','Photoperiod (h)', 'Day', 'Fresh Mass (g)', 'Dry Mass (g)'],default = "Energy")
        
    pairplot_hue = st.sidebar.select_slider(
        'Select hue for matrixplot:',
        options=['Species', 'Day' , 'Photoperiod (h)' , 'Treatment'],value='Treatment')


    fig1=sns.pairplot(data=Plant_Data,x_vars=pairplot_options_x,y_vars=pairplot_options_y, kind='scatter',hue=pairplot_hue,palette='hsv')


    c=alt.Chart(Plant_Data).mark_circle().encode(
        alt.X(alt.repeat("column"), type='quantitative'),
        alt.Y(alt.repeat("row"), type='quantitative'),
        color=pairplot_hue,
        tooltip=['Dry Mass (g)', 'Photoperiod (h)', 'RH ave', 'CO2 ave']
    ).properties(
        width=280,
        height=280
    ).repeat(
        row=pairplot_options_y,
        column=pairplot_options_x
    ).interactive()

    st.altair_chart(c, use_container_width=True)
    st.markdown('<p class="font_subtext">Fig. 4: Matrix plot for lettuce growth dataset.</p>', unsafe_allow_html=True)

    ####################################################################################################################################################################

    # st.markdown('<p class="font_text">Dry Mass Heatmap based on Red and Blue Wavebands:</p>', unsafe_allow_html=True)
    #st.markdown('<p class="font_subsubheader"> Correlation between  </p>', unsafe_allow_html=True)
    tab17, tab18 = st.tabs(["Heatmap", "Jointplot"])
    with tab17:
        col1,col2=st.columns(2,gap='small')
        option1 = col1.selectbox(
            'Studied feature 1:',
            ('Energy', 'Energy (400-500)','Energy (500-600)', 'Energy (600-700)', 'Energy (700-800)', 'PFD','PFD (400-500)', 'PFD (500-600)', 'PFD (600-700)', 'PFD (700-800)',
            'CO2 ave', 'CO2 std', 'T ave', 'T std', 'RH ave', 'RH std','Photoperiod (h)', 'Day'),index=1)
        option2 = col2.selectbox(
            'Studied feature 2:',
            ('Energy', 'Energy (400-500)','Energy (500-600)', 'Energy (600-700)', 'Energy (700-800)', 'PFD','PFD (400-500)', 'PFD (500-600)', 'PFD (600-700)', 'PFD (700-800)',
            'CO2 ave', 'CO2 std', 'T ave', 'T std', 'RH ave', 'RH std','Photoperiod (h)', 'Day'),index=2)
        option3 = col1.selectbox('Dry Mass or Fresh Mass',('Dry Mass (g)', 'Fresh Mass (g)'))
        option4_species = col2.multiselect(
        'Select species of cultivation:',
        ['Rouxai','Rex','Cherokee'],default = "Rex")
        if len(option4_species) ==1:
            Part_Plant_Data=Plant_Data.loc[(Plant_Data["Species"]==option4_species[0])]
        elif len(option4_species) ==2:
            Part_Plant_Data=Plant_Data.loc[(Plant_Data["Species"]==option4_species[0])|(Plant_Data["Species"]==option4_species[1])]
        else:
            Part_Plant_Data=Plant_Data.loc[(Plant_Data["Species"]==option4_species[0])|(Plant_Data["Species"]==option4_species[1])|(Plant_Data["Species"]==option4_species[2])]
        heatmap = alt.Chart(Part_Plant_Data).mark_rect().encode(
            alt.X(option1+':Q', bin=True),
            alt.Y(option3+':Q', bin=True),
            alt.Color('count()', scale=alt.Scale(scheme='greenblue'))
        ).properties(
            height=500,
            width=700
        )
        points = alt.Chart(Part_Plant_Data).mark_circle(
            color='black',
            size=5,
        ).encode(
            x=option1+':Q',
            y=option3+':Q',
        ).properties(
            height=500,
            width=700
        )
        G=heatmap+points
        col1.altair_chart(G, use_container_width=True)
        heatmap = alt.Chart(Part_Plant_Data).mark_rect().encode(
            alt.X(option2+':Q', bin=True),
            alt.Y(option3+':Q', bin=True),
            alt.Color('count()', scale=alt.Scale(scheme='greenblue'))
        ).properties(
            height=500,
            width=700
        )
        points = alt.Chart(Part_Plant_Data).mark_circle(
            color='black',
            size=5,
        ).encode(
            x=option2+':Q',
            y=option3+':Q',
        ).properties(
            height=500,
            width=700
        )
        G=heatmap+points
        col2.altair_chart(G, use_container_width=True)
        st.markdown('<p class="font_subtext">Fig. 5: Heatmap of lettuce mass with respect to a feature.</p>', unsafe_allow_html=True)

    with tab18:
        option3 = st.selectbox(
            'Feature 1',
            ('Energy', 'Energy (400-500)','Energy (500-600)', 'Energy (600-700)', 'Energy (700-800)', 'PFD','PFD (400-500)', 'PFD (500-600)', 'PFD (600-700)', 'PFD (700-800)',
            'CO2 ave', 'CO2 std', 'T ave', 'T std', 'RH ave', 'RH std','Photoperiod (h)', 'Day'),index=1)

        option4 = st.selectbox(
            'Feature 2',
            ('Energy', 'Energy (400-500)','Energy (500-600)', 'Energy (600-700)', 'Energy (700-800)', 'PFD','PFD (400-500)', 'PFD (500-600)', 'PFD (600-700)', 'PFD (700-800)',
            'CO2 ave', 'CO2 std', 'T ave', 'T std', 'RH ave', 'RH std','Photoperiod (h)', 'Day'),index=2)

        option5 = st.selectbox(
            'Color map:',
            ('mako','viridis','rocket','Spectral','coolwarm','cubehelix','dark:salmon_r'))

        option6 = st.slider('Number of contour level:', 0, 200, 20)

        sns.set_theme(style="white")
        fig = sns.JointGrid(data=Plant_Data, x=option3, y=option4, space=0)
        fig.plot_joint(sns.kdeplot,
                     fill=True,
                     thresh=0, levels=option6, cmap=option5)
        fig.plot_marginals(sns.histplot, color="blue", alpha=1, bins=30)
        st.pyplot(fig)
        st.markdown('<p class="font_subtext">Fig. 5: Jointplot for two of the investigated features.</p>', unsafe_allow_html=True)

    ####################################################################################################################################################################
    #col7,col8=st.columns(2,gap='small')
    st.sidebar.markdown('<p class="font_text">Fig. 6: 3D Scatter Plot:</p>', unsafe_allow_html=True)
    Scatter_3D_X =st.sidebar.selectbox(
        "Fig. 6: x-axis feature for 3D scatter plot:",
        ['Energy', 'Energy (400-500)','Energy (500-600)', 'Energy (600-700)', 'Energy (700-800)', 'PFD','PFD (400-500)', 'PFD (500-600)', 'PFD (600-700)', 'PFD (700-800)','CO2 ave', 'CO2 std', 'T ave', 'T std', 'RH ave', 'RH std','Photoperiod (h)', 'Day'],index=1)

    Scatter_3D_Y =st.sidebar.selectbox(
        "Fig. 6: y-axis feature for 3D scatter plot:",
        ['Energy', 'Energy (400-500)','Energy (500-600)', 'Energy (600-700)', 'Energy (700-800)', 'PFD','PFD (400-500)', 'PFD (500-600)', 'PFD (600-700)', 'PFD (700-800)','CO2 ave', 'CO2 std', 'T ave', 'T std', 'RH ave', 'RH std','Photoperiod (h)', 'Day'],index=2)

    Scatter_3D_Z =st.sidebar.selectbox(
        "Fig. 6: Z-axis feature for 3D scatter plot:",
        ['Dry Mass (g)', 'Fresh Mass (g)'])

    Scatter_3D_hue =st.sidebar.selectbox(
        "Fig. 6: hue for 3D scatter plot:",
        ['Species', 'Treatment'])

    fig=px.scatter_3d(Plant_Data, x=Scatter_3D_X, y=Scatter_3D_Y, z=Scatter_3D_Z,opacity = 0.7,height=600,
        width=1200,color=Scatter_3D_hue)
    st.plotly_chart(fig)
    st.markdown('<p class="font_subtext">Fig. 6: 3D scatter plot of lettuce mass versus other features.</p>', unsafe_allow_html=True)

####################################################################################################################################################################
#Linear Regression

with tab5:
    data=pd.read_csv("Project Data.csv")
    st.markdown('<p class="font_header">Linear Regression:</p>', unsafe_allow_html=True)
    Scaler = st.checkbox('Applying Scaler object for linear regression fitting')
    col1 , col2 , col3 , col4= st.columns(4,gap='small')
    Target_Variable = col3.selectbox('Select target feature:',['Fresh Mass (g)', 'Dry Mass (g)'],index = 0)
    Feature_Variable = col2.multiselect(
        'Select feature(s) for linear regression:',
        ['Energy', 'Energy (400-500)','Energy (500-600)', 'Energy (600-700)', 'Energy (700-800)', 'PFD','PFD (400-500)', 'PFD (500-600)', 'PFD (600-700)', 'PFD (700-800)',
        'CO2 ave', 'CO2 std', 'T ave', 'T std', 'RH ave', 'RH std','Photoperiod (h)', 'Day'],default = 'Energy')
    Y_Linear = data[Target_Variable].to_numpy()
    X_Linear = data[Feature_Variable].to_numpy()
    Index=np.linspace(0,Y_Linear.size-1,Y_Linear.size).astype(int)
    if Scaler:
        Scaler_Type = col4.selectbox('Select scaler object:',['Min-Max Scaler', 'Standard Scaler', 'Max-Abs Scaler'],index = 0)
        if Scaler_Type == 'Min-Max Scaler':
            Scaler_Object = MinMaxScaler()
        elif Scaler_Type == 'Standard Scaler':
            Scaler_Object = StandardScaler()
        else:
            Scaler_Object = MaxAbsScaler()
        X_Scaled =Scaler_Object.fit_transform(X_Linear)
    Linear_Regression = col1.selectbox('Select linear regression function:',['Linear', 'Ridge' , 'Lasso', 'Elastic net'],index = 0)
    if Linear_Regression == 'Linear':
        Linear_Regression_Object = LinearRegression()
    elif Linear_Regression == 'Ridge':
        col_1, col_2, col_3 = st.columns(3, gap = 'small')
        Solver_Ridge = col_1.selectbox('Select solver for ridge:',['auto', 'svd', 'sag'],index = 0)
        Alpha_Ridge = col_2.number_input('Input a non-negative value for alpha: ',value=1.00,format='%f')
        Random_State_Ridge = col_3.slider('Input a value for random state', 0, 200, 40)
        Linear_Regression_Object = Ridge(alpha = Alpha_Ridge, random_state = Random_State_Ridge, solver = Solver_Ridge)
    elif Linear_Regression == 'Lasso':
        col_1, col_2 = st.columns(2, gap = 'small')
        Alpha_Lasso = col_1.number_input('Input a non-negative value for alpha: ',value=1.00,format='%f')
        Random_State_Lasso = col_2.slider('Input a value for random state', 0, 200, 40)
        Linear_Regression_Object = Lasso(alpha = Alpha_Lasso, random_state = Random_State_Lasso)
    else:
        col_1, col_2, col_3 = st.columns(3, gap = 'small')
        Alpha_Elastic = col_1.number_input('Input a non-negative value for alpha: ',value=1.00,format='%f')
        L1_Ration_Elastic = col_2.number_input('Input a value for L1 ration between 0 and 1: ',value=0.500,step=0.001,format='%f')
        Random_State_Elastic = col_3.slider('Input a value for random state', 0, 200, 40)
        Linear_Regression_Object = ElasticNet(alpha = Alpha_Elastic, random_state = Random_State_Elastic, l1_ratio = L1_Ration_Elastic)
    
    if Scaler:
        reg=Linear_Regression_Object.fit(X_Scaled, Y_Linear)
        score=reg.score(X_Scaled, Y_Linear)
        Y_Predic = reg.predict(X_Scaled)
    else:
        reg=Linear_Regression_Object.fit(X_Linear, Y_Linear)
        score=reg.score(X_Linear, Y_Linear)
        Y_Predic = reg.predict(X_Linear)
    st.write('For linear regression methods ',Linear_Regression, ', the accuracy score based on r2 ',np.round(score,2),'.')
    st.write(Linear_Regression,' comparison plot:')
    Linear_Dataframe=pd.DataFrame(index=np.arange(len(Y_Linear)), columns=np.arange(3))
    Linear_Dataframe.columns=['Index','Actual','Predict']
    Linear_Dataframe.iloc[:,0]=Index
    Linear_Dataframe.iloc[:,1]=Y_Linear
    Linear_Dataframe.iloc[:,2]=Y_Predic
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Linear_Dataframe['Index'], y=Linear_Dataframe['Actual'],marker_symbol='square',
                        mode='markers',
                        name='Actual'))
    fig.add_trace(go.Scatter(x=Linear_Dataframe['Index'], y=Linear_Dataframe['Predict'],marker_symbol='circle',
                        mode='markers',
                        name='Prediction'))

    st.plotly_chart(fig)
    # Coef_Linear = reg.coef_
    # st.write(Linear_Regression,' Correlations:')
    # fig = make_subplots(rows=1, cols=len(Feature_Variable))
    # for i in range (len(Feature_Variable)):
        # fig1 = go.Figure()
        # fig1.add_trace(go.Scatter(x=data[Feature_Variable[i]], y=data[Target_Variable],marker_symbol='square',mode='markers',name='Actual'))
        # X_Linsp = np.linspace(data[Feature_Variable[i]].min(),data[Feature_Variable[i]].max(),100)
        # Y_Linsp = X_Linsp*Coef_Linear[i]
        # fig1.add_trace(go.Scatter(x=X_Linsp, y=Y_Linsp))
        # st.plotly_chart(fig1)
    

####################################################################################################################################################################
# Neural Network Visualization
with tab6:
    st.markdown('<p class="font_text">(Deep) Neural Network Visualization:</p>', unsafe_allow_html=True)
    #Neural Network Representation

    Num_Hidden_Layer = st.selectbox('Number of Hidden Layers: ',(1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20), index=0)
    
    col1 , col2 = st.columns(2,gap='small')
    Target_Variable_DNN = col2.multiselect('Select target feature for Neural Network:',['Fresh Mass (g)', 'Dry Mass (g)'],default = 'Dry Mass (g)')
    Feature_Variable_DNN = col1.multiselect(
        'Select feature(s) for Neural Network:',
        ['Energy', 'Energy (400-500)','Energy (500-600)', 'Energy (600-700)', 'Energy (700-800)', 'PFD','PFD (400-500)', 'PFD (500-600)', 'PFD (600-700)', 'PFD (700-800)',
        'CO2 ave', 'CO2 std', 'T ave', 'T std', 'RH ave', 'RH std','Photoperiod (h)', 'Day'],default = 'Energy')
    
    col = st.columns(Num_Hidden_Layer)

    Num_Neuron=np.zeros(Num_Hidden_Layer)
    for j in range (Num_Hidden_Layer):
        with col[j]:
            Num_Neuron[j] = st.slider('Number of Neurons in '+str(j+1)+' Hidden Layer', min_value=1, max_value=45, value=10, step=1)
    Num_Neuron=Num_Neuron.astype(int)
    DENSE = True
    SPARSE = False

    PENWIDTH = '15'
    FONT = 'Hilda 10'

    layer_nodes = np.zeros(Num_Hidden_Layer+2).astype(int)
    for i in range (len(layer_nodes)):
        if i ==0:
            layer_nodes[i]=len(Feature_Variable_DNN)
        elif i == len(layer_nodes)-1:
            layer_nodes[i]=len(Target_Variable_DNN)
        else:
            layer_nodes[i]=Num_Neuron[i-1]
                
    connections = DENSE 

    dot = graph.Digraph(comment='Neural Network', 
                    graph_attr={'nodesep':'0.04', 'ranksep':'0.05', 'bgcolor':'white', 'splines':'line', 'rankdir':'LR', 'fontname':FONT},
                    node_attr={'fixedsize':'true', 'label':"", 'style':'filled', 'color':'none', 'fillcolor':'gray', 'shape':'circle', 'penwidth':'10', 'width':'0.4', 'height':'0.4'},
                    edge_attr={'color':'black1', 'arrowsize':'.4','penwidth':'0.4'})

    for layer_no in range(len(layer_nodes)):
        with dot.subgraph(name='cluster_'+str(layer_no)) as c:
            c.attr(color='transparent') # comment this if graph background is needed
            if layer_no == 0:                 # first layer
                c.attr(label='Input Layer')
            elif layer_no == len(layer_nodes)-1:   # last layer
                c.attr(label='Output Layer')
            else:                      # layers in between
                c.attr(label=' Hidden Layer '+ str(layer_no) )
            for a in range(layer_nodes[layer_no]):
                if layer_no == 0: # or i == len(layers)-1: # first or last layer
                    c.node('l'+str(layer_no)+str(a), 'I', fontcolor='white', fillcolor='navyblue')#, fontcolor='white'
                elif layer_no == len(layer_nodes)-1:
                    c.node('l'+str(layer_no)+str(a), 'O', fontcolor='white', fillcolor='violetred')#, fontcolor='white'
                else:
                    # unicode characters can be used to inside the nodes as follows
                    # for a list of unicode characters refer this https://pythonforundergradengineers.com/unicode-characters-in-python.html
                    c.node('l'+str(layer_no)+str(a), 'H'+str(layer_no), fontsize='12', fillcolor='green') # to place "sigma" inside the nodes of a layer


    for layer_no in range(len(layer_nodes)-1):
        for node_no in range(layer_nodes[layer_no]):
            if connections == DENSE:
                for b in range(layer_nodes[layer_no+1]):
                    dot.edge('l'+str(layer_no)+str(node_no), 'l'+str(layer_no+1)+str(b),)
    st.graphviz_chart(dot)

####################################################################################################################################################################
# Neural Network Regression
with tab7:
    st.write(' ')
    st.markdown('<p class="font_text">(Deep) Neural Network Regression based on the neural network architecture in the previous tab including hidden layer configuration, number of input features, and number of target features:</p>', unsafe_allow_html=True)
    st.write(' ')
    New_Neural = st.checkbox('Do you want a new architecture for (deep) neural network regression? (unchecked box implies using existing architecture from neural network visualization)')
    if New_Neural:
        Num_Hidden_Layer_DNN = st.selectbox('Number of Hidden Layers for NN Regression: ',(1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20), index=0)
        
        col1 , col2 = st.columns(2,gap='small')
        Target_Variable_DNN = col2.multiselect('Select target feature for Neural Network Regression:',['Fresh Mass (g)', 'Dry Mass (g)'],default = 'Dry Mass (g)')
        Feature_Variable_DNN = col1.multiselect(
            'Select feature(s) for Neural Network Regression:',
            ['Energy', 'Energy (400-500)','Energy (500-600)', 'Energy (600-700)', 'Energy (700-800)', 'PFD','PFD (400-500)', 'PFD (500-600)', 'PFD (600-700)', 'PFD (700-800)',
            'CO2 ave', 'CO2 std', 'T ave', 'T std', 'RH ave', 'RH std','Photoperiod (h)', 'Day'],default = 'Energy')
        
        cols = st.columns(Num_Hidden_Layer_DNN)

        Num_Neuron_DNN=np.zeros(Num_Hidden_Layer_DNN)
        for j in range (Num_Hidden_Layer_DNN):
            with cols[j]:
                Num_Neuron_DNN[j] = st.slider('Number of Neurons in '+str(j+1)+' Hidden Layer for regression:', min_value=1, max_value=45, value=10, step=1)
        Num_Neuron_DNN=Num_Neuron_DNN.astype(int)
        st.write(' ')
        st.write('For neural network with ',Num_Hidden_Layer_DNN,' hidden layers, and ', str(Num_Neuron_DNN[:]), 'for number of neurons within those hidden layers,' , len(Target_Variable_DNN),' for number of target feature(s), and ',len(Feature_Variable_DNN),' for number of input feature(s).')
        st.write(' ')
    else:
        st.write(' ')
        st.write('For neural network with ',Num_Hidden_Layer,' hidden layers, and ', str(Num_Neuron[:]), 'for number of neurons within those hidden layers,' , len(Target_Variable_DNN),' for number of target feature(s), and ',len(Feature_Variable_DNN),' for number of input feature(s).')
        st.write(' ')
    
    col1 , col2 , col3 , col4, col5= st.columns(5,gap='small')
    st.write(' ')
    Activation_DNN = col1.selectbox('Select activation function:',['identity', 'relu', 'logistic', 'tanh'],index = 0)
    
    Solver_DNN = col2.selectbox('Select solver type:',['adam', 'sgd', 'lbfgs'],index = 0)
    
    Alpha_DNN = col3.number_input('Input a non-negative value for alpha: ',value=0.01,format='%f')
    
    Learning_Rate_DNN = col4.selectbox('Select learning rate type:',['constant', 'invscaling', 'adaptive'],index = 0)
    
    Learning_Rate_Init_DNN = col5.number_input('Input a value for initial learning rate: ',value=0.001,format='%f')
    st.write(' ')
    col1 , col2 , col3 , col4, col5= st.columns(5,gap='small')
    st.write(' ')
    Validation_Fraction_DNN = col2.number_input('Input a value for validation fraction:',value=0.2,format='%f')
    
    Max_Iteration_DNN = col3.slider('Input a value for number of iteration:', 0, 20000, 200)
    
    Random_State_DNN = col5.slider('Input a value for random state', 0, 200, 40)
    
    Tolerence_DNN = col1.number_input('Input a value for tolerence: ',value=0.0001,format='%f')
    
    Batch_Size_DNN = col4.slider('Input a value for batch size:', 0, len(Y_Linear), 40)
    st.write(' ')
    col1, col2, col3= st.columns(3,gap='small')
    st.write(' ')
    Y_DNN = data[Target_Variable_DNN].to_numpy()
    
    X_DNN = data[Feature_Variable_DNN].to_numpy()
    
    Train_Size_DNN = col1.number_input('Input a value for train-size ratio:',value=0.8,format='%f')
    
    Scaler = col2.checkbox('Applying Scaler object for neural network regression')
    
    X_Train_DNN, X_Test_DNN, Y_Train_DNN, Y_Test_DNN = train_test_split(X_DNN, Y_DNN, train_size=Train_Size_DNN)
    
    if Scaler:
        Scaler_Type = col3.selectbox('Select scaler object:',['Min-Max Scaler', 'Standard Scaler', 'Max-Abs Scaler'],index = 0)
        if Scaler_Type == 'Min-Max Scaler':
            Scaler_Object = MinMaxScaler()
        elif Scaler_Type == 'Standard Scaler':
            Scaler_Object = StandardScaler()
        else:
            Scaler_Object = MaxAbsScaler()
        X_Scaled_DNN = Scaler_Object.fit_transform(X_DNN)
        X_Train_Scaled_DNN =Scaler_Object.transform(X_Train_DNN)
        X_Test_Scaled_DNN =Scaler_Object.transform(X_Test_DNN)
    
    if len(Target_Variable_DNN)==1:
        MLP_Object=MLPRegressor(hidden_layer_sizes=Num_Neuron, activation=Activation_DNN, solver=Solver_DNN,
             alpha=Alpha_DNN, batch_size=Batch_Size_DNN, learning_rate=Learning_Rate_DNN,
             learning_rate_init=Learning_Rate_Init_DNN, max_iter=Max_Iteration_DNN, shuffle=True)
    else:
        MLP_Object=MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=Num_Neuron, activation=Activation_DNN, solver=Solver_DNN,
             alpha=Alpha_DNN, batch_size=Batch_Size_DNN, learning_rate=Learning_Rate_DNN,
             learning_rate_init=Learning_Rate_Init_DNN, max_iter=Max_Iteration_DNN, shuffle=True))
    
    if Scaler:
        MLP_Object.fit(X_Train_Scaled_DNN, Y_Train_DNN)
        score_DNN =MLP_Object.score(X_Test_Scaled_DNN, Y_Test_DNN)
        Y_Predic_DNN = MLP_Object.predict(X_Scaled_DNN)
    else:
        MLP_Object.fit(X_Train_DNN, Y_Train_DNN)
        score_DNN =MLP_Object.score(X_Test_DNN, Y_Test_DNN)
        Y_Predic_DNN = MLP_Object.predict(X_DNN)
    st.write(' ')
    st.markdown('<p class="font_text">Accuracy of the investigated (deep) neural network architecture:</p>', unsafe_allow_html=True)
    st.write(' ')
    if len(Target_Variable_DNN)==1:
        DNN_Dataframe=pd.DataFrame(index=np.arange(len(Y_Linear)), columns=np.arange(3))
        DNN_Dataframe.columns=['Index','Actual','Predict']
        DNN_Dataframe.iloc[:,0]=Index
        DNN_Dataframe.iloc[:,1]=Y_DNN
        DNN_Dataframe.iloc[:,2]=Y_Predic_DNN
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=DNN_Dataframe.iloc[:,0], y=DNN_Dataframe.iloc[:,1],marker_symbol='square',
                            mode='markers',
                            name='Actual '+Target_Variable_DNN[0] + ' vs. Index'))
        fig2.add_trace(go.Scatter(x=DNN_Dataframe.iloc[:,0], y=DNN_Dataframe.iloc[:,2],marker_symbol='circle',
                            mode='markers',
                            name='Prediction '+Target_Variable_DNN[0] + ' vs. Index'))
    else:
        DNN_Dataframe=pd.DataFrame(index=np.arange(len(Y_Linear)), columns=np.arange(5))
        DNN_Dataframe.columns=['Index','Actual '+Target_Variable_DNN[0],'Predict '+Target_Variable_DNN[0],'Actual '+Target_Variable_DNN[1],'Predict '+Target_Variable_DNN[1]]
        DNN_Dataframe.iloc[:,0]=Index
        DNN_Dataframe.iloc[:,1]=Y_DNN[:,0]
        DNN_Dataframe.iloc[:,2]=Y_Predic_DNN[:,0]
        DNN_Dataframe.iloc[:,3]=Y_DNN[:,1]
        DNN_Dataframe.iloc[:,4]=Y_Predic_DNN[:,1]
        fig2 = make_subplots(rows=1, cols=2)
        fig2.add_trace(
            go.Scatter(x=DNN_Dataframe.iloc[:,0], y=DNN_Dataframe.iloc[:,1],marker_symbol='square',
                            mode='markers',
                            name='Actual '+Target_Variable_DNN[0] + ' vs. Index'),
            row=1, col=1
        )
        fig2.add_trace(
            go.Scatter(x=DNN_Dataframe.iloc[:,0], y=DNN_Dataframe.iloc[:,2],marker_symbol='circle',
                            mode='markers',
                            name='Predict '+Target_Variable_DNN[0] + ' vs. Index'),
            row=1, col=1
        )

        fig2.add_trace(
            go.Scatter(x=DNN_Dataframe.iloc[:,0], y=DNN_Dataframe.iloc[:,3],marker_symbol='square',
                            mode='markers',
                            name='Actual '+Target_Variable_DNN[1] + ' vs. Index'),
            row=1, col=2
        )
        fig2.add_trace(
            go.Scatter(x=DNN_Dataframe.iloc[:,0], y=DNN_Dataframe.iloc[:,4],marker_symbol='circle',
                            mode='markers',
                            name='Predict '+Target_Variable_DNN[1] + ' vs. Index'),
            row=1, col=2
        )
        
    st.plotly_chart(fig2)    
    st.markdown('<p class="font_text">Learning curve based on the above hyper-parameters:</p>', unsafe_allow_html=True)


####################################################################################################################################################################

st.markdown('<p class="font_header">References: </p>', unsafe_allow_html=True)
st.markdown('<p class="font_text">Data used for plant growth visualization are obtained from the following refrences: </p>', unsafe_allow_html=True)
st.markdown('<p class="font_text">1) Meng, Q., Boldt, J., and Runkle, E. S. (2020). Blue radiation interacts with green radiation to influence growth and predominantly controls quality attributes of lettuce. Journal of the American Society for Horticultural Science 145, 75–87280. </p>', unsafe_allow_html=True)
st.markdown('<p class="font_text">2) Meng, Q., Kelly, N., and Runkle, E. S. (2019). Substituting green or far-red radiation for blue radiation induces shade avoidance and promotes growth in lettuce and kale. Environmental and experimental botany 162, 383–391283. </p>', unsafe_allow_html=True)
st.markdown('<p class="font_text">3) Meng, Q. and Runkle, E. S. (2019). Far-red radiation interacts with relative and absolute blue and red photon flux densities to regulate growth, morphology, and pigmentation of lettuce and basil seedlings. Scientia Horticulturae 255, 269–280. </p>', unsafe_allow_html=True)

##################################################################################################################################################################
