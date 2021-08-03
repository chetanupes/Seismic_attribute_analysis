from functools import total_ordering
from os import write
from IPython.core.display import Image
from numpy.core.numeric import False_
import pandas as pd
import numpy as np
#Streamlit
import streamlit as st
from streamlit.state.session_state import Value
from streamlit_pandas_profiling import st_profile_report
st.set_page_config(layout="wide")

#Plotting
import seaborn as sns
import plotly.express as px
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
plt.style.use('seaborn')

#Warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

#Widgets libraries
import ipywidgets as wd
from IPython.display import display, clear_output
from ipywidgets import interactive
from PIL import Image

#Data Analysis Library
from pandas_profiling import ProfileReport
from sklearn.mixture import GaussianMixture
from sklearn import mixture
from sklearn import metrics
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.cluster import MeanShift, estimate_bandwidth

#############################################################################################################################################

st.title('Geological Formation Identification Using Seismic Attributes')

st.write('This project uses KMeans, GMM(Gaussian Mixture Modelling) and Mean Shift algorithms for identifying geological formation using seismic attributes.')
st.write('1. **K-means** clustering is a distance-based algorithm (Euclidean Distance or the Manhattan Distance). This means that it tries to group the closest points to form a cluster. It is used to group data into K number of clusters by minimising the distance between the data point and the centroid.')
st.write('2. **Gaussian Mixture Models (GMMs)** assume that there are a certain number of Gaussian distributions, and each of these distributions represent a cluster. Hence, a Gaussian Mixture Model tends to group the data points belonging to a single distribution together. The GMM method also allows data points to be clustered, except that it accounts for data variance, results in a softer classification, and rather than being distance-based it is distribution-based.')
st.write('3. **Mean Shift** looks at the “mode” of the density, and where it is highest, it will iteratively shift points in the plot towards the closest mode – resulting in a number of clusters. This way, even when the clusters aren’t perfectly separated, Mean Shift will likely be able to detect them anyway.')


st.title('Dataset Report')

attr=pd.DataFrame(pd.read_csv('AttributeData.txt', delimiter = "\t"))

#Creating a profile report

report=ProfileReport(attr, title='Profiling Reoprt')

if st.checkbox('Preview Profile Report'):
    st_profile_report(report)
st.write('The profiling function presents the user with a descriptive statistical summary of all the features of the dataset.')

#Dropping duplicates and where y is 0
df=attr.copy()
df=df[df['Y']!=0]
df=df.drop_duplicates()

#PLotting clean data
st.title('Plotting Cleaned Data')
st.write('1. Data is cleaned removing duplicates and where attribute Y=0')

fig=px.scatter(df,x=df.X, y=df.Y, labels={'color': 'KMean'}, color_discrete_sequence=['red'])
fig.update_xaxes(title_text='Attribute (X)')
fig.update_yaxes(title_text='Attribute (Y)')
fig.update_layout(showlegend=True, height=800, width=800)
st.plotly_chart(fig, use_container_width=True)

# Preprocessing the dataset
X=df
scale = StandardScaler()
x_scaled = scale.fit_transform(X)
df_scaled = pd.DataFrame(x_scaled)

# Selecting the method for analysis   
st.sidebar.markdown("# Machine Learning Models")
Select_Method=st.sidebar.selectbox('Select a Model', ('KMeans','GMM (Gaussian Mixture Modelling)', 'Mean Shift'))

#Cluster 
st.sidebar.markdown("# Selecting Clusters")
if Select_Method=='Mean Shift':
    st.sidebar.write('For Mean Shift clusters are selected automatically by the algorithm.')
else:
    Select_cluster=st.sidebar.selectbox('Select clusters', ('Manually','Optimal clusters (calculated)'))

if Select_Method!='Mean Shift':
    st.title('Clusters Analysis')
    st.write('Cluster selection can be done by the interpretor either manually, can be calculated (optimal clusters) or automatically selected by some algorithm.')
    st.write('KMeans and GMM cluster selection can be done using the Elbow plot or the Silhoutte plot. MeanShift automatically selects the number of clusters. Some of these plots are depicted below.')
    st.write('**Note: ** These plots are not calculated on the fly to save computation time as data does not change for the analysis.') 

    st.write('**1. Elbow Method**')
    st.write('Calculates the inertia ( a measure of how internally coherent clusters are) or within-cluster-sum of squared errors (WSS) for different values of k, and choose the k for which WSS first starts to diminish. In the plot of Inertia vs k, this is visible as an elbow.')
    st.latex('\sum_{i=0}^{n} min(||x_{i}-u_{j}||^2')
    
    st.write('**2. Silhouette Method**')
    st.write('The silhouette coefficient is a measure of how similar a data point is within-cluster (cohesion) compared to other clusters (separation).')
    st.latex(r'''S(i)=\left(\frac{b(i)-a(i)}{max(a(i),b(i))}\right)''')
    st.write('**a)** *S(i)* is the silhouette coefficient of the data point *i*.')
    st.write('**b)** *a(i)* is the average distance between *i* and all the other data points in the cluster to which *i* belongs.')
    st.write('**c)** *b(i)* is the average distance from *i* to all clusters to which *i* does not belong.')

    if Select_cluster=='Manually':
        cluster_value = st.sidebar.slider('Select a cluster value',2, 10, value=3)
    if Select_cluster=='Optimal clusters (calculated)':
        st.sidebar.write('Optimal number of clusters calculated are 5.')

    # Plot!
    kmean_image=Image.open('KMean-Manually.png')
    st.image(kmean_image, width=None)

    # Plot!
    kmean_image1=Image.open('Sil Plot-KMean.png')
    st.image(kmean_image1, width=None)

    # Plot!
    kmean_image2=Image.open('KMean-Auto.png')
    st.image(kmean_image2, width=None)
    #st.markdown("<h1 style='text-align: center; color: red;'>Optimal number of clusters calculated=5</h1>", unsafe_allow_html=True)
    st.write('*Optimal number of clusters calculated=5*')

    # Plot!
    gmm_image=Image.open('Sil Plot-GMM.png')
    st.image(gmm_image, width=None)

################################################################################################################################################
#Now we can do the analysis for both KMean and GMM
st.title('Projecting Features for {}'.format(Select_Method))

#df_scaled_embedded = TSNE(n_components=2,learning_rate=200,random_state=10,perplexity=50).fit_transform(df_scaled) #t-SNE

df1=pd.read_excel('test.xlsx')
df_scaled_embedded=df1.to_numpy()
#Plotting KMean and GMM data in 2d


def plot_KG(ana, clus):
    
    if (ana=='KMeans') & (clus=='Manually'):
        # k-means implementation with selected clusters
        kmeans = KMeans(n_clusters=cluster_value,random_state=42)
        kmeans.fit(df_scaled)
        labels_rocks = kmeans.predict(df_scaled)
        df['KMean'] = labels_rocks

        # Projecting the well log features into 2d projection using t-SNE
        fig2=px.scatter(df_scaled_embedded,x=0, y=1, color=df.KMean, labels={'color': 'KMean'})
        fig2.update_layout(showlegend=True, height=800, width=800,title_text='t-SNE 2D Projection', coloraxis_showscale=False)

        # Plot!
        st.plotly_chart(fig2, use_container_width=True)

    if (ana=='KMeans') & (clus=='Optimal clusters (calculated)'):
            # k-means implementation with 5 clusters
        kmeans = KMeans(n_clusters=5,random_state=42)
        kmeans.fit(df_scaled)
        labels_rocks = kmeans.predict(df_scaled)
        df['KMean'] = labels_rocks

        # Projecting the well log features into 2d projection using t-SNE
        fig2=px.scatter(df_scaled_embedded,x=0, y=1, color=df.KMean, labels={'color': 'KMean'})
        fig2.update_layout(showlegend=True, height=800, width=800,title_text='t-SNE 2D Projection',coloraxis_showscale=False)
        
        # Plot!
        st.plotly_chart(fig2, use_container_width=True)

    if (ana=='GMM (Gaussian Mixture Modelling)') & (clus=='Manually'):
        gmm=GaussianMixture(n_components=cluster_value,random_state=42)
        gmm.fit(df_scaled)
        labels_rocks1 = gmm.predict(df_scaled)
        df['GMM'] = labels_rocks1

        # Projecting the well log features into 2d projection using t-SNE
        fig3=px.scatter(df_scaled_embedded,x=0, y=1, color=df.GMM, labels={'color': 'GMM'})
        fig3.update_layout(showlegend=False, height=800, width=800,title_text='t-SNE 2D Projection',coloraxis_showscale=False)

        # Plot!
        st.plotly_chart(fig3, use_container_width=True)
    elif (ana=='GMM (Gaussian Mixture Modelling)') & (clus=='Optimal clusters (calculated)'):
        gmm=GaussianMixture(n_components=5,random_state=42)
        gmm.fit(df_scaled)
        labels_rocks1 = gmm.predict(df_scaled)
        df['GMM'] = labels_rocks1

        # Projecting the well log features into 2d projection using t-SNE
        fig4=px.scatter(df_scaled_embedded,x=0, y=1, color=df.GMM, labels={'color': 'GMM'})
        fig4.update_layout(showlegend=False, height=800, width=800,title_text='t-SNE 2D Projection',coloraxis_showscale=False)

        # Plot!
        st.plotly_chart(fig4, use_container_width=True)

#Plotting Mean shift in 2d
def plot_MS(ana):
    
    #bandwidth = estimate_bandwidth(df_scaled, quantile=0.2, n_samples=500)
    #meanshift = MeanShift(bandwidth=bandwidth)
    #meanshift.fit(df_scaled)
    #labels_rocks2 = meanshift.predict(df_scaled)
    #df['MeanShift'] = labels_rocks2

    # Projecting the well log features into 2d projection using t-SNE
    #fig4=px.scatter(df_scaled_embedded,x=0, y=1, color=df.MeanShift, labels={'color': 'GMM'})
    #fig4.update_layout(showlegend=False, height=800, width=800,title_text='t-SNE 2D Projection')

    # Plot!
    #st.plotly_chart(fig4, use_container_width=True)
    ms_t_SNE_image=Image.open('t-SNE_Meanshift.png')
    st.image(ms_t_SNE_image, width=None)
    st.write('**Note:** t-SNE plot for Mean Shift is not calculated on the fly as it is computationally expensive. Thus, only the results are presented.')
      



if Select_Method!='Mean Shift':
    plot_KG(Select_Method, Select_cluster)
else:
    plot_MS(Select_Method)

########################################################################################################################################################
#Plotting the results
st.title('Displaying Results for {}'.format(Select_Method))


def plot_res(data):

    if data=='KMeans':
        df1=df.copy()
        df1=df1.sort_values(by='KMean')
        df1['Formation']='Formation '+(df1['KMean']+1).astype(str)

        # Plotting KMeans results
        fig=px.scatter(df1,x='X', y='Y', color='Formation')
        fig.update_layout(showlegend=True, height=800, width=800,title_text='KMean Results')
        fig.update_xaxes(title_text='Attribute (X)')
        fig.update_yaxes(title_text='Attribute (Y)')

        # Plot!
        st.plotly_chart(fig, use_container_width=True)
    if data=='GMM (Gaussian Mixture Modelling)':
        df2=df.copy()
        df2=df2.sort_values(by='GMM')
        df2['Formation']='Formation '+(df2['GMM']+1).astype(str)

        # Plotting KMeans results
        fig=px.scatter(df2,x='X', y='Y', color='Formation')
        fig.update_layout(showlegend=True, height=800, width=800,title_text='GMM Results')
        fig.update_xaxes(title_text='Attribute (X)')
        fig.update_yaxes(title_text='Attribute (Y)')

        # Plot!
        st.plotly_chart(fig, use_container_width=True)
    elif data=='Mean Shift':
        #df3=df.copy()
        #df3=df.sort_values(by='MeanShift')
        #df3['Formation']='Formation '+(df3['MeanShift']+1).astype(str)

        # Plotting KMeans results
        #fig=px.scatter(df3,x='X', y='Y', color='Formation')
        #fig.update_layout(showlegend=True, height=800, width=1000,title_text='Mean Shift Results')
        #fig.update_xaxes(title_text='Attribute (X)')
        #fig.update_yaxes(title_text='Attribute (Y)')

        # Plot!
          #st.plotly_chart(fig, use_container_width=True)
        # Plot!
        ms_image=Image.open('Meanshift.png')
        st.image(ms_image, width=None)
        st.write('**Note:** Result plot for Mean Shift is not calculated on the fly as it is computationally expensive. Thus, only the results are presented.')
      

plot_res(Select_Method)

#Conclusions and Recommendations
st.title('Conclusions')
st.write('**1.** The data was relatively clean with some duplicates and some missing values for Y attribute. The data was cleaned for the analysis.')
st.write('**2.** K-Means clustering works great if the data clusters are circular, however, geological situations data rarely forms nice circular patterns. GMM modelling uses elliptical shaped cluster/decision boundaries and is, therefore, more flexible providing better clustering results.')
st.write('**3.** The Mean Shift model predicted 5 formations, but the results are not convincing when plotted.')
