#Importing Libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn import  preprocessing
import matplotlib.pyplot as plt
import pylab as pl
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import math
from PIL import Image
import seaborn as sns

st.set_page_config(page_title='PaleoApp, Machine-Learning for Paleontologists',
    layout='wide')

st.title("FossilUp - Educational Machine Learning app for Paleontologists")

image = Image.open('jurassic.jpg')
st.image(image, width = 400)

header = st.container()
with header:
    st.text("""
    Developed by Angelos Tamvakis
    """)

st.sidebar.header("1. Select an example dataset")
dataset_name = st.sidebar.selectbox("Choose a dataset:", ("Iris", "Import your own"))
with st.sidebar.header('2. Upload you own dataset'):
    uploaded_file = st.sidebar.file_uploader('Upload your input CSV file:', ["csv"])

@st.cache
def load_data(filename):
    if filename == "Iris":
        data = datasets.load_iris()
        df = pd.DataFrame(data.data)
        df.columns = data.feature_names
        df["target"] = data.target
        df['class'] = ''
        for i in range(0, df.shape[0]):
            if df.loc[i, 'target'] == 0:
                df.loc[i, 'class'] = 'setosa'
            elif df.loc[i, 'target'] == 1:
                df.loc[i, 'class'] = 'versicolor'
            elif df.loc[i, 'target'] == 2:
                df.loc[i, 'class'] = 'virginica'
        df.drop(['target'], axis = 1, inplace = True)
        X = data.data
        y = data.target
    elif filename == "Import your own":
        df = pd.read_csv(uploaded_file)
        X = df.select_dtypes(include='number')
        y = df.select_dtypes(include='object')
    return df,X,y

df,X,y = load_data(dataset_name)        

st.sidebar.info('* **Uploaded Dataframe must have target column in the last position**')

if st.checkbox('Show the dataset as table data'):
    st.dataframe(df)


scal_opt = st.sidebar.selectbox("Select a scaler", ('Standard scaler', 'Minmax scaler', 'None'))

st.sidebar.info("Choose Scaling method")

if scal_opt == 'Standard scaler':
    scaler = StandardScaler()
    sc_x =  scaler.fit_transform(df.select_dtypes(include='number'))
elif scal_opt == 'Minmax scaler':
    scaler = MinMaxScaler()
    sc_x = scaler.fit_transform(df.select_dtypes(include='number'))
elif scal_opt == 'None':
    scaler = None
    sc_x = df.select_dtypes(include='number')

target_col = st.sidebar.selectbox("Select target column", df.select_dtypes(include='object').columns)
scaled_df = pd.DataFrame(sc_x, columns = df.select_dtypes(include='number').columns.values)
scaled_df['target'] = df[target_col]

st.write("Dataframe number of features and rows:", df.shape)
st.write("Classes:", df[target_col].unique(), len(df[target_col].unique()))

if st.checkbox("Show scaled data"):
    st.dataframe(scaled_df)

pca_header = st.container()

with pca_header:
    st.header("3. Dimensionality Reduction with PCA and t-SNE")
    st.text("This is an example of PCA and KPCA/t-SNE analysis")
    st.sidebar.header('3.1 PCA User Input')
    num_pca = st.sidebar.number_input("Select number of components for PCA/KPCA", value = 2, step = 1, min_value = 2, max_value = scaled_df.select_dtypes(include='number').shape[1])
    pca = PCA(n_components= num_pca)
  
    pca_fit = pca.fit_transform(scaled_df.select_dtypes(include='number'))

    pca_col = ["PC"+str(i) for i in list(range(1, num_pca+1))]

    pca_plot = pd.DataFrame(data = pca_fit, columns = pca_col[:num_pca])
    pca_plot['class'] = df[target_col]
    
    features = pca_plot['class'].unique()
    
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    loadings_df = pd.DataFrame(loadings[:, :num_pca], columns = pca_col)
    loadings_df.set_index(df.select_dtypes(include='number').columns.values, inplace = True) 


    fig = px.scatter(pca_plot, x= 'PC1', y= 'PC2', color= pca_plot['class'], title = 'PCA plot', labels = {'0': 'PC1', '1' : 'PC2'})

    for i, feature in enumerate(features):
        fig.add_shape(
            type='line',
            x0=0, y0=0,
            x1=loadings[i, 0],
            y1=loadings[i, 1]
        )
        fig.add_annotation(
        x=loadings[i, 0],
        y=loadings[i, 1],
        ax=0, ay=0,
        xanchor="center",
        yanchor="bottom",
        text=feature,
        )

    col_fig, col_load = st.columns(2)
    col_fig.subheader("PCA plot")
    col_fig.write(fig)
    col_load.subheader("PCA Loadings")
    col_load.write(loadings_df)

    #KPCA
    kpca_kernel = st.sidebar.selectbox('Select kernel for KPCA',  ("linear", 'poly', 'rbf', 'sigmoid', 'cosine'))
    gamma_kpca = st.sidebar.number_input('Select gamma value for KPCA', value = 0, step = 1)
    kpca = KernelPCA(n_components = num_pca, kernel = kpca_kernel, gamma = gamma_kpca)
    kpca_res = kpca.fit_transform(scaled_df.select_dtypes(include='number'))

    kpca_col = ["PC"+str(i) for i in list(range(1, num_pca+1))]

    kpca_plot = pd.DataFrame(data = kpca_res, columns = kpca_col[:num_pca])
    kpca_plot['class'] = df[target_col]

    kpca_fig = px.scatter(kpca_plot, x= 'PC1', y= 'PC2', color= kpca_plot['class'], title = 'KPCA plot', labels = {'0': 'PC1', '1' : 'PC2'})

    st.write(kpca_fig)

    if st.checkbox("Show PCA result"):
        st.dataframe(pca_plot)

    st.sidebar.header("3.2 t-SNE User Input")
    st.text("t-SNE Example")

    tsne_num_comp = st.sidebar.number_input("Select components", value = 2, step = 1)
    tsne_perp = st.sidebar.number_input("Select perplexity value", value = 10, step = 1)
    tsne_early_ex = st.sidebar.number_input("Select early exaggeration", value = 12, step = 1)
    tsne_rate = st.sidebar.number_input("Select learning rate", value = 200, step = 1)
    tsne_num_iter = st.sidebar.number_input("Select iterations number", value = 1000, step = 1)
    tsne = TSNE(n_components = tsne_num_comp, perplexity = tsne_perp, early_exaggeration= tsne_early_ex, learning_rate = tsne_rate, n_iter = tsne_num_iter, verbose= 3)

    tsne_res = tsne.fit_transform(scaled_df.select_dtypes(include='number'))
    tsne_df = pd.DataFrame(tsne_res, columns = ['t-SNE1', 't-SNE2'])
    tsne_df['class'] = df[target_col]

    tsne_fig = px.scatter(tsne_df, x='t-SNE1', y='t-SNE2', color= tsne_df['class'], title = 't-SNE plot', labels = {'0': 't-sNE1', '1' : 't-SNE2'})

    st.write(tsne_fig)

classification = st.container()

with classification:
    st.header("4. Classification/Regression using SVMs and KNN models")

    st.sidebar.header("4. SVM\KNN User Input")

    task_info = st.sidebar.selectbox('Select whether it is classification or regression task', ('Classification', 'Regression'))

    if task_info == 'Classification':
        target_sel = st.sidebar.selectbox("Select target column to be encoded", df.select_dtypes(include='object').columns)
        if target_sel != target_col:
            st.sidebar.info("Please select same target column as previously")
        target_var = df[target_sel]
        l_enc = preprocessing.LabelEncoder()
        target_var = l_enc.fit_transform(target_var)
    else:
        target_sel = st.sidebar.selectbox("Select target variable", df.columns)
        target_var = df[target_sel]

    if task_info == 'Regression' and scal_opt == 'Standard scaler':
        target_scaler = StandardScaler()
        target_var = target_scaler.fit_transform(target_var.values.reshape(-1, 1))
    elif task_info == 'Regression' and scal_opt == 'Minmax scaler':
        target_scaler = MinMaxScaler()
        target_var = target_scaler.fit_transform(target_var.values.reshape(-1, 1))

    class_df = pd.DataFrame(sc_x, columns = df.select_dtypes(include='number').columns)

    if task_info == 'Classification':
        class_df['target'] = target_var

    if st.checkbox("Click to show hints!"):
        st.markdown("* To perform **classification** firstly the target variables (e.g. for Iris dataset, species) need to be encoded into numerical values.")
        st.markdown("* To do so, the **LabelEncoder** will be used")

    if st.checkbox('Click to show processed data'):
        st.dataframe(class_df)

    clf_select = st.sidebar.selectbox("Select classifier", ('SVM', 'KNN'))

    if clf_select == "KNN":
        if task_info == 'Classification':
            clf = KNeighborsClassifier(n_neighbors = st.sidebar.number_input("Select number of K-Nearest neighbors to be used", value = 3, min_value = 1, step = 1))
        if task_info == 'Regression':
            clf = KNeighborsRegressor(n_neighbors = st.sidebar.number_input("Select number of K-Nearest neighbors to be used", value = 3, min_value = 1, step = 1))
    elif clf_select == "SVM":
        if task_info == 'Classification':
            clf = SVC(C = st.sidebar.number_input("Select C", value =0.1, step = 0.1), gamma= st.sidebar.number_input("Select gamma", value =0.1, step = 0.1), kernel = st.sidebar.selectbox("Select kernel", ("linear", "poly", "rbf")), probability = True)
        if task_info == 'Regression':
            clf = SVR(C = st.sidebar.number_input("Select C", value =0.1, step = 0.1), gamma= st.sidebar.number_input("Select gamma", value =0.1, step = 0.1), kernel = st.sidebar.selectbox("Select kernel", ("linear", "poly", "rbf")), epsilon = st.sidebar.number_input("Select epsilon", value =0.1, step = 0.1))

    split_prc = st.sidebar.slider('Data split ratio (% for Training Set)', min_value = 10, max_value = 50, value = 20)

    if st.checkbox("Click for hints!"):
        st.markdown("* For the classification data must be splitted into: **Training** and **Test** sets")
        st.markdown("* The model is **trained** on the **training set** and then its perfomance is **evaluated** on the **test set**")

    x = class_df.drop(['target'], axis = 1)
    y = class_df['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = (split_prc /100))

    st.markdown("**4.1 Train/Test sets**")
    st.write('Training set: ', x_train.shape)
    st.write('Test set: ', x_test.shape)

    clf.fit(x_train, y_train)

    pred = clf.predict(x_test)

    if task_info == 'Classification':
        acc = clf.score(x_test, y_test)
        st.write('Model accuracy', acc)
        probs = clf.predict_proba(x_test)
        probs = pd.DataFrame(probs, columns = l_enc.inverse_transform(class_df['target'].unique()))

        if st.checkbox('Show predicted probabilities of assigned class:'):
            st.dataframe(probs)

    elif task_info == 'Regression':
        pred = pd.DataFrame(pred, columns = ['predicted_value'])
        if scaler != None:
            pred_inv = target_scaler.inverse_transform(pred)
            test_inv = target_scaler.inverse_transform(y_test.values.reshape(-1, 1))
            col1, col2 = st.columns(2)
            col1.subheader('Actual values')
            col1.write(test_inv)
            col2.subheader('Predicted values')
            col2.write(pred_inv)
            
            MSE = mean_squared_error(test_inv, pred_inv)
            RMSE = math.sqrt(MSE)
            st.write("Root Mean Square Error: ", RMSE)
        else:
            MSE = mean_squared_error(y_test, pred)
            RMSE = math.sqrt(MSE)
            st.write("Root Mean Square Error: ", RMSE)
    
    if task_info == 'Classification':
        if st.checkbox("Show decision boundary plot"):
            x_plot = pca_plot.iloc[:, :2]
            y_plot = class_df['target']

            def make_meshgrid(x, y, h=.02):
                x_min, x_max = x.min() - 1, x.max() + 1
                y_min, y_max = y.min() - 1, y.max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                return xx, yy

            def plot_contours(ax, clf, xx, yy, **params):
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                out = ax.contourf(xx, yy, Z, **params)
                return out

            clf_plot = clf.fit(x_plot, y_plot)
            
            clf_fig, ax = plt.subplots(figsize = (7,5))
            # title for the plots
            plot_title = ('Decision surface plot')
            # Set-up grid for plotting.
            X0, X1 = x_plot.iloc[:, 0], x_plot.iloc[:, 1]
            xx, yy = make_meshgrid(X0, X1)

            plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
            ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(plot_title)
            ax.legend()
            plt.show()

            st.write(clf_fig)

    
clustering = st.container()

with clustering:
    st.header("4. Clustering using KMeans and Spectral clustering algorithms")
 

    df_to_show = ['Initial Data', 'Scaled Data']
    data_to_show = st.radio('Show initial data', df_to_show)

    if st.checkbox('Show Data'):
        if data_to_show == 'Initial Data':
            st.dataframe(df)
        else:
            st.dataframe(scaled_df)
    
    st.sidebar.header('5.1 KMeans User Input')
    num_clusters = st.sidebar.number_input("Select number of clusters", value = 2, step = 1)
    num_init = st.sidebar.number_input("Select initializations", value = 10, step = 1)
    kmeans = KMeans(n_clusters = num_clusters, n_init = num_init)
    kmeans.fit(scaled_df.select_dtypes(include='number'))
    pred_clusters = kmeans.predict(scaled_df.select_dtypes(include='number'))
    kmeans_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    cluster_fig, cluster_ax = plt.subplots(figsize = (12,9))
    cluster_plot_title = ("KMeans Clustering plot")
    plt.scatter(pca_plot.iloc[:, 0], pca_plot.iloc[:, 1], c = pred_clusters, s = 50, cmap = 'viridis')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c = 'black', s = 200, alpha = 0.5)
    cluster_ax.set_xlabel("PC1")
    cluster_ax.set_ylabel("PC2")
    cluster_ax.set_title(cluster_plot_title)

    st.sidebar.header("5.2 Spectral Clustering User Input")
    sp_num_clusters = st.sidebar.number_input("Select num clusters", value = 2, step = 1)
    sp_num_init = st.sidebar.number_input("Select number of initializations", value = 10, step = 1)
    sp_gamma = st.sidebar.number_input("Select gamma coefficient", value = 2, step = 1)
    sp_affinity = st.sidebar.selectbox("Select affinity method", ('nearest_neighbors', 'rbf'))
    num_neigh = st.sidebar.number_input("Select number of neighbors", value = 10, step = 1)
    sp_model = SpectralClustering(n_clusters = sp_num_clusters, n_init = sp_num_init, gamma = sp_gamma, affinity = sp_affinity, n_neighbors = num_neigh)
    #sp_model.fit(scaled_df.select_dtypes(include='number'))
    sp_clusters = sp_model.fit_predict(scaled_df.select_dtypes(include='number'))
    sp_labels = sp_model.labels_

    sp_fig, sp_ax = plt.subplots(figsize = (12, 9))
    sp_plot_title = ("Spectral Clustering plot")
    plt.scatter(pca_plot.iloc[:, 0], pca_plot.iloc[:, 1], c = sp_clusters, s = 50, cmap = 'viridis')
    sp_ax.set_xlabel("PC1")
    sp_ax.set_ylabel("PC2")
    sp_ax.set_title(sp_plot_title)
    
    kmeans_col, sp_col = st.columns(2)
    kmeans_col.subheader("Kmeans plot")
    kmeans_col.write(cluster_fig)
    sp_col.subheader("Spectral Clustering plot")
    sp_col.write(sp_fig)

    

    



        


