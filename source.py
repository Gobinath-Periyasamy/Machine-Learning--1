# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 23:53:34 2022

@author: Gobinath Periyasamy
Roll No : C21M501
"""

"""
importing packages numpy, pandas, matplotlib
"""

#importing necessary libraries
import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sb
import sys


#Reading the given i/p CSV Dataset using pandas

data = pd.read_csv("Dataset.csv",header = None,prefix= 'Feature ')
# The read data is in Dataframe form
# Let us plot and see the visualization of given data

final_df_p = pd.DataFrame(data.head(1000), columns = ['Feature 0','Feature 1'])
plt.figure(figsize = (7,7))
plt.title("Given Original Dataset")
plt.grid()
plt.xlim(-10, 10)
plt.ylim(-10, 10)
sb.scatterplot(data = final_df_p , x = 'Feature 0', y = 'Feature 1', s =20, palette= 'Accent')

# Converting the data into numpy

data_a = data.to_numpy()
data_a

# Centering dataset using the mean of the Dataset Columns
data_meaned = data - np.mean(data)
data_meaned

# Finding the covariance of the matrix

covariance_matrix  = np.cov(data_meaned , rowvar = False)
covariance_matrix

#Step-3

# Finding the Eigen Values and Eigen Vectors of the Matrix 

eigen_values , eigen_vectors = np.linalg.eigh(covariance_matrix)
print(eigen_values)
print(eigen_vectors)

#Step-4
# Should be sorted in descending diagonal order
sorted_index = np.argsort(eigen_values)[::-1]
sorted_eigenvalue = eigen_values[sorted_index]
sorted_eigenvectors = eigen_vectors[:,sorted_index]

print(sorted_index)
print(sorted_eigenvalue)
print(sorted_eigenvectors)
sorted_index

#Step-5
n_components = 2
p_components_req = sorted_eigenvectors[:,0:n_components]

#Explained Variance
sum(sorted_eigenvalue[:n_components])/sum(sorted_eigenvalue)

#Step-6
projected_data = np.dot(p_components_req.transpose() , data_meaned.transpose() ).transpose()
projected_data

#Downstreamed dataset
final_df = pd.DataFrame(projected_data, columns = ['Feature 0','Feature 1'])
#final_df = pd.concat([final_df , pd.DataFrame(label , columns = ['label'])] , axis = 1)
final_df

plt.figure(figsize = (7,7))
plt.grid()
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.title('PCA plot')
sb.scatterplot(data = final_df , x = 'Feature 0', y = 'Feature 1', s = 20 , palette= 'Accent')


import math


#Explained Variance

n_ver = [np.round(sorted_eigenvalue[i]/sum(sorted_eigenvalue)*100,decimals = 1) for i in range(n_components)]

print(n_ver)
labels = ['PC' + str(x) for x in range(1,3)]
plt.bar(x =range(1,3), height = n_ver, tick_label = labels)
plt.xlim()
plt.ylim(0,100)
plt.title("Variance " + str(n_ver))
plt.ylabel('<--Variance Percent-->')
plt.xlabel('<--PC-->')
plt.show()

n_components = 1
p_components_req = sorted_eigenvectors[:,0:n_components]

 
# def PCA(X , num_components):
     
#     #Step-1
#     X_meaned = X - np.mean(X , axis = 0)
     
#     #Step-2
#     cov_mat = np.cov(X_meaned , rowvar = False)
     
#     #Step-3
#     eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
#     #Step-4
#     sorted_index = np.argsort(eigen_values)[::-1]
#     sorted_eigenvalue = eigen_values[sorted_index]
#     sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
#     #Step-5
#     eigenvector_subset = sorted_eigenvectors[:,0:num_components]
     
#     #Step-6
#     X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
     
#     return X_reduced



# # Reading the CSV file 
# data = pd.read_csv("Dataset.csv",header = None,prefix=('column_'))


# X =  data.to_numpy()
# # X_meaned = X - np.mean(X , axis = 0)

# # cov_mat = np.cov(X_meaned , rowvar = False)

# # #Calculating Eigenvalues and Eigenvectors of the covariance matrix
# # eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
# # #sort the eigenvalues in descending order
# # sorted_index = np.argsort(eigen_values)[::-1]
 
# # sorted_eigenvalue = eigen_values[sorted_index]
# # #similarly sort the eigenvectors 
# # sorted_eigenvectors = eigen_vectors[:,sorted_index]

# # # select the first n eigenvectors, n is desired dimension
# # # of our final reduced data.
 
# # n_components = 2 #you can select any number of components.
# # eigenvector_subset = sorted_eigenvectors[:,0:n_components]

# # #Transform the data 
# # X_reduced = np.dot(eigenvector_subset.transpose(),X_meaned.transpose()).transpose()


# mat_reduced = PCA(X , 2)

# #Creating a Pandas DataFrame of reduced Dataset
# principal_df = pd.DataFrame(mat_reduced , columns = ['PC1','PC2'])

# #Concat it with target variable to create a complete Dataset
# principal_df = pd.concat([principal_df] )



# plt.figure.Figure(figsize = (6,6))
# sb.scatterplot(data = principal_df , x = 'PC1' ,y= 'PC2', palette= 'icefire')

# #axs = principal_df.plot.line(figsize=(20, 2), subplots=True)



# # plt.figure.Figure(figsize = (30,4))
# # #plt.hl(0.001,0.002,1000)  # Draw a horizontal line
# # #plt.xlim(0,1)
# # #plt.ylim(0.5 ,1.5)
# # y = np.ones(np.shape(principal_df))   # Make all y values the same
# # plt.pyplot.plot(principal_df,y,'.',color ='red', ms = 5)  # Plot a line at each location specified in a
# # plt.axis('off')
# # plt.show()
