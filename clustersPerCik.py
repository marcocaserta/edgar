"""
/***************************************************************************
 *   copyright (C) 2018 by Marco Caserta                                   *
 *   marco dot caserta at ie dot edu                                       *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

 Clustering algorithm on Press Releases.

 Author: Marco Caserta (marco dot caserta at ie dot edu)
 Started : 05.04.2018
 Ended   : 10.04.2018

 Command line options (see parseCommandLine)

NOTE: This code is based on the tutorial from:

https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb

"""

from multiprocessing import Pool
import csv
import sys, getopt
import os
from os import path, listdir
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import logging
from collections import namedtuple
import glob

from gensim.models import doc2vec

from sklearn import manifold
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import fcluster

from nltk.corpus import stopwords
from nltk import word_tokenize
stop_words = stopwords.words("english")


prefix = path.expanduser("~/research/nlp/")
vocab_folder       = "data/google_vocab/"
perDayFolders = "data/v5/"
namefile    = "clusters/mapping."
task        = -1
clusterType = -1
nCores      =  4
yearTarget  = "-1"
nTot = 1
maxK = 25


def summary(task, clusterType, yearTarget):
    """
    Print out some summmary statistics.
    """

    print("\n\n")
    print("*"*80)
    print("*\t marco caserta (c) 2018 {0:>48s}".format("*"))
    print("*"*80)
    if task == "0":
        msg = "Folder Restructuring"
    elif task == "1":
        msg = "Preprocessing"
    elif task == "2":
        msg = "Doc2Vec Creation"
    elif task == "3":
        msg = "Distance Matrix"
    elif task == "4":
        msg = "Clustering (Type " + str(clusterType) + ")"
    print(" Task type   :: {0:>60s} {1:>3s}".format(msg,"*"))
    print(" Target year :: {0:>60s} {1:>3s}".format(yearTarget,"*"))
    print("*"*80)
    print("\n\n")

def parseCommandLine(argv):
    """
    Parse command line.
    """

    global task
    global clusterType
    global yearTarget
    
    try:
        opts, args = getopt.getopt(argv, "ht:c:y:", ["help","task=",
        "clusterType=", "year="])
    except getopt.GetoptError:
        print ("Usage : python clustering.py -t <task> -c <clustering> -y <year>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print ("Usage : python doc2Vec.py -t <type> -i <inputfile> ")
            sys.exit()
        elif opt in ("-t", "--task"):
            task = arg
        elif opt in ("-c", "--clusterType"):
            clusterType = arg
        elif opt in ("-y", "--year"):
            yearTarget = arg

    if task == -1:
        print("Error : Task type not defined. Select one using -t : ")
        print("\t 0 :  Folder Restructuring")
        print("\t 1 :  Preprocessing")
        print("\t 2 :  Doc2Vec Creation")
        print("\t 3 :  Distance Matrix")
        print("\t 4 :  Clustering (define type using -c)")
        sys.exit(2)

    if yearTarget == "-1":
        print("Error : Target year not defined. Use -y ")
        sys.exit(2)

def createMaps():
    nRows = sum(1 for line in open('mapping.txt'))
    dfMap = pd.DataFrame(index=np.arange(0, nRows), columns=('name', 'cik',
    'year', 'month','day','nr','tag') )

    dfMap = pd.read_csv("mapping.txt")

    print("... dfMap created ")
    dfMap["year"] = dfMap["year"].astype("int")
    dfMap["month"] = dfMap["month"].astype("int")
    dfMap["day"] = dfMap["day"].astype("int")
    dfMap["nr"] = dfMap["nr"].astype("int")
    dfMap["tag"] = dfMap["tag"].astype("int")
    print("... creating of file2tag")
    file2tag = {f:t for f,t in zip(dfMap["name"],dfMap["tag"])}
    print("... creating of tag2file")
    tag2file = {t:f for t,f in zip(dfMap["tag"],dfMap["name"])}


    return dfMap, file2tag, tag2file

def hierarchicalClustering(distanceMatrix, withDendrogram=False):
    """ 
    Use the linkage function of scipy.cluster.hierarchy.

    Parameters:
    - withDendrogram : True if the dendrogram should be produced and saved on disk

    We try out different methods for hierarchical clustering, based on
    different ways of computing distances between clusters. We select the one 
    that maximizes the Cophenetic Correlation Coefficient.

    Returns:
    - labels: The clusters labels
    """

    # convert symmetric distance matrix into upper triangular array
    distArray = ssd.squareform(np.asmatrix(distanceMatrix), checks=False)
    # find "best" method
    methods    = ["ward", "median", "average", "single", "complete"]
    bestVal    = 0.0
    bestMethod = " "
    for mm in methods:
        #  Z = linkage(distArray, method=mm, optimal_ordering=True)
        Z = linkage(distArray, method=mm)

        # test the goodness of cluster with cophenetic correl coefficient
        c, cophDist = cophenet(Z, distArray)
        print("[ {0:10s} ] Cophenetic = {1:5.2f}".format(mm, c))
        if c > bestVal:
            bestVal    = c
            bestMethod = mm

    # repeat with best method
    Z = linkage(distArray, method=bestMethod)
    #  Z = linkage(distArray, method=bestMethod, optimal_ordering=True)
    print(Z)
    # note: The Z gives the distances at which each cluster was merged

    # get the cluster for each point
    #  maxD   = 0.95
    maxD   = 0.5
    labels = fcluster(Z, maxD, criterion="distance")
    labels = labels - [1]*len(labels)  #  start from 0

    if withDendrogram:
        plt.figure(figsize=(25, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        dendrogram(
            Z,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
            show_leaf_counts=True,
            get_leaves=True,
            #  truncate_mode="level",
            #  p =5,
        )
        plt.axhline(y=maxD, c='k')
        plt.savefig("dendrogram.png")
        print("Dendrogram saved on disk ('dendrogam.png')")

    return labels

def loadDoc2VecModel(year):

    dirFull = path.join(prefix,perDayFolders,year)
    fullname = dirFull + "/doc2vec.model." + year
    print("Loading model ", fullname)
    modelDoc2Vec = doc2vec.Doc2Vec.load(fullname)
    print("The model contains {0} vectors.".format(len(modelDoc2Vec.docvecs)))

    return modelDoc2Vec

def nltkClustering(vectors, nrClusters):
    """
    Use the clustering function of nltk to define my own distance function.
    """

    import nltk
    from nltk.cluster import KMeansClusterer
    from sklearn import metrics

    num_clusters = nrClusters
    kclusterer = KMeansClusterer(num_clusters, 
        distance = nltk.cluster.util.cosine_distance,
        #  distance = nltk.cluster.util.euclidean_distance,
        repeats = 50)
    labels = kclusterer.cluster(vectors, assign_clusters=True)
    score = metrics.silhouette_score(vectors, labels, metric="cosine")
    means = kclusterer.means()
    print("Silhouette score (nc = ", nrClusters, " ) = ", score )

    return labels, score, means

def clusterPerCik():

    model = loadDoc2VecModel(str(2015))
    dfMap, file2tag, tag2file = createMaps()
    dTop = dfMap.groupby(["cik"]).size().sort_values(ascending=False).head(n=nTot)
    ciks = dTop.keys()

    for comp in ciks:
        vecs = []
        tagList = []
        print("Working with company ", comp)
        rows = dfMap.groupby(["cik"]).get_group(comp)

        count  = 0
        for tag, year in zip(rows["tag"], rows["year"]):
            count += 1
            print("tag  =", tag, " and year = ", year)
            vv = model.docvecs[str(tag)]
            vecs.append(vv)
            tagList.append(tag)
            if count == 10:
                break

        # compute distance matrix for this company
        nDocs = len(tagList)
        distMatrix = [ [0.0 for i in range(nDocs)] for j in range(nDocs)]
        print("computing distance matrix ")
        for i in range(nDocs):
            tag_i = tagList[i]
            for j in range(i, nDocs):
                tag_j = tagList[j]
                distMatrix[i][j] = round(1.0-model.docvecs.similarity(str(tag_i),str(tag_j)), 4)
                distMatrix[j][i] = distMatrix[i][j]

        print("done with distance matrix ")

    return tagList, distMatrix

def clusterPerCikNLTK(rows, comp, model):


    vecs = []
    tagList = []
    print("Working with company ", comp)

    #  tagList = rows["tag"]
    #  vect = [model.docvecs[str(i) for i in tagList]
    count  = 0
    for tag, year in zip(rows["tag"], rows["year"]):
        count += 1
        #  print("tag  =", tag, " and year = ", year)
        vv = model.docvecs[str(tag)]
        vecs.append(vv)
        tagList.append(tag)
        #  if count == 20:
            #  break

    return tagList, vecs

def computeVariation(dfCik, means, label, tagList, vecs, comp):
    """
    For each document of the given company, compute the distance from the
    center of the cluster. Store the cluster a document belongs to and the
    distance from center in a dataframe. Use cik to identify the namefile.
    """

    #  dfCik = dfCik.head(20)
    dists = []
    for tag, lab, vv in zip(tagList,label, vecs):
        #  print("Doc tag = ", tag, " with cluster = ", lab)
        meanVec = means[lab]
        dd = ssd.cosine(vv, meanVec)
        dists.append(dd)


    dfCik["cluster"] = pd.Series(label, index=dfCik.index)
    dfCik["vars"] = pd.Series(dists, index=dfCik.index)
    #  print(dfCik.head())

    namedf = namefile + str(comp) + ".csv"
    dfCik.to_csv(namedf)
    print("Mapping written on disk - file '{0}'".format(namedf))


def main(argv):
    '''
    This code is used to create clusters at cik level (clusters for all the
    documents belonging to a given company over the entire time horizon) and to
    compute the average distance of each document with its cluster center.

    First identify a good number of clusters with the cycle. Once an estimate
    has been obtained, store the information (cluster a document belongs to and
    distance with the cluster center) in a file. The file uses the company cik
    as unique identifier.

    Note: We are using the doc2vec model of 2015, i.e., the most complete one.
    However, we could easily modify it to use the doc2vec model of each year,
    depending on when the document has been released.
    '''
    #  parseCommandLine(argv)
    #  summary(task, clusterType, yearTarget)

    # activate this part to use hierarchical clustering
    #  tagList, distanceMatrix = clusterPerCik()
    #  hierarchicalClustering(distanceMatrix, withDendrogram=True)

    model = loadDoc2VecModel(str(2015))
    dfMap, file2tag, tag2file = createMaps()

    dTop = dfMap.groupby(["cik"]).size().sort_values(ascending=False).head(n=nTot)
    ciks = dTop.keys()

    # set the cik of the company we want to study
    for comp in ciks:

        dfCik = dfMap.groupby(["cik"]).get_group(comp)
        tagList, vecs = clusterPerCikNLTK(dfCik, comp, model)
        silhouette_score = []
        for nc in range(3,maxK):
            label, score, means = nltkClustering(vecs, nrClusters = nc)
            silhouette_score.append(score)

        # save to file selection of k*
        df = pd.DataFrame({
            "k" : range(3,maxK),
            "silhouette": silhouette_score
        })
        nameff = "clusters/silhouette." + str(comp) + ".csv"
        df.to_csv(nameff)

        computeVariation(dfCik, means, label, tagList, vecs, comp)

    


if __name__ == '__main__':

    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO
    )

    main(sys.argv[1:])
