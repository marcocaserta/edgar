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

 Profiling algorithm on Press Releases.

 Author: Marco Caserta (marco dot caserta at ie dot edu)
 Started : 17.04.2018
 Ended   :

 Command line options (see parseCommandLine):
-i inputfile

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
import seaborn as sns
sns.set_style("darkgrid")

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

prefix        = path.expanduser("~/research/nlp/")
vocab_folder  = "data/google_vocab/"
perDayFolders = "data/v5/"
task          = -1
nCores        =  4
nTot          =  100


def summary(task):
    """
    Print out some summmary statistics.
    """

    print("\n\n")
    print("*"*80)
    print("*\t marco caserta (c) 2018 {0:>48s}".format("*"))
    print("*"*80)
    if task == "0":
        msg = "Profiling within-year"
    elif task == "1":
        msg = "Profiling between-year"
    print(" Task type   :: {0:>60s} {1:>3s}".format(msg,"*"))
    print("*"*80)
    print("\n\n")

def parseCommandLine(argv):
    """
    Parse command line.
    """

    global task
    
    try:
        opts, args = getopt.getopt(argv, "ht:", ["help","task="])
    except getopt.GetoptError:
        print ("Usage : python clustering.py -t <task> ")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print ("Usage : python clustering.py -t <task> ")
            sys.exit()
        elif opt in ("-t", "--task"):
            task = arg

    if task == -1:
        print("Error : Task type not defined. Select one using -t : ")
        print("\t 0 :  Within-year Profiling")
        print("\t 1 :  Between-year Profiling")
        sys.exit(2)


def createMaps():
    nRows = sum(1 for line in open('mapping.txt'))
    dfMap = pd.DataFrame(index=np.arange(0, nRows), columns=('name', 'cik', 'year', 'month','day','tag') )

    dfMap = pd.read_csv("mapping.txt")

    print("... dfMap created ")
    dfMap["year"] = dfMap["year"].astype("int")
    dfMap["month"] = dfMap["month"].astype("int")
    dfMap["day"] = dfMap["day"].astype("int")
    dfMap["tag"] = dfMap["tag"].astype("int")
    print("... creating of file2tag")
    file2tag = {f:t for f,t in zip(dfMap["name"],dfMap["tag"])}
    print("... creating of tag2file")
    tag2file = {t:f for t,f in zip(dfMap["tag"],dfMap["name"])}


    return dfMap, file2tag, tag2file


def buildDistanceMatrix(year):
    """
    Here, we upload a doc2vec model for a specific year and use it to build a
    distance matrix. In addition, to separate model creation from model use, we
    need to find out which documents belong to the corpus.

    We can get the tag of each document in the current corpus, and directly
    obtain the embedding of that document using modelDoc2Vec.docvecs[tag].

    Let us first get a list of documents tag to be used to compute the distance
    matrix. Which documents should be included here depends on the strategy
    used. For now, assume we want to use all the documents of a given year.

    Note: In this case, we do not even need to load the corpus. We just need to
    create a list of tags corresponding to the documents we want to use in the
    distance matrix computation.
    """

    dfMap, file2tag, tag2file = createMaps()

    dirFull = path.join(prefix,perDayFolders) + year
    fullname = dirFull + "/doc2vec.model." + year
    print("Loading model ", fullname)
    modelDoc2Vec = doc2vec.Doc2Vec.load(fullname)
    print("The model contains {0} vectors.".format(len(modelDoc2Vec.docvecs)))
    
    # get absolute tag for each document in the current time window
    tagList = []
    nRows = len(dfMap)
    for row in range(nRows):
        #print(dfMap.iloc[row])
        if dfMap.iloc[row]["year"] == int(year):
            #print("file ", dfMap.iloc[row]["name"], " selected")
            tagList.append(dfMap.iloc[row]["tag"])
    print(tagList)

    # compute and store distance matrix
    nDocs = len(tagList)
    nameMatrix = "doc2vecDistMatrix.txt." + str(year)
    fullname = path.join(dirFull,nameMatrix)

    f = open(fullname, "w")
    writer = csv.writer(f)
    writer.writerow([nDocs])
    writer.writerow(tagList)
    vals = [ [0.0 for i in range(nDocs)] for j in range(nDocs)]
        
    for i in range(nDocs):
        tag_i = tagList[i]
            
        for j in range(i,nDocs):
            tag_j = tagList[j]
            val = round(1.0-modelDoc2Vec.docvecs.similarity(str(tag_i),str(tag_j)), 4)
            writer.writerow([tag_i, tag_j, val])

    f.close()
    print("Distance Matrix saved :: ", fullname)

def loadDoc2VecModel(year):

    dirFull = path.join(prefix,perDayFolders,year)
    fullname = dirFull + "/doc2vec.model." + year
    print("Loading model ", fullname)
    modelDoc2Vec = doc2vec.Doc2Vec.load(fullname)
    print("The model contains {0} vectors.".format(len(modelDoc2Vec.docvecs)))

    return modelDoc2Vec

def buildProfileWithin():

    
    model = loadDoc2VecModel(str(2015))
    dfMap, file2tag, tag2file = createMaps()
    dTop = dfMap.groupby(["cik"]).size().sort_values(ascending=False).head(n=nTot)
    ciks = dTop.keys()
    
    avgs  = []
    years = []
    tots  = []
    names = []


    for comp in ciks:
        print("Profile for Company ", comp)
        rows = dfMap.groupby(["cik"]).get_group(comp)

        # over the years
        yearList = rows.groupby(["year"]).size().keys()

        for year in yearList:
            vecs    = []
            tagList = []
            #  model = loadDoc2VecModel(str(year))
            rowsT = rows.groupby(["year"]).get_group(year)
            if rowsT.shape[0] > 1:
                for tag, year in zip(rowsT["tag"], rowsT["year"]):
                    #  print("tag = ", tag, " and year = ", year)

                    vv = model.docvecs[str(tag)]
                    vecs.append(vv)
                    tagList.append(tag)

                # compute distance for the current year
                dd = np.mean(ssd.pdist(vecs, metric="cosine"))
                avgs.append(dd)
                years.append(year)
                tots.append(rowsT.shape[0])
                names.append(comp)


    df = pd.DataFrame({
        "cik": names,
        "year": years,
        "avg" : avgs,
        "tot" : tots
    })
    #  print(df)
    #  print(df.dtypes)

    return df

def plotProfile(df):

    
    minX  = df["year"].min()
    maxX  = df["year"].max()
    maxY  = df["avg"].max()
    minY  = 0.0
    nTot  = 6
    ncols = 3
    nrows = int(np.ceil(nTot / ncols))
    f, axarr = plt.subplots(nrows,ncols)
    ix = 0
    jx = 0

    ciks = df.groupby(["cik"]).size().keys()

    for comp in ciks:
        dfSub = df.groupby(["cik"]).get_group(comp)
        
        print("Setting subplot (", ix, ",",jx,")")

        axarr[ix,jx].axhline(y=np.mean(dfSub["avg"]), linestyle=":", color="red",alpha=0.25)
        axarr[ix,jx].plot(dfSub["year"], dfSub["avg"], linestyle="-", marker="o", color="b",
        markersize=3, linewidth=1)
        fullTitle = "cik = " + str(comp) + str("  ($\sigma= $") + str(round(np.std(dfSub["avg"]),3)) + ")"
        axarr[ix,jx].set_title(fullTitle,fontsize=8)
        axarr[ix,jx].tick_params(axis = 'both', which = 'major', labelsize = 6)
        msg = "n = " + str(sum(dfSub["tot"]))
        axarr[ix,jx].annotate(msg, xy=((minX+maxX)/2,0.01),size=6)
        if (jx+1) % ncols == 0:
            jx  = 0
            ix += 1
        else:
            jx += 1

    xx = ["95","97","99","01","03","05","07","09","11","13","15"]
    xl = np.arange(1995,2016,2)
    yy = np.arange(minY,maxY+0.05,0.05)
    plt.setp(axarr,xticks=xl, xticklabels=xx, yticks=yy)

    f.subplots_adjust(hspace=0.3)
    namefile = "chart_type" + task + ".png"
    plt.savefig(namefile)

def buildProfileBetween():

    model = loadDoc2VecModel(str(2015))
    dfMap, file2tag, tag2file = createMaps()
    dTop = dfMap.groupby(["cik"]).size().sort_values(ascending=False).head(n=nTot)
    ciks = dTop.keys()

    avgs  = []
    years = []
    tots  = []
    names = []

    for comp in ciks:
        print("Profile for company ", comp)
        rows = dfMap.groupby(["cik"]).get_group(comp)
        yearList = rows.groupby(["year"]).size().keys()

        for t in range(len(yearList)-1):
            year0 = yearList[t]
            year1 = yearList[t+1]
            vecs0 = []
            vecs1 = []
            rows0 = rows.groupby(["year"]).get_group(year0)
            rows1 = rows.groupby(["year"]).get_group(year1)
            #  print("-"*80)
            #  print(rows0)
            #  print("-"*80)
            #  print(rows1)
            #  print("-"*80)

            if rows0.shape[0] > 1 or rows1.shape[0] > 1:
                for tag, year in zip(rows0["tag"], rows0["year"]):
                    vv = model.docvecs[str(tag)]
                    vecs0.append(vv)
                for tag, year in zip(rows1["tag"], rows1["year"]):
                    vv = model.docvecs[str(tag)]
                    vecs1.append(vv)

                #  temp = ssd.cdist(vecs0,vecs1, metric="cosine")
                #  print("Len temp = ", len(temp))
                #  print(temp)

                dd = np.mean(ssd.cdist(vecs0,vecs1, metric="cosine"))
                avgs.append(dd)
                years.append(year1)
                tots.append(rows0.shape[0]*rows1.shape[0])
                names.append(comp)

    df = pd.DataFrame({
        "cik": names,
        "year": years,
        "avg" : avgs,
        "tot" : tots
    })
    #  print(df)
    #  print(df.dtypes)

    return df



def main(argv):
    '''
    Entry point. Five types of tasks, controlled via command line.
    '''
    parseCommandLine(argv)
    summary(task)

    if task == "0":
        df = buildProfileWithin()
    elif task == "1":
        df = buildProfileBetween()

    df.to_csv("profile.csv")

    plotProfile(df)




if __name__ == '__main__':

    #  logging.basicConfig(
    #      format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
    #      level=logging.INFO
    #  )

    main(sys.argv[1:])
