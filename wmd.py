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

Word Movers Distance (Transportation Problem) on Press Releases.

 Author: Marco Caserta (marco dot caserta at ie dot edu)
 Started : 09.05.2018
 Ended   : 09.05.2018

 This code implements a different method to assess the within-year variablity
 per company. We take all the PR of a year t, and all the PR of year t+1. Next,
 we apply the transportation problem to find how far away the PR of year t are
 from the PR of year t+1, considering all the PR of year t as sources and all
 the PR of year t+1 as destination.

 It does not seem to produce good results, probably due to the fact that the
 number of press releases in each year is large and, therefore, the
 transportation problem tends to fragment the solution a lot.


 Command line options (see parseCommandLine)

"""

import sys, getopt
import os
from os import path, listdir
import cplex
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import glob
from scipy.spatial.distance import cdist

from gensim.models import doc2vec

import scipy.spatial.distance as ssd


prefix = path.expanduser("~/research/nlp/")
vocab_folder       = "data/google_vocab/"
perDayFolders = "data/v5/"
nCores      =  4
nTop = 100

def plotProfile(df):

    
    minX  = df["year"].min()
    maxX  = df["year"].max()
    maxY  = df["avg"].max()
    minY  = df["avg"].min()
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
    namefile = "chartWmd.png"

    plt.savefig(namefile)

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


def loadDoc2VecModel(year):

    dirFull = path.join(prefix,perDayFolders,year)
    fullname = dirFull + "/doc2vec.model." + year
    print("Loading model ", fullname)
    modelDoc2Vec = doc2vec.Doc2Vec.load(fullname)
    print("The model contains {0} vectors.".format(len(modelDoc2Vec.docvecs)))

    return modelDoc2Vec

def solveTransport(matrixC, cap, dem, nS, nD):
    """
    Solve transportation problem as an LP.
    This is my implementation of the WMD.
    """
    
    cpx   = cplex.Cplex()
    x_ilo = []
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    for i in range(nS):
        x_ilo.append([])
        for j in range(nD):
            x_ilo[i].append(cpx.variables.get_num())
            varName = "x." + str(i) + "." + str(j)
            cpx.variables.add(obj   = [float(matrixC[i][j])],
                              lb    = [0.0],
                              names = [varName])
    # capacity constraint
    for i in range(nS):
        index = [x_ilo[i][j] for j in range(nD)]
        value = [1.0]*nD
        capacity_constraint = cplex.SparsePair(ind=index, val=value)
        cpx.linear_constraints.add(lin_expr = [capacity_constraint],
                                   senses   = ["L"],
                                   rhs      = [cap[i]])

    # demand constraints
    #  for j in dctTarget:
    for j in range(nD):
        index = [x_ilo[i][j] for i in range(nS)]
        value = [1.0]*nS
        demand_constraint = cplex.SparsePair(ind=index, val=value)
        cpx.linear_constraints.add(lin_expr = [demand_constraint],
                                   senses   = ["G"],
                                   rhs      = [dem[j]])
    cpx.parameters.simplex.display.set(0)
    cpx.solve()

    z = cpx.solution.get_objective_value()

    return z

def wmdTransport():


    dfMap, file2tag, tag2file = createMaps()
    dTop = dfMap.groupby(["cik"]).size().sort_values(ascending=False).head(n=nTop)
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
            S = []
            D = []
            year0 = yearList[t]
            year1 = yearList[t+1]
            if t == 0:
                model0 = loadDoc2VecModel(str(year0))
            else:
                model0 = model1
            model1 = loadDoc2VecModel(str(year1))

            rows0 = rows.groupby(["year"]).get_group(year0)
            rows1 = rows.groupby(["year"]).get_group(year1)
            if rows0.shape[0] > 1 and rows1.shape[0] > 1:
                for tag, year in zip(rows0["tag"], rows0["year"]):
                    vv = model0.docvecs[str(tag)]
                    S.append(vv)
                for tag, year in zip(rows1["tag"], rows1["year"]):
                    vv = model1.docvecs[str(tag)]
                    D.append(vv)

                nS = len(S)
                nD = len(D)

            # define transportation problem
            dd = cdist(S,D)
            cap = [1.0/nS]*nS
            dem = [1.0/nD]*nD
            
            result = solveTransport(dd, cap, dem, nS, nD)
            #  print("Result {0}-{1} is  = {2}".format(year,year+1, result))
            avgs.append(result)
            years.append(year1)
            tots.append(nS*nD)
            names.append(comp)

    df = pd.DataFrame({
        "cik": names,
        "year": years,
        "avg" : avgs,
        "tot" : tots
    })

    return df

def wmdTransport2():


    yearList = np.arange(1995,2016,1)
    dfMap, file2tag, tag2file = createMaps()
    dTop = dfMap.groupby(["cik"]).size().sort_values(ascending=False).head(n=nTop)
    ciks = dTop.keys()

    avgs  = []
    years = []
    tots  = []
    names = []

    for t in range(len(yearList)-1):
        year0 = yearList[t]
        year1 = yearList[t+1]
        print("** Years {0} and {1}".format(year0,year1))
        if t == 0:
            model0 = loadDoc2VecModel(str(year0))
        else:
            model0 = model1
        model1 = loadDoc2VecModel(str(year1))

        for comp in ciks:
            print("Profile for company ", comp)

            rows = dfMap.groupby(["cik"]).get_group(comp)
            yearsCik = rows.groupby(["year"]).size().keys()
            if year0 not in yearsCik or year1 not in yearsCik:
                continue

            S = []
            D = []

            rows0 = rows.groupby(["year"]).get_group(year0)
            rows1 = rows.groupby(["year"]).get_group(year1)
            if rows0.shape[0] > 1 and rows1.shape[0] > 1:
                for tag, year in zip(rows0["tag"], rows0["year"]):
                    vv = model0.docvecs[str(tag)]
                    S.append(vv)
                for tag, year in zip(rows1["tag"], rows1["year"]):
                    vv = model1.docvecs[str(tag)]
                    D.append(vv)

                nS = len(S)
                nD = len(D)

                # define transportation problem
                dd = cdist(S,D)
                cap = [1.0/nS]*nS
                dem = [1.0/nD]*nD
                
                result = solveTransport(dd, cap, dem, nS, nD)
                #  print("Result {0}-{1} is  = {2}".format(year,year+1, result))
                avgs.append(result)
                years.append(year1)
                tots.append(nS*nD)
                names.append(comp)

    df = pd.DataFrame({
        "cik": names,
        "year": years,
        "avg" : avgs,
        "tot" : tots
    })

    return df



def main(argv):

    df = wmdTransport2()

    df.to_csv("profileWmd.csv")

    plotProfile(df)



if __name__ == '__main__':

    main(sys.argv[1:])
