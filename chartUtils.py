# Python libs.
import pylab
import ast
import numpy as np
import math
import argparse
from os import listdir, remove, walk, system
from os.path import isfile, isdir, join
import numpy as np  

def loadData(experimentName, experimentAgents):
    '''
    Args:
        experimentName (str): Points to the file containing all the data.
        experimentAgents (list): Points to which results files will be plotted.

    Returns:
        result (list): A 3d matrix containing rewards, where the dimensions are [algorithm][instance][episode].
    '''

    result = []
    for alg in experimentAgents:

        # Load the reward for all instances of each agent
        allReward = open(experimentName + "/" + str(alg) + ".csv", "r")
        allInstances = []

        # Put the reward instances into a list of floats.
        for instance in allReward.readlines():
            allEpisodesForInstance = [float(r) for r in instance.split(",")[:-1]]
            allInstances.append(allEpisodesForInstance)

        result.append(allInstances)

    return result


def averageData(data, cumulative=False):
    '''
    Args:
        data (list): a 3D matrix, [algorithm][instance][episode]
        cumulative (bool) *opt: determines if we should compute the average cumulative reward/cost or just regular.

    Returns:
        (list): a 2D matrix, [algorithm][episode], where the instance rewards have been averaged.
    '''

    numAlgorithms = len(data)
    numInstances = len(data[0])

    result = [None for i in xrange(numAlgorithms)] # [Alg][avgRewardEpisode], where avg is summed up to episode i if @cumulative=True

    for i, algInstances in enumerate(data):

        # Take the average.
        algInstances = np.array(algInstances)
        avged = None
        try:
            avged = algInstances.sum(axis=0)/float(numInstances)
        except TypeError:
            print "Error: something went wrong. I couldn't find an algorithm or things were run a weird number of times."
            quit()

        if cumulative:
            # If we're summing over episodes.
            temp = []
            totalSoFar = 0
            for rew in avged:
                totalSoFar += rew

                temp.append(totalSoFar)

            avged = temp

        result[i] = avged

    return result

def computeConfidenceIntervals(data, cumulative=False):
    '''
    Args:
        data (list): A 3D matrix, [algorithm][instance][episode]
        cumulative (bool) *opt
    '''

    # Relevant params.
    confidenceIntervalsEachAlg = [] # [alg][conf_inv_for_episode]
    
    for i, algInstances in enumerate(data):

        numInstances = len(data[i])
        numEpisodes = len(data[i][0])

        algInstances = np.array(algInstances)
        thisAlgsCI = []
        totalSoFar = np.zeros(numInstances)
        for j in xrange(numEpisodes):
            # Compute datum for confidence interval.
            nextVector = algInstances[:,j]

            while len(nextVector) < numInstances:
                nextVector = np.append(nextVector, 0)
            if cumulative:
                # Cumulative.
                summedVector = np.add(nextVector, totalSoFar)
                totalSoFar = np.add(nextVector, totalSoFar)
                nextVector = summedVector

            # Compute the interval and add it to list.
            ci = computeSingleConfidenceInterval(nextVector)
            thisAlgsCI.append(ci)

        confidenceIntervalsEachAlg.append(thisAlgsCI)

    return confidenceIntervalsEachAlg


def computeSingleConfidenceInterval(datum):
    '''
    Args:
        datum (list): A vector of data points to compute the confidence interval of.

    Returns:
        (float): Margin of error.
    '''
    stdDev = np.std(datum)
    stdErr = stdDev / math.sqrt(len(datum))
    marginOfError = stdErr * 2
    return marginOfError


def plot(data, agents, useCost=False, cumulative=False, confidenceIntervals=[0.01]*8, resultsPrefix=""):
    '''
    Args:
        data (list of lists): each element is itself the reward from an episode for an algorithm.
        useCost (bool) [optional]: If true, plots are in terms of cost. Otherwise, plots are in terms of reward.
        cumulative (bool) [optional]: If true, plots are cumulative cost/reward.
        confidenceIntervals (list of floats) [optional]: confidence intervals to display with the chart.
        resultsPrefix (str)

    Summary:
        Makes (and opens) a single reward chart plotting all of the data in @data.
    '''

    # Some nice colors for plotting.
    colors = [[240,163,255],[0,117,220],[153,63,0],[76,0,92],[25,25,25],[0,92,49],[255,204,153],[128,128,128],[148,255,181],[143,124,0],[157,204,0],[194,0,136],[0,51,128],[255,164,5],[255,168,187],[66,102,0],[255,0,16],[94,241,242],[0,153,143],[224,255,102],[116,10,255],[153,0,0],[255,255,128],[255,255,0],[255,80,5]]
    colors = [[shade / 255.0 for shade in rgb] for rgb in colors] # Map them to floats in [0:1].

    # Puts the legend into the best location in the plot.
    pylab.rcParams['legend.loc'] = 'best' 

    results = data
    if useCost:
        results = [[-x for x in alg] for alg in data]

    ymin = min([min(d) for d in results])
    ymax = max([max(d) for d in results])
    pylab.ylim(ymin=ymin,ymax=ymax)

    randColorIndex = 0
    showErrorBarsEveryNPoints = 8 # Will show the error bars every N episodes. (pretty messy otherwise)

    # Make the plot.
    for i, alg in enumerate(agents):
        # Add figure for this algorithm.
        randColor = colors[(randColorIndex + i + 5) % len(colors)]

        # Compute CIs
        CI = confidenceIntervals[i] #[::showErrorBarsEveryNPoints]
        try:
            pylab.errorbar(range(len(results[i])), results[i], yerr=CI, errorevery=showErrorBarsEveryNPoints, fmt='', color=randColor)
        except ValueError:
            print "Error: dimension mismatch. Two algorithms were run a different number of episodes."
            quit()
        # plt.fill_between(x, y-yerr, y+yerr,facecolor='r',alpha=0.5)

        legendInfo = alg

        print "Mean (" + str(agents[i]) + ") :", results[i][-1], "(CI:",CI[-1],")"

        pylab.plot(range(len(results[i])), results[i], color=randColor, label=legendInfo)
        pylab.legend()

    # Configure plot naming information.
    unit = "Cost" if useCost else "Reward"

    plotLabel = "Cumulative" if cumulative else "Average"
    plot_name = resultsPrefix + "/all_" + plotLabel.lower() + "_" + unit.lower() + ".pdf"
    plotTitle = plotLabel + " " + unit + ""
    yAxisLabel = plotLabel + " " + unit

    # Make the plot.
    pylab.xlabel('Episode Number')
    pylab.ylabel(yAxisLabel)
    pylab.title(plotTitle)
    pylab.grid(True)
    pylab.savefig(plot_name, format="pdf")
    pylab.cla() # Clears.

    system("open " + plot_name)


def makePlots(experimentName, experimentAgents, cumulative=True):
    '''
    Args:
        experimentName (str)
        experimentAgents (list)
        cumulative (bool) [opt]

    Summary:
        Creates plots for all agents run under the experiment.
        Stores the plot in results/<experiment_name>/results.pdf
    '''

    # Load the data.
    data = loadData(experimentName, experimentAgents) # [alg][instance][episode]

    # Average the data.
    avgData = averageData(data, cumulative=cumulative)

    # Compute confidence intervals.
    confIntervals = computeConfidenceIntervals(data, cumulative=cumulative)

    # Create plot.
    plot(avgData, experimentAgents, useCost=True, cumulative=cumulative, confidenceIntervals=confIntervals, resultsPrefix = experimentName)
