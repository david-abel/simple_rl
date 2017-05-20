'''
chart_utils.py: Charting utilities for RL.

Functions:
    load_data: Loads data from csv files into lists.
    average_data: Averages data across instances.
    compute_conf_intervals: Confidence interval computation.
    compute_single_conf_interval: Helper function for above.
    plot: Creates (and opens) a single plot using matplotlib.pyplot
    make_plots: Puts everything in order to create the plot.
    main: Loads data from a given path and creates plot.
'''

# Python imports.
import math
import sys
import os
import matplotlib.pyplot as pyplot
import numpy


def load_data(experiment_dir, experiment_agents):
    '''
    Args:
        experiment_dir (str): Points to the file containing all the data.
        experiment_agents (list): Points to which results files will be plotted.

    Returns:
        result (list): A 3d matrix containing rewards, where the dimensions are [algorithm][instance][episode].
    '''

    result = []
    for alg in experiment_agents:

        # Load the reward for all instances of each agent
        all_reward = open(experiment_dir + "/" + str(alg) + ".csv", "r")
        all_instances = []

        # Put the reward instances into a list of floats.
        for instance in all_reward.readlines():
            all_episodes_for_instance = [float(r) for r in instance.split(",")[:-1]]
            all_instances.append(all_episodes_for_instance)

        result.append(all_instances)

    return result


def average_data(data, cumulative=False):
    '''
    Args:
        data (list): a 3D matrix, [algorithm][instance][episode]
        cumulative (bool) *opt: determines if we should compute the average cumulative reward/cost or just regular.

    Returns:
        (list): a 2D matrix, [algorithm][episode], where the instance rewards have been averaged.
    '''

    num_algorithms = len(data)

    result = [None for i in xrange(num_algorithms)] # [Alg][avgRewardEpisode], where avg is summed up to episode i if @cumulative=True

    for i, all_instances in enumerate(data):

        # Take the average.
        num_instances = len(data[i])
        all_instances = numpy.array(all_instances)
        avged = None
        try:
            avged = all_instances.sum(axis=0)/float(num_instances)
        except TypeError:
            print "Error: something went wrong. I couldn't find an algorithm or some algorithm was run under inconsinsent parameters (perhaps experiments are still running)."
            quit()

        if cumulative:
            # If we're summing over episodes.
            temp = []
            total_so_far = 0
            for rew in avged:
                total_so_far += rew

                temp.append(total_so_far)

            avged = temp

        result[i] = avged

    return result

def compute_conf_intervals(data, cumulative=False):
    '''
    Args:
        data (list): A 3D matrix, [algorithm][instance][episode]
        cumulative (bool) *opt
    '''

    confidence_intervals_each_alg = [] # [alg][conf_inv_for_episode]

    for i, all_instances in enumerate(data):

        num_instances = len(data[i])
        num_episodes = len(data[i][0])

        all_instances = numpy.array(all_instances)
        alg_i_ci = []
        total_so_far = numpy.zeros(num_instances)
        for j in xrange(num_episodes):
            # Compute datum for confidence interval.
            episode_j_all_instances = all_instances[:, j]

            if cumulative:
                # Cumulative.
                summed_vector = numpy.add(episode_j_all_instances, total_so_far)
                total_so_far = numpy.add(episode_j_all_instances, total_so_far)
                episode_j_all_instances = summed_vector

            # Compute the interval and add it to list.
            conf_interv = compute_single_conf_interval(episode_j_all_instances)
            alg_i_ci.append(conf_interv)

        confidence_intervals_each_alg.append(alg_i_ci)

    return confidence_intervals_each_alg


def compute_single_conf_interval(datum):
    '''
    Args:
        datum (list): A vector of data points to compute the confidence interval of.

    Returns:
        (float): Margin of error.
    '''
    std_deviation = numpy.std(datum)
    std_error = 1.96*(std_deviation / math.sqrt(len(datum)))

    return std_error


def plot(results, experiment_dir, agents, conf_intervals=[], use_cost=False, cumulative=False, episodic=True):
    '''
    Args:
        results (list of lists): each element is itself the reward from an episode for an algorithm.
        experiment_dir (str): path to results.
        agents (list): each element is an agent that was run in the experiment.
        conf_intervals (list of floats) [optional]: confidence intervals to display with the chart.
        use_cost (bool) [optional]: If true, plots are in terms of cost. Otherwise, plots are in terms of reward.
        cumulative (bool) [optional]: If true, plots are cumulative cost/reward.
        episodic (bool): If true, labels the x-axis "Episode Number". Otherwise, "Step Number". 

    Summary:
        Makes (and opens) a single reward chart plotting all of the data in @data.
    '''

    if experiment_dir[-1] == "/":
        experiment_dir = experiment_dir[:-1]

    # Some nice markers and colors for plotting.
    markers = ['o', 's', 'D', '^', '*', '+', 'p', 'x', 'v','|']
    colors = [[240, 163, 255], [113, 113, 198],[197, 193, 170],\
                [113, 198, 113],[85, 85, 85], [198, 113, 113],\
                [142, 56, 142], [125, 158, 192],[184, 221, 255],\
                [153, 63, 0], [142, 142, 56], [56, 142, 142]]

    x_axis_unit = "episode" if episodic else "step"

    # Map them to floats in [0:1].
    colors = [[shade / 255.0 for shade in rgb] for rgb in colors]
    
    # Puts the legend into the best location in the plot and use a tight layout.
    pyplot.rcParams['legend.loc'] = 'best'
    pyplot.xlim(0, len(results[0]) - 1)

    # Negate everything if we're plotting cost.
    if use_cost:
        results = [[-x for x in alg] for alg in results]

    # Make the plot.
    for i, alg in enumerate(agents):

        # Add figure for this algorithm.
        series_color = colors[i % len(colors)]
        series_marker = markers[i % len(markers)]
        y_axis = results[i]
        x_axis = range(len(y_axis))

        # Plot Confidence Intervals.
        if conf_intervals != []:
            alg_conf_interv = conf_intervals[i]
            top = numpy.add(y_axis, alg_conf_interv)
            bot = numpy.subtract(y_axis, alg_conf_interv)
            pyplot.fill_between(x_axis, top, bot, facecolor=series_color, edgecolor=series_color, alpha=0.25)

        # print "Mean last " + x_axis_unit + ": (" + str(agents[i]) + ") :", y_axis[-1], "(conf_interv:", alg_conf_interv[-1], ")"

        marker_every = max(len(y_axis) / 30,1)
        pyplot.plot(x_axis, y_axis, color=series_color, marker=series_marker, markevery=marker_every, label=alg)
        pyplot.legend()

    # Configure plot naming information.
    unit = "Cost" if use_cost else "Reward"
    plot_label = "Cumulative" if cumulative else "Average"
    plot_name = experiment_dir + "/all_" + plot_label.lower() + "_" + unit.lower() + ".pdf"
    plot_title = plot_label + " " + unit + ": " + experiment_dir.split("/")[-1]
    y_axis_label = plot_label + " " + unit
    pyplot.xlabel(x_axis_unit[0].upper() + x_axis_unit[1:] + " Number")
    pyplot.ylabel(y_axis_label)
    pyplot.title(plot_title)
    pyplot.grid(True)

    # Save the plot.
    pyplot.savefig(plot_name, format="pdf")
    os.system("open " + plot_name)
    
    pyplot.cla() # Clears.

def make_plots(experiment_dir, experiment_agents, cumulative=True, use_cost=False, episodic=True, open_plot=False):
    '''
    Args:
        experiment_dir (str): path to results.
        experiment_agents (list): agent names (looks for "<agent-name>.csv").
        cumulative (bool): If true, plots show cumulative results.
        use_cost (bool): If true, plots are in terms of cost. Otherwise, plots are in terms of reward.
        episodic (bool): If true, labels the x-axis "Episode Number". Otherwise, "Step Number". 

    Summary:
        Creates plots for all agents run under the experiment.
        Stores the plot in results/<experiment_name>/results.pdf
    '''

    # Load the data.
    data = load_data(experiment_dir, experiment_agents) # [alg][instance][episode]

    # Average the data.
    avg_data = average_data(data, cumulative=cumulative)

    # Compute confidence intervals.
    conf_intervals = compute_conf_intervals(data, cumulative=cumulative)

    # Create plot.
    plot(avg_data, experiment_dir, experiment_agents, conf_intervals=conf_intervals, use_cost=use_cost, cumulative=cumulative, episodic=episodic)


def main():
    '''
    Summary:
        For manual plotting.
    '''
    # Make sure we've been given a legitimate batch of results.
    if len(sys.argv) < 2:
        print "Error: you must specify a directory containing the relevant csv files of data."
        print "Usage: python chart_utils.py <path-to-data>"
        quit()

    # Grab agents.
    data_dir = sys.argv[1]
    agents = [agent.replace(".csv","") for agent in os.listdir(data_dir) if ".csv" in agent]
    if len(agents) == 0:
        print "Error: no csv files found."
        quit()

    # Plot.
    make_plots(data_dir, agents)

if __name__ == "__main__":
    main()