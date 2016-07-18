'''
Code for running experiments on an MDP

Instructions:
    (1) Set mdp in main.
    (2) Create agent instances.
    (3) Set experiment parameters (num_instances, num_episodes, num_steps).
    (4) Call run_agents_on_mdp(agents, mdp).

    -> Runs all experiments and will open a plot with results when finished.

Functions:
    run_agents_on_mdp: Carries out an experiment with the given agents, mdp, and parameters.
    main: Creates an MDP, a few agents, and runs an experiment.

Author: David Abel (cs.brown.edu/~dabel/)
'''

# Python imports.
import time
from collections import defaultdict

# Local imports.
from simple_rl.tasks import GridWorldMDP
from simple_rl.tasks import ChainMDP
from simple_rl.experiments import Experiment
from simple_rl.agents import RandomAgent
from simple_rl.agents import RMaxAgent
from simple_rl.agents import QLearnerAgent

def run_agents_on_mdp(agents, mdp, num_instances=5, num_episodes=20, num_steps=20):
    '''
    Args:
        agent (Agent): See agents/AgentClass.py (and friends).
        mdp (MDP): See mdp/MDPClass.py for the abstract class. Specific MDPs in tasks/*.
        num_instances (int) [opt]: Number of times to run each agent (for confidence intervals).
        num_episodes (int) [opt]: Number of episodes for each learning instance.
        num_steps (int) [opt]: Number of steps per episode.

    Summary:
        Runs each agent on the given mdp according to the given parameters.
        Stores results in results/<agent_name>.csv and automatically
        generates a plot and opens it.
    '''

    # Experiment (for reproducibility, plotting).
    exp_params = {"num_instances":num_instances, "num_episodes":num_episodes, "num_steps":num_steps}
    experiment = Experiment(agents=agents, mdp=mdp, params=exp_params)

    # Record how long each agent spends learning.
    times = defaultdict(float)
    print "Running experiment: \n" + str(experiment)
    # Learn.
    for agent in agents:
        print str(agent) + " is learning."
        start = time.clock()

        # For each instance of the agent.
        for instance in xrange(num_instances):
            # For each episode.
            for episode in xrange(num_episodes):

                # Compute initial state/reward.
                state = mdp.get_init_state()
                reward = 0

                for step in xrange(num_steps):
                    # Compute the agent's policy.
                    action = agent.act(state, reward)

                    # Execute the action in the MDP.
                    reward, next_state = mdp.execute_agent_action(action)

                    # Record the experience.
                    experiment.add_experience(agent, state, action, reward, next_state)

                    # Update pointer.
                    state = next_state

                # Process experiment info at end of episode.
                experiment.end_of_episode(agent)

            # Process that learning instance's info at end of learning.
            experiment.end_of_instance(agent)

            # Reset the agent and MDP.
            agent.reset()

        # Track how much time this agent took.
        end = time.clock()
        times[agent] = round(end - start, 3)

    # Time stuff.
    print "\n--- TIMES ---"
    for agent in times.keys():
        print str(agent) + " agent took " + str(times[agent]) + " seconds."
    print "-------------\n"

    experiment.make_plots()


def main():
    '''
    Summary:
        Main function:
            (1) Create an MDP.
            (2) Create each agent instance.
            (3) Run them on the mdp.
    '''
    # MDP.
    mdp = GridWorldMDP(10, 10, (1, 1), (10, 10))
    # mdp = ChainMDP(15)
    actions = mdp.get_actions()
    gamma = mdp.get_gamma()

    # Agent.
    random_agent = RandomAgent(actions)
    rmax_agent = RMaxAgent(actions, gamma=gamma)
    qlearner_agent = QLearnerAgent(actions, gamma=gamma)

    # Run experiments.
    run_agents_on_mdp([qlearner_agent, rmax_agent, random_agent], mdp)


if __name__ == "__main__":
    main()
