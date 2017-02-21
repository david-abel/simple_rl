# Python imports
import numpy as np

# Local imports.
from AgentClass import Agent

class PolicyGradient(Agent):
    def __init__(self):
        # self.weights = np.zeros(self.numActions))

    def update(self, phi_t, phi_tp, reward, compatFeatures):
        pass

    def agent_step(self,reward, observation):
        """Take one step in an episode for the agent, as the result of taking the last action.
        Args:
            reward: The reward received for taking the last action from the previous state.
            observation: The next observation of the episode, which is the consequence of taking the previous action.
        Returns:
            The next action the RL agent chooses to take, represented as an RLGlue Action object.
        """

        newState = numpy.array(list(observation.doubleArray))
        lastState = numpy.array(list(self.lastObservation.doubleArray))
        lastAction = self.lastAction.intArray[0]
        newDiscState = self.getDiscState(observation.intArray)
        lastDiscState = self.getDiscState(self.lastObservation.intArray)
        newIntAction = self.getAction(newState, newDiscState)

        phi_t = numpy.zeros((self.weights.shape[0], self.weights.shape[1]))
        phi_tp = numpy.zeros((self.weights.shape[0], self.weights.shape[1]))
        phi_t[lastDiscState, :] = self.basis.computeFeatures(lastState)
        phi_tp[newDiscState, :] = self.basis.computeFeatures(newState)

        self.step_count += 1
        self.update(phi_t, phi_tp, reward, self.getCompatibleFeatures(phi_t, lastAction, reward, phi_tp, newIntAction))

        returnAction=Action()
        returnAction.intArray=[newIntAction]
        self.lastAction=copy.deepcopy(returnAction)
        self.lastObservation=copy.deepcopy(observation)
        return returnAction

    def agent_end(self,reward):
        """Receive the final reward in an episode, also signaling the end of the episode.
        Args:
            reward: The reward received for taking the last action from the previous state.
        """
        lastState = numpy.array(list(self.lastObservation.doubleArray))
        lastAction = self.lastAction.intArray[0]
        lastDiscState = self.getDiscState(self.lastObservation.intArray)

        phi_t = numpy.zeros((self.weights.shape[0], self.weights.shape[1]))
        phi_tp = numpy.zeros((self.weights.shape[0], self.weights.shape[1]))
        phi_t[lastDiscState, :] = self.basis.computeFeatures(lastState)

        self.update(phi_t, phi_tp, reward, self.getCompatibleFeatures(phi_t, lastAction, reward, phi_tp, 0))

    def getAction(self, state, discState):
        features = numpy.zeros(self.weights.shape[:-1])
        features[discState, :] = self.basis.computeFeatures(state)
        policy = self.getPolicy(features)
        return numpy.where(policy.cumsum() >= numpy.random.random())[0][0]

    def getPolicy(self, features):
        if self.softmax:
            return self.softmax_policy(features)
        else:
            return self.gauss_policy(features)

    def gauss_policy(self, features):
        # Not currently supported...
        return self.softmax_policy(features)

    def softmax_policy(self, features):
        # Compute softmax policy
        policy = numpy.dot(self.weights.reshape((features.size,self.numActions)).T, features.ravel())
        policy = numpy.exp(numpy.clip(policy/self.epsilon, -500, 500))
        policy /= policy.sum()
        return policy

    def getCompatibleFeatures(self, features, action, reward, next_features, next_action):
        policy = -1.0 * self.getPolicy(features)
        policy[action] += 1.0
        features = numpy.repeat(features.reshape((features.size,1)), self.numActions, axis=1)
        return numpy.dot(features, numpy.diag(policy)) # This is probably a slow way to do it
