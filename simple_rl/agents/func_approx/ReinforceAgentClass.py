''' ReinforceAgentClass.py: Class for an agent acting according to REINFORCE. '''

# Python imports.
import random

# Local imports.
from AgentClass import Agent

class ReinforceAgent(Agent):
    ''' Class for a REINFORCE agent. '''

    def __init__(self, actions):
    	self.policy
        Agent.__init__(self, name="reinforce", actions=actions)

    def act(self, state, reward):
        return random.choice(self.actions)





class REINFORCE(policy_gradient):
    """REINFORCE policy gradient algorithm.
    From the paper:
    Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning,
    Ronald Williams, 1992.
    """

    name = "REINFORCE"

   	def __init__(self, actions):
   		self.base_numerator = numpy.zeros(self.weights.shape)
        self.base_denominator = numpy.zeros(self.weights.shape)
        self.gradient_estimate = numpy.zeros(self.weights.shape)
		Agent.__init__(self, name="reinforce", actions=actions)



    def agent_init(self,taskSpec):
        super(REINFORCE, self).agent_init(self,taskSpec)
        self.baseline_numerator = numpy.zeros(self.weights.shape)
        self.baseline_denom = numpy.zeros(self.weights.shape)
        self.gradient_estimate = numpy.zeros(self.weights.shape)
        self.ep_count = 0

    def init_parameters(self):
        super(REINFORCE, self).init_parameters(self)
        self.num_rollouts = self.params.setdefault('num_rollouts', 5)

    @classmethod
    def agent_parameters(cls):
        param_set = super(REINFORCE, cls).agent_parameters()
        add_parameter(param_set, "num_rollouts", default=5, type=int, min=1, max=50)
        return param_set

    def agent_start(self,observation):
        if self.ep_count > 0:
            self.baseline_numerator += (self.traces**2) * self.Return
            self.baseline_denom += self.traces**2
            self.gradient_estimate += self.traces * -((self.baseline_numerator/self.baseline_denom) - self.Return)
        if self.ep_count == self.num_rollouts:
            # update the parameters...
            self.weights += self.step_sizes * self.gradient_estimate
            # Clear estimates for next set of roll outs
            self.gradient_estimate.fill(0.0)
            self.baseline_numerator.fill(0.0)
            self.baseline_denom.fill(0.0)
            self.ep_count = 0

        self.ep_count += 1
        self.Return = 0.0
        return super(REINFORCE, self).agent_start(self, observation)

    def update(self, phi_t, phi_tp, reward, compatFeatures):
        self.traces += compatFeatures
        self.Return += (self.gamma**(self.step_count-1.)) * reward