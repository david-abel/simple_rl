''' DQNAgentClass.py: Class for Deep Q-network agent. Built based on the network
in DeepMind, Multi-agent RL in Sequential Social Dilemmas paper. '''

# Python imports.
import tensorflow as tf
import numpy as np
import random

# Other imports.
from simple_rl.agents.AgentClass import Agent

class DQNAgent(Agent):

    NAME = "dqn-deep-mind"

    def __init__(self, actions, name=NAME, learning_rate=1e-4,  x_dim=210, y_dim=160, eps_start=1.0, eps_decay=0.0000001, eps_end=0.1, num_channels=3, should_train=True, from_checkpoint=None, player_id=1):
        Agent.__init__(self, name=name, actions=[])
        self.learning_rate = learning_rate
        self.x_dim, self.y_dim = x_dim, y_dim
        self.actions, self.num_actions = actions, len(actions)
        self.hidden_layers = [32, 32]
        self.num_channels = num_channels
        self.eps_start, self.epsilon_decay, self.epsilon_end = eps_start, eps_decay, eps_end
        self.should_train = should_train
        self.reset()

        # Parameters for updating target network.
        tau = 0.001

        # TODO: Update to support player_id > 2.
        # NOTE: This is a bit of a hack to update the variables in the target
        # network. It can be fixed by using scope and Tensorflow 1.4 which takes
        # a scope argument in tf.trainable_variables().
        if player_id == 2:
            vs = tf.trainable_variables()
            self.target_ops = update_target_graph(vs[len(vs)//2:], tau)
        else:
            self.target_ops = update_target_graph(tf.trainable_variables(), tau)

        # Load model from a checkpoint
        if not (from_checkpoint is None):
            self.saver.restore(self.sess, from_checkpoint)
            print('Restored model from checkpoint: {}'.format(from_checkpoint))

    def act(self, state, reward):
        '''
        Args:
            state (simple_rl.State)
            reward (float)

        Returns:
            (str)
        '''
        # Training
        if self.should_train and self.total_steps > 0 and self.total_steps % self.update_freq == 0:
            s, a, r, s2, t = self.experience_buffer.sample(self.batch_size)
            targetVals = self.targetQN.predict_target(self.sess, s2)

            # Compute y-vals
            y = np.zeros(self.batch_size)
            for i in range(self.batch_size):
                if t[i]:
                    y[i] = r[i]
                else:
                    y[i] = r[i] + targetVals[i]
            l = self.mainQN.train(self.sess, s, a, y)

            if self.print_loss and (self.total_steps % self.print_every == 0):
                print('Loss for step {}: {}'.format(self.total_steps, l))

            update_target(self.target_ops, self.sess)

        # Not Training (or after training)
        if random.random() < self.epsilon:
            action =  np.random.choice(self.num_actions) # NOTE:  Again assumes actions encoded as integers
        else:
            img = state.to_rgb(self.x_dim, self.y_dim)
            action = self.mainQN.get_best_action(self.sess, img)[0]

        if not (self.prev_state is None) and not (self.prev_action is None):
            self.experience_buffer.add((self.prev_state, self.prev_action, reward, state.to_rgb(self.x_dim, self.y_dim), state.is_terminal()))

        self.prev_state, self.prev_action = state.to_rgb(self.x_dim, self.y_dim), action

        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay

        # Saving checkpoints (NOTE: We only save checkpoints when training)
        if self.should_train and self.should_save and self.total_steps > 0 and self.total_steps % self.save_every == 0:
            save_path = self.saver.save(self.sess, '/tmp/{}.ckpt'.format(self.name))
            print('At step {}, saved model to {}'.format(self.total_steps, save_path))

        self.curr_step += 1
        self.total_steps += 1
        if state.is_terminal():
            self.curr_step = 0
            self.curr_episode += 1

        self.action_counts[action] += 1
        return self.actions[action]

    def __str__(self):
        return str(self.name)

    def reset(self):
        self.mainQN = QNetwork(learning_rate=self.learning_rate, num_actions=self.num_actions, x_dim=self.x_dim, y_dim=self.y_dim, num_channels=self.num_channels)
        self.targetQN = QNetwork(learning_rate=self.learning_rate, num_actions=self.num_actions, x_dim=self.x_dim, y_dim=self.y_dim, num_channels=self.num_channels)
        self.sess = tf.Session()
        self.experience_buffer = ExperienceBuffer(buffer_size=10e5)
        self.epsilon = self.eps_start
        self.prev_state, self.prev_action = None, None
        self.curr_step, self.total_steps = 0, 0
        self.curr_episode = 0
        self.update_freq = 100
        self.batch_size = 32
        self.update_target = 100
        self.should_save, self.save_every = True, 100000
        self.print_loss, self.print_every = True, 10000
        self.saver = tf.train.Saver()
        self.action_counts = np.zeros(self.num_actions)
        self.sess.run(tf.global_variables_initializer())


# --------------
# -- QNetwork --
# --------------
class QNetwork():
    def __init__(self, learning_rate=1e-4, num_actions=8, x_dim=21, y_dim=16, num_channels=3):
        self.learning_rate = learning_rate
        self.num_channels = num_channels
        self.hidden_layers = [32, 32]
        self.num_actions = num_actions
        self.x_dim, self.y_dim = x_dim, y_dim
        self.num_channels = num_channels

        self.image = tf.placeholder(tf.float32, shape=[None, self.x_dim, self.y_dim, self.num_channels], name='image')
        self.targetQ = tf.placeholder(tf.float32, shape=[None], name='targetQ')

        self.actions = tf.placeholder(tf.int32, shape=[None], name='actions')

        self.out = self.setup_network(self.image)

        self.predict = tf.argmax(self.out, 1)

        self.loss_val = self.loss(self.out)

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val)

    def setup_network(self, inpt):
        flattened_input = tf.reshape(inpt, [-1, self.num_channels*self.x_dim*self.y_dim])

        curr_layer = flattened_input
        for i, layer_size in enumerate(self.hidden_layers):
            curr_layer = tf.layers.dense(curr_layer, units=layer_size, activation=tf.nn.relu)

        return tf.layers.dense(curr_layer, units=self.num_actions)

    def loss(self, output):
        actions_onehot = tf.one_hot(self.actions, self.num_actions, dtype=tf.float32)
        Q = tf.reduce_sum(actions_onehot * self.out, axis=1)
        return tf.reduce_mean(tf.square(self.targetQ - Q))

    def train(self, sess, s, a, y):
        _, l = sess.run([self.train_op, self.loss_val], feed_dict={self.targetQ: y, self.image: s, self.actions: a})
        return l

    def predict_target(self, sess, states):
        vals = sess.run(self.out, feed_dict={self.image: states})
        return np.max(vals, axis=1)

    def get_best_action(self, sess, img):
        return sess.run(self.predict, feed_dict={self.image: [img]})

# -----------------------
# -- Experience Buffer --
# -----------------------
class ExperienceBuffer():

    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)

        self.buffer.append(experience)

    def sample(self, size):
        indexes = np.random.randint(0, high=len(self.buffer), size=size)
        s1 = [self.buffer[index][0] for index in indexes]
        a = [self.buffer[index][1] for index in indexes]
        r = [self.buffer[index][2] for index in indexes]
        s2 = [self.buffer[index][3] for index in indexes]
        t = [self.buffer[index][4] for index in indexes]
        return [s1, a, r, s2, t]

# Used to update TF networks
def update_target_graph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def update_target(op_holder,sess):
    for op in op_holder:
        sess.run(op)
