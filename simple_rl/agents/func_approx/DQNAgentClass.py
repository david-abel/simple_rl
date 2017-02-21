
# Local classes
from ..QLearnerAgentClass import QLearnerAgent

def class_vars(obj):
    return {k:v for k, v in inspect.getmembers(obj) if not k.startswith('__') and not callable(k)}

class DQNAgent(QLearnerAgent):
    '''
    Agent Class with a Deep Q Network.

    Based on the the dqn implementation here: https://github.com/devsisters/DQN-tensorflow
    '''

    def __init__(self, actions, config, name="dqn", alpha=0.05, gamma=0.95, epsilon=0.01, explore="uniform"):
        '''
        Args:
        '''
        self.name = "linear-" + explore
        QLearnerAgent.__init__(self, actions=list(actions), name=name, alpha=alpha, gamma=gamma, epsilon=epsilon, explore=explore)

        self._saver = None
        self.config = config

        try:
          self._attrs = config.__dict__['__flags']
        except:
          self._attrs = class_vars(config)
        pp(self._attrs)

        self.config = config

        for attr in self._attrs:
            name = attr if not attr.startswith('_') else attr[1:]
            setattr(self, name, getattr(self.config, attr))

    def save_model(self, step=None):
        print(" [*] Saving checkpoints...")
        model_name = type(self).__name__

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess, self.checkpoint_dir, global_step=step)

    def load_model(self):
        print(" [*] Loading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.checkpoint_dir, ckpt_name)
            self.saver.restore(self.sess, fname)
            print(" [*] Load SUCCESS: %s" % fname)
            return True
        else:
            print(" [!] Load FAILED: %s" % self.checkpoint_dir)
            return False

    def act(self, state, reward):
        '''
        Args:
            state (State): see StateClass.py
            reward (float): the reward associated with arriving in state @state.

        Returns:
            (str): action.
        '''

    def epsilon_greedy_policy(self, state):
        # ep = test_ep or (self.ep_end + max(0., (self.ep_start - self.ep_end)
          # * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

        if random.random() < ep:
          action = random.randrange(self.env.action_size)
        else:
          action = self.q_action.eval({self.s_t: [s_t]})[0]

        return action

    def train(self):
        start_step = self.step_op.eval()
        start_time = time.time()

        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        max_avg_ep_reward = 0
        ep_rewards, actions = [], []

        screen, reward, action, terminal = self.env.new_random_game()

        for _ in range(self.history_length):
            self.history.add(screen)

        for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
            if self.step == self.learn_start:
                num_game, self.update_count, ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_rewards, actions = [], []

            # 1. predict
            action = self.predict(self.history.get())
            # 2. act
            screen, reward, terminal = self.env.act(action, is_training=True)
            # 3. observe
            self.observe(screen, reward, action, terminal)

            if terminal:
                screen, reward, action, terminal = self.env.new_random_game()

                num_game += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.
            else:
                ep_reward += reward

            actions.append(action)
            total_reward += reward

            if self.step >= self.learn_start:
                if self.step % self.test_step == self.test_step - 1:
                    avg_reward = total_reward / self.test_step
                    avg_loss = self.total_loss / self.update_count
                    avg_q = self.total_q / self.update_count

                try:
                    max_ep_reward = np.max(ep_rewards)
                    min_ep_reward = np.min(ep_rewards)
                    avg_ep_reward = np.mean(ep_rewards)
                except:
                    max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

                print '\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
                    % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game)

                if max_avg_ep_reward * 0.9 <= avg_ep_reward:
                    self.step_assign_op.eval({self.step_input: self.step + 1})
                    self.save_model(self.step + 1)

                    max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

                if self.step > 180:
                    self.inject_summary({
                        'average.reward': avg_reward,
                        'average.loss': avg_loss,
                        'average.q': avg_q,
                        'episode.max reward': max_ep_reward,
                        'episode.min reward': min_ep_reward,
                        'episode.avg reward': avg_ep_reward,
                        'episode.num of game': num_game,
                        'episode.rewards': ep_rewards,
                        'episode.actions': actions,
                        'training.learning_rate': self.learning_rate_op.eval({self.learning_rate_step: self.step}),
                    }, self.step)

                num_game = 0
                total_reward = 0.
                self.total_loss = 0.
                self.total_q = 0.
                self.update_count = 0
                ep_reward = 0.
                ep_rewards = []
                actions = []

    def observe(self, screen, reward, action, terminal):
        reward = max(self.min_reward, min(self.max_reward, reward))

        self.history.add(screen)
        self.memory.add(screen, reward, action, terminal)

        if self.step > self.learn_start:
            if self.step % self.train_frequency == 0:
                self.q_learning_mini_batch()
            if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                self.update_target_q_network()

    @property
    def checkpoint_dir(self):
        return os.path.join('checkpoints', self.model_dir)

    @property
    def model_dir(self):
        model_dir = self.config.env_name
        for k, v in self._attrs.items():
            if not k.startswith('_') and k not in ['display']:
                model_dir += "/%s-%s" % (k, ",".join([str(i) for i in v])
                if type(v) == list else v)
        return model_dir + '/'

    @property
    def saver(self):
        if self._saver == None:
          self._saver = tf.train.Saver(max_to_keep=10)
        return self._saver

    def __str__(self):
        return str(self.name)
