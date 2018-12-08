import numpy
import matplotlib.pyplot as plt

import tensorflow as tf

import trainer.model as Model
from trainer.agent import Agent
from trainer.memory import ExperienceReplay
from trainer.dual_memory import DualExperienceReplay


class DQN:
    def __init__(self, model, target_model, tau, memory_size, img_size, num_frames, actions, output_dir=''):
        self.model = model
        self.target_model = target_model
        self.tau = tau
        self.memory = DualExperienceReplay(memory_size[0], memory_size[1], 0, True)
        self.img_size = img_size
        self.num_frames = num_frames
        self.actions = actions
        self.num_actions = len(actions)
        self.frames = None

        # Log
        self.output_dir = output_dir
        self.writer = tf.summary.FileWriter(output_dir)

    def get_frames(self, board):
        if self.frames is None:
            self.frames = [board] * self.num_frames
        else:
            self.frames.append(board)
            self.frames.pop(0)
        return numpy.expand_dims(self.frames, 0)

    def clear_frames(self):
        self.frames = None

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
        print('-- update_target_model')

    def train(self, episodes, batch_size, gamma, epsilon, epsilon_rate,
              explore_exploit_ratio, explore_exploit_rate,
              test_at_episode=100, test_num_game=10, save_checkpoint=-1):

        # Init epsilon rate
        delta_epsilon = ((epsilon[0] - epsilon[1]) / (episodes * epsilon_rate))
        final_epsilon = epsilon[1]
        epsilon = epsilon[0]

        # Init explore exploit rate of memory
        delta_exploit_ratio = \
            ((explore_exploit_ratio[0] - explore_exploit_ratio[1]) / (episodes * explore_exploit_rate))
        final_exploit_ratio = explore_exploit_ratio[1]
        explore_exploit_ratio = explore_exploit_ratio[0]

        eat_count = 0

        test_score = []

        # Sync both model by update the target weights
        self.update_target_model()

        for e in range(episodes):
            agent = Agent(e, board_size=(self.img_size - 2, self.img_size - 2, 20), random_body=False)

            _, _, _, board, reward = agent.get_state()
            self.clear_frames()
            state = self.get_frames(board)

            loss = 0.0

            while agent.alive:
                if numpy.random.random() > epsilon:
                    # use prediction
                    # q_state = self.model.predict(state.reshape(-1, self.img_size, self.img_size, self.num_frames))
                    q_state = self.model.predict(state.reshape(-1, self.num_frames, self.img_size, self.img_size))
                    action_index = int(numpy.argmax(q_state[0]))
                else:
                    # Explore
                    action_index = numpy.random.randint(self.num_actions)
                    # action_index = agent.get_random_legal_action()

                _, _, _, board, reward = agent.next_state(action_index)

                next_state = self.get_frames(board)
                transition = [state, action_index, reward, next_state, not agent.alive]
                self.memory.remember(*transition)
                state = next_state

                batch = self.memory.get_batch(model=self.model, target_model=self.target_model,
                                              batch_size=batch_size, explore_exploit_ratio=explore_exploit_ratio,
                                              gamma=gamma)
                if batch:
                    inputs, targets = batch
                    loss += self.model.train_on_batch(inputs, targets)

            # Update the target weights
            if not e % self.tau:
                self.update_target_model()

            # Tune epsilon
            if epsilon > final_epsilon:
                epsilon -= delta_epsilon

            # Adjust dual memory ratio
            if explore_exploit_ratio > final_exploit_ratio:
                explore_exploit_ratio -= delta_exploit_ratio

            # Log
            if agent.score:
                eat_count += 1

            # Show result
            print('Episode {:03d}/{:03d} | Loss {:.4f} | Epsilon {:.2f} | '
                  'Step {} | Score {} | Eat {}'.format(e + 1, episodes, loss, epsilon, agent.step, agent.score,
                                                       eat_count))

            # Write scalars on tensorboard
            self.write_scalar('loss', loss, e)

            self.write_scalar('eat_count', eat_count, e)

            self.write_scalar('epsilon', epsilon, e)

            self.write_scalar('explore_exploit_ratio', explore_exploit_ratio, e)

            self.write_scalar('explore_memory_size', len(self.memory.explore_memory), e)

            self.write_scalar('exploit_memory_size', len(self.memory.exploit_memory), e)

            # Test in game at every N episode
            if not e % test_at_episode:
                avg_score, avg_step, avg_loop_detected, avg_bad_dead_detected = self.test_game(episodes=test_num_game)
                test_score.append([e, avg_score])

                self.write_scalar('avg_game_score_by_episode', avg_score, e)

                self.write_scalar('avg_game_step_by_episode', avg_step, e)

                self.write_scalar('avg_loop_detected_by_episode', avg_loop_detected, e)

                self.write_scalar('avg_bad_dead_detected_by_episode', avg_bad_dead_detected, e)

            # Save weight as checkpoint
            if save_checkpoint != -1 and not e % save_checkpoint:
                Model.save_checkpoint(self.model, self.output_dir, e)

        # Plot test game score
        self.plot_test_game_score(test_score, test_at_episode, test_num_game)

    def write_scalar(self, tag, value, e):
        self.writer.add_summary(tf.Summary(value=[
            tf.Summary.Value(tag=tag, simple_value=value),
        ]), e)

    def plot_test_game_score(self, test_score, test_at_episode, test_num_game):
        plt.plot([x[0] for x in test_score], [y[1] for y in test_score], '-o')

        plt.xlabel('At Episode')
        plt.ylabel('Avg Score')
        plt.title('Game Score by Episode \n\nTest at every {} episode for {} games'
                  .format(test_at_episode, test_num_game))
        plt.savefig(self.output_dir + 'game_score_by_episode.png')

    def test_game(self, episodes=10, visualization=False, game_speed=60, random_on_loop=False):
        # State info
        score_list = []
        step_list = []
        num_loop_detected = 0
        num_bad_dead_detected = 0

        for e in range(episodes):
            # Handle snake loop forever
            loop_detected = False
            limit_step_per_food = (self.img_size ** 2)  # * 2
            step_per_food = 0
            cur_score = 0

            agent = Agent(e, False, visualization, game_speed, board_size=(self.img_size - 2, self.img_size - 2, 20))

            s, _, _, board, _ = agent.get_state()
            self.clear_frames()
            state = self.get_frames(board)

            # Log
            pre_s = None
            pre_h = None

            while agent.alive:
                if loop_detected and random_on_loop:
                    loop_detected = False
                    step_per_food = 0
                    # Escape looping forever (might move to wrong direction and die)
                    action_index = numpy.random.randint(self.num_actions)
                    # action_index = agent.get_random_legal_action()
                else:
                    # use prediction
                    # q_state = self.model.predict(state.reshape(-1, self.img_size, self.img_size, self.num_frames))
                    q_state = self.model.predict(state.reshape(-1, self.num_frames, self.img_size, self.img_size))

                    # filter legal action from predicted q value
                    # q_state = agent.filter_legal_action(q_state)

                    action_index = int(numpy.argmax(q_state[0]))

                pre_s = s
                pre_h = agent.snake.heading_direction
                s, _, _, board, _ = agent.next_state(action_index)

                state = self.get_frames(board)

                pre_score = cur_score
                cur_score = agent.score

                if pre_score == cur_score:
                    step_per_food += 1
                else:
                    step_per_food = 0

                if step_per_food > limit_step_per_food:
                    print(e, 'loop detected')
                    num_loop_detected += 1
                    loop_detected = True
                    if not random_on_loop:
                        break

            # Record state
            score_list.append(agent.score)
            step_list.append(agent.step)

            if sum(pre_s) != 3:
                num_bad_dead_detected += 1

            print(e, pre_s, pre_h, agent.snake.heading_direction, agent.score)

        avg_score = sum(score_list) / float(len(score_list))
        avg_step = sum(step_list) / float(len(step_list))
        avg_loop_detected = num_loop_detected / episodes
        avg_bad_dead_detected = num_bad_dead_detected / episodes

        print('------------------------------------------------------')
        print('Total Games:', episodes)
        print('Total Steps:', sum(step_list))
        print('Total Loop Detected:', num_loop_detected)
        print('Avg Loop Detected:', avg_loop_detected)
        print('Total Bad Dead Detected:', num_bad_dead_detected)
        print('Avg Bad Dead Detected:', avg_bad_dead_detected)
        print('Avg Steps:', avg_step)
        print('Max Score:', max(score_list))
        print('Avg Score:', avg_score)
        print('______________________________________________________')

        return avg_score, avg_step, avg_loop_detected, avg_bad_dead_detected
