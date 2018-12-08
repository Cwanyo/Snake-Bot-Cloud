import numpy
from random import sample
from collections import deque

from keras import backend as K


# ref: https://github.com/farizrahman4u/qlearning4k/blob/master/qlearning4k/memory.py
class DualExperienceReplay:

    def __init__(self, explore_memory_size=100, exploit_memory_size=100, reward_threshold=0, fast=True):
        self.fast = fast
        self.explore_memory = deque() if explore_memory_size == -1 else deque(maxlen=explore_memory_size)
        self.explore_memory_size = explore_memory_size
        self.exploit_memory = deque() if exploit_memory_size == -1 else deque(maxlen=exploit_memory_size)
        self.exploit_memory_size = exploit_memory_size
        self.reward_threshold = reward_threshold
        self.input_shape = None
        self.batch_function = None

        # Log
        self.explore_memory_filled = False
        self.exploit_memory_filled = False

    def remember(self, state, action_index, reward, state_next, game_over):
        self.input_shape = state.shape[1:]

        if reward > self.reward_threshold:
            self.exploit_memory.append(numpy.concatenate(
                [state.flatten(), numpy.array(action_index).flatten(), numpy.array(reward).flatten(),
                 state_next.flatten(), 1 * numpy.array(game_over).flatten()]))

            # Log exploit
            if not self.exploit_memory_filled and not len(self.exploit_memory) % (self.exploit_memory_size / 10):
                print('-- Exploit Memory:', len(self.exploit_memory))
            if not self.exploit_memory_filled and len(self.exploit_memory) == self.exploit_memory_size:
                self.exploit_memory_filled = True
                print('-- Exploit Memory: filled')
        else:
            self.explore_memory.append(numpy.concatenate(
                [state.flatten(), numpy.array(action_index).flatten(), numpy.array(reward).flatten(),
                 state_next.flatten(), 1 * numpy.array(game_over).flatten()]))

            # Log explore
            if not self.explore_memory_filled and not len(self.explore_memory) % (self.explore_memory_size / 10):
                print('-- Explore Memory:', len(self.explore_memory))
            if not self.explore_memory_filled and len(self.explore_memory) == self.explore_memory_size:
                self.explore_memory_filled = True
                print('-- Explore Memory: filled')

    def check_memory_availability(self, batch_size, explore_exploit_ratio):
        explore_size = int(batch_size * explore_exploit_ratio)
        exploit_size = int(batch_size - explore_size)

        if len(self.explore_memory) < explore_size or len(self.exploit_memory) < exploit_size:
            return False
        else:
            return True

    def get_sample(self, batch_size, explore_exploit_ratio):
        explore_size = int(batch_size * explore_exploit_ratio)
        exploit_size = int(batch_size - explore_size)

        explore_sample = numpy.array(sample(self.explore_memory, explore_size))
        exploit_sample = numpy.array(sample(self.exploit_memory, exploit_size))

        concat_sample = numpy.concatenate([explore_sample, exploit_sample])
        numpy.random.shuffle(concat_sample)

        return concat_sample

    def get_batch(self, model, target_model, batch_size, explore_exploit_ratio, gamma=0.9):
        if self.fast:
            return self.get_batch_fast(model, target_model, batch_size, explore_exploit_ratio, gamma)

        if not self.check_memory_availability(batch_size, explore_exploit_ratio):
            return None

        num_actions = model.get_output_shape_at(0)[-1]
        samples = self.get_sample(batch_size, explore_exploit_ratio)
        input_dim = numpy.prod(self.input_shape)

        state = samples[:, 0: input_dim]
        action_index = samples[:, input_dim]
        reward = samples[:, input_dim + 1]
        next_state = samples[:, input_dim + 2: 2 * input_dim + 2]
        game_over = samples[:, 2 * input_dim + 2]

        reward = reward.repeat(num_actions).reshape((batch_size, num_actions))
        game_over = game_over.repeat(num_actions).reshape((batch_size, num_actions))
        state = state.reshape((batch_size,) + self.input_shape)
        next_state = next_state.reshape((batch_size,) + self.input_shape)

        q_next_state = numpy.max(target_model.predict(next_state), axis=1).repeat(num_actions).reshape(
            (batch_size, num_actions))

        delta = numpy.zeros((batch_size, num_actions))
        action_index = numpy.cast['int'](action_index)
        delta[numpy.arange(batch_size), action_index] = 1
        targets = (1 - delta) * model.predict(state) + delta * (reward + gamma * (1 - game_over) * q_next_state)
        return state, targets

    def reset_explore_memory(self):
        self.explore_memory.clear()

    def reset_exploit_memory(self):
        self.exploit_memory.clear()

    def remove_explore_memory(self, old_percent):
        i = int(len(self.explore_memory) * old_percent)
        for _ in range(i):
            self.explore_memory.popleft()

    def remove_exploit_memory(self, old_percent):
        i = int(len(self.exploit_memory) * old_percent)
        for _ in range(i):
            self.exploit_memory.popleft()

    def set_batch_function(self, model, target_model, input_shape, batch_size, num_actions, gamma):
        input_dim = numpy.prod(input_shape)
        samples = K.placeholder(shape=(batch_size, input_dim * 2 + 3))

        state = samples[:, 0: input_dim]
        action_index = samples[:, input_dim]
        reward = samples[:, input_dim + 1]
        next_state = samples[:, input_dim + 2: 2 * input_dim + 2]
        game_over = samples[:, 2 * input_dim + 2: 2 * input_dim + 3]

        reward = K.reshape(reward, (batch_size, 1))
        reward = K.repeat(reward, num_actions)
        reward = K.reshape(reward, (batch_size, num_actions))
        game_over = K.repeat(game_over, num_actions)
        game_over = K.reshape(game_over, (batch_size, num_actions))
        state = K.reshape(state, (batch_size,) + input_shape)
        next_state = K.reshape(next_state, (batch_size,) + input_shape)

        q_next_state = K.max(target_model(next_state), axis=1)
        q_next_state = K.reshape(q_next_state, (batch_size, 1))
        q_next_state = K.repeat(q_next_state, num_actions)
        q_next_state = K.reshape(q_next_state, (batch_size, num_actions))

        delta = K.reshape(K.one_hot(K.reshape(K.cast(action_index, "int32"), (-1, 1)), num_actions),
                          (batch_size, num_actions))

        targets = (1 - delta) * model(state) + delta * (reward + gamma * (1 - game_over) * q_next_state)
        self.batch_function = K.function(inputs=[samples], outputs=[state, targets])

    def get_batch_fast(self, model, target_model, batch_size, explore_exploit_ratio, gamma):
        if not self.check_memory_availability(batch_size, explore_exploit_ratio):
            return None
        samples = self.get_sample(batch_size, explore_exploit_ratio)
        if not hasattr(self, 'batch_function') or self.batch_function is None:
            self.set_batch_function(model, target_model, self.input_shape, batch_size,
                                    model.get_output_shape_at(0)[-1], gamma)
        state, targets = self.batch_function([samples])
        return state, targets
