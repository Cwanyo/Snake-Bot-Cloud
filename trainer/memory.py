import numpy
from random import sample
from collections import deque

from keras import backend as K


# ref: https://github.com/farizrahman4u/qlearning4k/blob/master/qlearning4k/memory.py
class ExperienceReplay:

    def __init__(self, memory_size=100, fast=True):
        self.fast = fast
        self.memory = deque() if memory_size == -1 else deque(maxlen=memory_size)
        self.memory_size = memory_size
        self.input_shape = None
        self.batch_function = None

        # Log
        self.memory_filled = False

    def remember(self, state, action_index, reward, state_next, game_over):
        self.input_shape = state.shape[1:]
        self.memory.append(numpy.concatenate(
            [state.flatten(), numpy.array(action_index).flatten(), numpy.array(reward).flatten(),
             state_next.flatten(), 1 * numpy.array(game_over).flatten()]))

        # Log
        if not self.memory_filled and not len(self.memory) % (self.memory_size / 10):
            print('-- Memory:', len(self.memory))
        if not self.memory_filled and len(self.memory) == self.memory_size:
            self.memory_filled = True
            print('-- Memory: filled')

    def get_batch(self, model, target_model, batch_size, gamma=0.9):
        if self.fast:
            return self.get_batch_fast(model, target_model, batch_size, gamma)
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)

        num_actions = model.get_output_shape_at(0)[-1]
        samples = numpy.array(sample(self.memory, batch_size))
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

    def reset_memory(self):
        self.memory.clear()

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

    def get_batch_fast(self, model, target_model, batch_size, gamma):
        if len(self.memory) < batch_size:
            return None
        samples = numpy.array(sample(self.memory, batch_size))
        if not hasattr(self, 'batch_function') or self.batch_function is None:
            self.set_batch_function(model, target_model, self.input_shape, batch_size,
                                    model.get_output_shape_at(0)[-1], gamma)
        state, targets = self.batch_function([samples])
        return state, targets
