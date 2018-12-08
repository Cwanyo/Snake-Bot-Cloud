import numpy
import math
import pygame

from trainer.game.snake import Snake
from trainer.game.food import Food


class Agent:
    def __init__(self, code_id, log=False, visualization=False, fps=60, board_size=(10, 10, 20), random_body=False):
        self.code_id = code_id

        self.window_width = board_size[0]
        self.window_height = board_size[1]
        self.pixel_size = board_size[2]

        # random spawn
        body_size = numpy.random.randint(1, self.window_height - 1) if random_body else 1
        init_x = numpy.random.randint(self.window_width)
        init_y = numpy.random.randint(self.window_height - body_size)  # 1 is body size

        self.snake = Snake(self.window_width, self.window_height, self.pixel_size,
                           init_x, init_y, body_size)
        self.food = Food(self.window_width, self.window_height, self.pixel_size)
        self.food.spawn(self.snake)

        self.log = log
        self.visualization = visualization
        self.window, self.clock = self.init_visualization()
        self.fps = fps

        # basic infos
        self.alive = True
        self.score = 0
        self.pre_score = self.score
        self.step = 0
        # useful infos
        self.s_obstacles = self.get_surrounding_obstacles()
        self.food_angle = self.get_food_angle()
        self.pre_food_distance = 1
        self.food_distance = 1
        self.legal_action = True
        self.get_food_distance()
        self.board = self.get_board()
        self.reward = self.get_reward_v1()

    def init_visualization(self):
        if self.visualization:
            window = pygame.display.set_mode(
                (self.window_width * self.pixel_size, self.window_height * self.pixel_size))
            fps = pygame.time.Clock()
            return window, fps
        else:
            return None, None

    def get_random_legal_action(self):
        action_index = numpy.random.randint(len(self.snake.moves))
        move = [self.snake.heading_direction, action_index]

        while move in self.snake.forbidden_moves:
            action_index = numpy.random.randint(len(self.snake.moves))
            move = [self.snake.heading_direction, action_index]

        return action_index

    def filter_legal_action(self, q_value):
        curr_heading = self.snake.heading_direction

        for i in range(len(q_value[0])):
            if [curr_heading, i] in self.snake.forbidden_moves:
                q_value[0][i] = -1

        return q_value

    def next_state(self, move_index):
        self.step += 1
        info = 'CodeID: {} | Step: {} | Score: {}'.format(self.code_id, self.step, self.score)

        self.legal_action = self.snake.change_direction(move_index)
        self.snake.move()

        self.pre_score = self.score
        if self.snake.collision_food(self.food.location):
            self.score += 1
            self.food.state = False

        self.food.spawn(self.snake)

        if self.snake.collision_obstacles():
            info += ' >> Game Over!'
            self.alive = False

        if self.snake.get_length() == self.window_width * self.window_height:
            info += ' >> Win!'
            self.alive = False

        if self.log:
            print(info)

        if self.visualization:
            self.window.fill((0, 0, 0))
            self.food.render(self.window)
            self.snake.render(self.window)
            pygame.display.set_caption(info)
            pygame.display.update()
            pygame.event.get()
            self.clock.tick(self.fps)

        return self.get_state()

    def get_state(self):
        return self.get_surrounding_obstacles(), self.get_food_angle(), self.get_food_distance(), \
               self.get_board(), self.get_reward_v1()

    def get_surrounding_obstacles(self):
        # check front
        snake_head = self.snake.head
        snake_heading_direction = self.snake.heading_direction
        left = self.snake.moves[(snake_heading_direction - 1) % len(self.snake.moves)]
        front = self.snake.moves[snake_heading_direction]
        right = self.snake.moves[(snake_heading_direction + 1) % len(self.snake.moves)]
        l_location = [snake_head[0] + left[0], snake_head[1] + left[1]]
        f_location = [snake_head[0] + front[0], snake_head[1] + front[1]]
        r_location = [snake_head[0] + right[0], snake_head[1] + right[1]]

        s_locations = [l_location, f_location, r_location]
        self.s_obstacles = [0, 0, 0]

        # check wall
        for i in range(0, len(s_locations)):
            if s_locations[i][0] < 0 or s_locations[i][0] >= self.window_width \
                    or s_locations[i][1] < 0 or s_locations[i][1] >= self.window_height:
                self.s_obstacles[i] = 1

        # check body
        for b in self.snake.body:
            if b in s_locations:
                self.s_obstacles[s_locations.index(b)] = 1

        return self.s_obstacles

    def get_food_angle(self):
        # get direction of heading
        heading_direction = numpy.array(self.snake.moves[self.snake.heading_direction])
        # get direction of food (distant)
        food_direction = numpy.array(self.food.location) - numpy.array(self.snake.head)

        h = heading_direction / numpy.linalg.norm(heading_direction)
        f = food_direction / numpy.linalg.norm(food_direction)

        fa = math.atan2(h[0] * f[1] - h[1] * f[0], h[0] * f[0] + h[1] * f[1]) / math.pi

        if fa == -1 or fa == 1:
            fa = 1

        self.food_angle = fa

        return self.food_angle

    def get_food_distance(self):
        head = numpy.array(self.snake.head)
        food = numpy.array(self.food.location)

        max_dis = numpy.linalg.norm(numpy.array([0, 0]) - numpy.array([self.window_width - 1, self.window_height - 1]))
        dis = numpy.linalg.norm(head - food)

        # normalize distance to the range 0 - 1
        self.pre_food_distance = self.food_distance
        self.food_distance = dis / max_dis

        return self.food_distance

    def get_board(self):
        # TODO - change values
        # coordinate x,y are opposite to array => y,x
        temp_board = [[0] * (self.window_width + 2) for i in range(self.window_height + 2)]

        # mark top & bottom wall
        for i in range(len(temp_board[0])):
            temp_board[0][i] = 1
            temp_board[len(temp_board) - 1][i] = 1

            # mark left and right wall
        for i in range(len(temp_board)):
            temp_board[i][0] = 1
            temp_board[i][len(temp_board[0]) - 1] = 1

            # mark snake
        temp_board[int(self.snake.head[1]) + 1][int(self.snake.head[0]) + 1] = 1
        for b in self.snake.body:
            temp_board[int(b[1]) + 1][int(b[0]) + 1] = 1

            # mark food
        temp_board[int(self.food.location[1]) + 1][int(self.food.location[0]) + 1] = 0.5

        self.board = temp_board

        return self.board

    def get_board_v1(self):
        # TODO - change values
        # coordinate x,y are opposite to array => y,x
        temp_board = [[0] * (self.window_width + 2) for i in range(self.window_height + 2)]

        # mark top & bottom wall
        for i in range(len(temp_board[0])):
            temp_board[0][i] = -1
            temp_board[len(temp_board) - 1][i] = -1

            # mark left and right wall
        for i in range(len(temp_board)):
            temp_board[i][0] = -1
            temp_board[i][len(temp_board[0]) - 1] = -1

            # mark snake
        temp_board[int(self.snake.head[1]) + 1][int(self.snake.head[0]) + 1] = 0.5
        for b in self.snake.body:
            temp_board[int(b[1]) + 1][int(b[0]) + 1] = -1

            # mark food
        temp_board[int(self.food.location[1]) + 1][int(self.food.location[0]) + 1] = 1

        self.board = temp_board

        return self.board

    def get_reward(self):
        if not self.alive:
            self.reward = -1
        elif self.score > self.pre_score:
            self.reward = len(self.snake.head) + len(self.snake.body)
        else:
            self.reward = 0

        return self.reward

    def get_reward_v1(self):
        if not self.alive:
            # if snake is dead
            self.reward = -1
        elif not self.legal_action:
            # if action is illegal
            self.reward = -1
        elif self.score > self.pre_score:
            # if scored
            self.reward = 1
        elif self.food_distance != self.pre_food_distance:
            # if distance from s to s1 is change
            snake_len = len(self.snake.head) + len(self.snake.body)
            d1 = self.pre_food_distance
            d2 = self.food_distance
            self.reward = math.log((snake_len + d1) / (snake_len + d2), snake_len)
        else:
            self.reward = 0

        return self.reward
