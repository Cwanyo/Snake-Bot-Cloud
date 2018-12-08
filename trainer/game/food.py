import pygame
import random

from trainer.game.snake import Snake


class Food:
    def __init__(self, window_width, window_height, pixel_size):
        self.window_width = window_width
        self.window_height = window_height
        self.pixel_size = pixel_size

        self.food_color = (255, 0, 0)

        self.state = False
        self.location = [-1, -1]

        self.freeLocation = []

    # Check free location on the board
    def check_free_location(self, snake: Snake):
        self.freeLocation.clear()

        free = [[0] * self.window_width for i in range(self.window_height)]
        free[int(snake.head[1])][int(snake.head[0])] = 1

        for b in snake.body:
            free[int(b[1])][int(b[0])] = 1

        for r in range(len(free)):
            for c in range(len(free[0])):
                if free[r][c] == 0:
                    self.freeLocation.append([c, r])

    # Spawn the food
    def spawn(self, snake: Snake):
        if not self.state:
            self.check_free_location(snake)
            # print('---- free len:', len(self.freeLocation))

            self.location = self.freeLocation[random.randrange(0, len(self.freeLocation))]
            # print('---- spawn at: ', self.location)
            self.state = True

    # Render food
    def render(self, win):
        if self.state:
            pygame.draw.rect(win, self.food_color,
                             (self.location[0] * self.pixel_size, self.location[1] * self.pixel_size, self.pixel_size,
                              self.pixel_size))
