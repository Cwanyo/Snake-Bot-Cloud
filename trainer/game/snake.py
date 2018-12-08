import pygame


class Snake:
    def __init__(self, window_width, window_height, pixel_size, init_x, init_y, body_size=1):
        self.window_width = window_width
        self.window_height = window_height
        self.pixel_size = pixel_size

        self.moves = [[-1, 0],  # 0 - left
                      [0, -1],  # 1 - up
                      [1, 0],  # 2 - right
                      [0, 1]]  # 3 - down

        self.forbidden_moves = [[0, 2], [2, 0],
                                [1, 3], [3, 1]]

        self.heading_direction = 1

        self.head_color = (0, 100, 0)
        self.body_color = (0, 200, 0)

        self.head = [init_x, init_y]
        self.body = []
        # Generate snake's body
        for i in range(1, body_size+1):
            self.body.append([self.head[0], self.head[1] + i])

    # Change direction with 0-4 index
    def change_direction(self, move_index):
        move = [self.heading_direction, move_index]

        if not (move in self.forbidden_moves):
            self.heading_direction = move_index
            return True
        else:
            return False

    # Move snake
    def move(self):
        self.body.insert(0, self.head)
        self.body.pop()
        self.head = [self.head[0] + self.moves[self.heading_direction][0],
                     self.head[1] + self.moves[self.heading_direction][1]]

    # Check whether snake collide with the food
    def collision_food(self, food_location):
        if self.head == food_location:
            self.body.insert(-1, self.body[-1])
            return True
        else:
            return False

    # Check whether snake collide with the wall or its body
    def collision_obstacles(self):
        if self.head[0] < 0 or self.head[0] >= self.window_width \
                or self.head[1] < 0 or self.head[1] >= self.window_height:
            return True

        for b in self.body:
            if self.head == b:
                return True

        return False

    # Get snake length (head + body)
    def get_length(self):
        return len(self.body) + 1

    # Render snake
    def render(self, win):
        for b in self.body:
            pygame.draw.rect(win, self.body_color,
                             (b[0] * self.pixel_size, b[1] * self.pixel_size, self.pixel_size, self.pixel_size))

        pygame.draw.rect(win, self.head_color,
                         (self.head[0] * self.pixel_size, self.head[1] * self.pixel_size, self.pixel_size,
                          self.pixel_size))
