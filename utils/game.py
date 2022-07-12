import pygame


class Player:
    """
    The player character with basic, acceleration-based movement
    Slows down exponentially when no action is present
    """
    def __init__(self, x, y, width, height, ax, window_size):
        self.x, self.y = x, y
        self.width, self.height = width, height
        self.rect = pygame.Rect(x, y, width, height)

        # This character can only move along the horizontal axis
        self.ax = ax  # acceleration
        self.vx = 0  # velocity

        # Used for boundary detection
        self.window = window_size

    def update_pos(self, clock, left, right):
        # TO-DO: frame independent movement

        acc = 0
        if left:
            acc -= self.ax
        elif right:
            acc += self.ax
        else:
            self.vx *= 0.9

        self.vx += acc
        self.x += self.vx

        # Boundary detection - make player bounce off wall?
        if self.x <= 0:
            self.x = 0
            self.vx = 0
        if self.x + self.width >= self.window[0]:
            self.x = self.window[0] - self.width
            self.vx = 0

        if self.vx < -6:
            self.vx = -6
        if self.vx > 6:
            self.vx = 6

        self.rect.update(self.x, self.y, self.width, self.height)
        return self.x, self.y


class Enemy:
    """
    Horizontal bars that move from one side of the screen to another
    (Theoretically the game would have enemies coming from all directions,
        and the player would have four-directional movement)
    """
    def __init__(self, x, y, width, height, window_size, dx=0, dy=1, color=[255, 90, 61]):
        self.x, self.y = x, y
        self.dx, self.dy = dx, dy
        self.width, self.height = width, height
        self.rect = pygame.Rect(x, y, width, height)

        self.color=color
        self.window_size = window_size

    def update_pos(self):
        self.x += self.dx
        self.y += self.dy

        # Boundary detection, destroy on impact
        if self.x > self.window_size[0] + 100 or self.x < -100:
            return None
        if self.y > self.window_size[1] + 100 or self.y < -100:
            return None

        self.rect.update(self.x, self.y, self.width, self.height)
        return self.x, self.y

    @staticmethod
    def create_enemies(row, type, window_size, speed=3):
        # Types are not fully implemented
        types = {
            0: [0, 1, 0, -50],
            1: [0, -1, 0, window_size[1] + 50],
            2: [1, 0, -50, 0],
            3: [-1, 0, -50, window_size[0] + 50]
        }

        # Instantiate enemies given a `row` (array)
        row_length = len(row)
        block_width = window_size[0] // row_length
        row.append(0)
        enemies = []

        w, h = 0, 50
        x, y = types[type][2:]
        prev = 0

        for i in row:
            if i == 1:
                w += block_width
            elif i == 0 and prev == 1:
                # Ending an enemy instance
                dx, dy = types[type][:2]
                enemies.append(Enemy(x, y, w, h, window_size, dx * speed, dy * speed))
                x += w + block_width
                w = 0
            else:
                x += block_width
            prev = i

        return enemies
