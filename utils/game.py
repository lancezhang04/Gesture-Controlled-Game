class Player:
    def __init__(self, x, y, width, height, ax, window):
        self.x, self.y = x, y
        self.width, self.height = width, height

        # This character can only move along the horizontal axis
        self.ax = ax  # acceleration
        self.vx = 0  # velocity

        # Used for boundary detection
        self.window = window

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

        return self.x, self.y
