import pygame


window_size = [500, 500]
delay = 1
player_width, player_height = 50, 50
player_acc = 2  # divided by fps -> doesn't actually work
player_color = [203, 96, 21]

pygame.init()
window = pygame.display.set_mode(window_size)
pygame.display.set_caption('Prototype')
clock = pygame.time.Clock()


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
        acc = 0
        if left:
            acc -= self.ax / clock.get_fps()
        if right:
            acc += self.ax / clock.get_fps()

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


run = True
draw_fps = False
font = pygame.font.SysFont(None, 16)
player = Player(
    x=window_size[0] // 2 - player_width // 2,
    y=window_size[1] // 2 - player_height // 2,
    width=player_width, height=player_height,
    ax=player_acc, window=window_size
)

while run:
    pygame.time.delay(delay)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            break
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                draw_fps = not draw_fps

    # Handle key presses
    keys = pygame.key.get_pressed()
    left_pressed, right_pressed = False, False
    if keys[pygame.K_LEFT]:
        left_pressed = True
    if keys[pygame.K_RIGHT]:
        right_pressed = True

    # Update player position
    x, y = player.update_pos(clock, left_pressed, right_pressed)

    # Render game
    window.fill((255, 255, 255))
    pygame.draw.rect(window, player_color, (x, y, player.width, player_height))
    if draw_fps:
        text = 'FPS: ' + str(int(clock.get_fps()))
        fps_image = font.render(text, True, (255, 255, 255), (0, 0, 0))
        window.blit(fps_image, (20, 20))

    pygame.display.update()
    clock.tick()


pygame.quit()
