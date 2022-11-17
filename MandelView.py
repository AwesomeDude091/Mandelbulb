import cv2
import pygame
from PIL import Image
from screeninfo import get_monitors

from Mandelbrot import Mandelbrot


class MandelView:
    def __init__(self, width, height, cores, iterations, color_iterations):
        self.width = width
        self.height = height
        self.cores = cores
        self.iterations = iterations
        self.color_iterations = color_iterations
        self.start_x = -8 / 3
        self.end_x = 8 / 3
        self.start_y = -1.5
        self.end_y = 1.5
        self.window_x = -8/3
        self.window_x_end = 8/3
        self.window_y = -1.5
        self.window_y_end = 1.5
        self.img = None

    def start(self):
        index = 0
        monitor = None
        for m in get_monitors():
            if m.is_primary:
                monitor = m
        brot = Mandelbrot(self.width, self.height, self.cores, self.iterations, debug=True,
                          color_iteration=self.color_iterations)
        brot.generate_image(-8/3, 8/3, -1.5)
        data = brot.get_image_array()
        tempImg = Image.fromarray(data)
        tempImg = tempImg.resize((monitor.width, monitor.height))
        self.img = tempImg
        pygame.init()
        screen = pygame.display.set_mode((monitor.width, monitor.height), pygame.FULLSCREEN)
        pygame.mouse.set_visible(True)
        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_CROSSHAIR)
        pygame.display.set_caption('Mandelbulb')
        screen.blit(self.pilImageToSurface(tempImg), (0, 0))
        pygame.image.save(screen, str(index) + '.png')
        pygame.display.flip()
        first_click = True
        image_ready = False
        pygame.time.Clock()
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    print("Down")
                    if first_click:
                        screen.blit(self.pilImageToSurface(self.img), (0, 0))
                        pygame.display.flip()
                        self.start_x, self.start_y = pygame.mouse.get_pos()
                        first_click = False
                        image_ready = False
                    else:
                        self.end_x, self.end_y = pygame.mouse.get_pos()
                        height = (self.end_x - self.start_x) / (16 / 9)
                        local = (self.end_y + self.start_y) / 2
                        self.end_y = local + height / 2
                        self.start_y = local - height / 2
                        pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(self.start_x, self.start_y,
                                                                              (self.end_x - self.start_x),
                                                                              (self.end_y - self.start_y)), 2)
                        pygame.display.flip()
                        first_click = True
                        image_ready = True
                elif e.type == pygame.KEYDOWN:
                    if pygame.key.name(e.key) == 'return' and image_ready:
                        pygame.image.save(screen, str(index) + '-rect.png')
                        self.img = self.give_game_image(monitor)
                        screen.blit(self.pilImageToSurface(self.img), (0, 0))
                        pygame.display.flip()
                        index += 1
                        pygame.image.save(screen, str(index) + '.png')
                        image_ready = False
                    elif pygame.key.name(e.key) == 'q':
                        pygame.quit()

    def set_iterations(self, iterations):
        self.iterations = iterations

    def give_game_image(self, monitor):
        brot = Mandelbrot(self.width, self.height, self.cores, self.iterations, debug=True,
                          color_iteration=self.color_iterations)
        real_start_x = ((self.start_x * (self.width / monitor.width) *
                         (self.window_x_end - self.window_x)) / self.width) + self.window_x
        real_end_x = ((self.end_x * (self.width / monitor.width) *
                       (self.window_x_end - self.window_x)) / self.width) + self.window_x
        real_start_y = ((self.start_y * (self.height / monitor.height) *
                         (self.window_y_end - self.window_y)) / self.height) + self.window_y
        real_end_y = ((self.end_y * (self.height / monitor.height) *
                      (self.window_y_end - self.window_y)) / self.height) + self.window_y
        brot.generate_array(real_start_x, real_end_x, real_start_y, real_end_y)
        data = brot.get_image_array()
        tempImg = Image.fromarray(data)
        self.window_x = real_start_x
        self.window_x_end = real_end_x
        self.window_y = real_start_y
        self.window_y_end = real_end_y
        print("Area: ")
        print(real_start_x, real_end_x, real_start_y)
        return tempImg.resize((monitor.width, monitor.height))

    @staticmethod
    def pilImageToSurface(pilImage):
        return pygame.image.fromstring(
            pilImage.tobytes(), pilImage.size, pilImage.mode).convert()


if __name__ == '__main__':
    MandelView(3840, 2160, 24, 16384, 1).start()
