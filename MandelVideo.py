import cv2
import numpy

from Mandelbrot import Mandelbrot


class MandelVideo:

    def __init__(self, start_left_x, start_right_x, start_y, end_left_x, end_right_x, end_y, fps: int, duration: int,
                 log: bool, cores):
        self.start_left_x = start_left_x
        self.start_right_x = start_right_x
        self.start_y = start_y
        self.start_right_y = start_y + ((start_right_x - start_left_x)/16)*9
        self.end_x = end_left_x
        self.end_right_x = end_right_x
        self.end_y = end_y
        self.end_right_y = end_y + ((self.end_right_x - end_left_x)/16)*9
        self.fps: int = fps
        self.duration: int = duration
        self.log: bool = log
        self.cores = cores
        # print(self.start_left_x, self.start_right_x, self.start_y, self.start_right_y)
        # print(end_left_x, self.end_right_x, end_y, self.end_right_y)

    def generate_video(self, color_iteration):
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter('mandel.mp4', fourcc, self.fps, (3840, 2160))
        bread = Mandelbrot(3840, 2160, self.cores, 256, color_iteration=color_iteration)
        total_frames = (self.fps * self.duration)
        if self.log:
            progression_left_x = (self.end_x - self.start_left_x) / (numpy.log(total_frames))
            progression_right_x = (self.end_right_x - self.start_right_x) / (numpy.log(total_frames))
            progression_left_y = (self.end_y - self.start_y) / (numpy.log(total_frames))
        else:
            progression_left_x = (self.end_x - self.start_left_x) / (total_frames - 1)
            progression_right_x = (self.end_right_x - self.start_right_x) / (total_frames - 1)
            progression_left_y = (self.end_y - self.start_y) / (total_frames - 1)
        for i in range(0, total_frames):
            progress = i * 100 / total_frames
            print("Progress: " + str(progress) + " %")
            if self.log:
                current_left = progression_left_x * numpy.log(i + 1) + self.start_left_x
                current_right = progression_right_x * numpy.log(i + 1) + self.start_right_x
                current_bottom = progression_left_y * numpy.log(i + 1) + self.start_y
            else:
                current_left = self.start_left_x + (i * progression_left_x)
                current_right = self.start_right_x + (i * progression_right_x)
                current_bottom = self.start_y + (i * progression_left_y)
            bread.generate_image(current_left, current_right, current_bottom)
            writer.write(cv2.flip(cv2.cvtColor(bread.get_image_array(), cv2.COLOR_RGB2BGR), 0))

        writer.release()


if __name__ == '__main__':
    # Mandelbrot(3840, 2160, 16).generate_image(-0.04491, -0.04486, 0.98261)
    # Mandelbrot(3840, 2160, 16).generate_image(-8 / 3, 8 / 3, -1.5)
    MandelVideo(-(8/3), 8/3, -1.5, -0.04491, -0.04486, 0.98261, fps=24, duration=60, log=True, cores=16)\
        .generate_video(8)
