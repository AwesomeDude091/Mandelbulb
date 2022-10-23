import PIL.ImageColor
from PIL import Image as im
import numpy as np
import multiprocessing as mp


class Mandelbrot:
    image_height: int

    def __init__(self, image_width: int, image_height: int, cores: int):
        self.image_width = image_width
        self.image_height = image_height
        self.cores = cores
        self.array = np.array([])
        self.colors = ['DarkBlue', 'DarkSlateBlue', 'DeepSkyBlue', 'LightGreen', 'Green', 'GreenYellow', 'Orange',
                       'OrangeRed', 'Red', 'Violet']

    def generate_array(self, start_x: float, end_x: float, start_y: float, end_y: float):
        self.array = np.zeros(shape=(self.image_height, self.image_width, 3), dtype=np.uint8)
        width = end_x - start_x
        height = end_y - start_y
        # print(width, height)
        x_iteration = width / self.image_width
        y_iteration = height / self.image_height
        # print(x_iteration, y_iteration)
        pool = mp.Pool(self.cores)
        x_interval = self.image_width
        y_interval = int(self.image_height / self.cores)
        result = {}
        for i in range(0, self.cores):
            # print("X: " + str(x_interval - 1), "Y: " + str(((i+1)*y_interval) - 1))
            subarray = self.array[(i * y_interval):((i + 1) * y_interval), 0:x_interval]
            result[i] = pool.apply_async(self.calc_points, [subarray, start_x, start_y + (y_iteration * i * y_interval),
                                                            x_iteration, y_iteration, y_interval, 0])

        final_array = result[0].get()
        for j in range(1, self.cores):
            partial_array = result[j].get()
            final_array = np.vstack((final_array, partial_array))

        self.array = final_array
        pool.close()

    def calc_points(self, array, start_x, start_y, x_iteration, y_iteration, y_max, y_min):
        print(start_x, start_y, x_iteration, y_iteration)
        for x in range(0, self.image_width):
            progress = (x / self.image_width) * 100
            # print("Progress: " + str(progress) + "%")
            real = start_x + (x_iteration * x)
            for y in range(y_min, y_max):
                imag = start_y + (y_iteration * y)
                is_out_of_bound, t = self.is_out_of_bounds(complex(real, imag))
                if is_out_of_bound:
                    colorIndex = t % len(self.colors)
                    color = PIL.ImageColor.getrgb(self.colors[colorIndex])
                    array[y, x] = color
                else:
                    color = PIL.ImageColor.getrgb('black')
                    array[y, x] = color
        return array

    def generate_image(self, start_x: float, end_x: float, start_y: float, end_y: float = None):
        if end_y is None:
            width = end_x - start_x
            height = ((width / 16) * 9)
            end_y = start_y + height
        self.generate_array(start_x, end_x, start_y, end_y)
        data = im.fromarray(self.array)
        data.save('test.jpg')

    def get_image_array(self):
        return self.array

    @staticmethod
    def is_out_of_bounds(cmplx):
        z = complex(0, 0)
        t = 0
        for i in range(0, 100):
            z = z ** 2 + cmplx
            t = i
            if abs(z) > 1000:
                break
        return abs(z) > 3, t

    @staticmethod
    def myRange(start, end, step, round_value=10000):
        i = start
        while i < end:
            yield i
            i = i + step
            i = round(i, round_value)
        yield end


if __name__ == '__main__':
    Mandelbrot(3840, 2160, 16).generate_image(-0.08, 1.05, 0.96)
    # Mandelbrot(3840, 2160, 16).generate_image(-8 / 3, 8 / 3, -1.5)
