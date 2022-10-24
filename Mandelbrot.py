from os import getenv

import PIL.ImageColor
import cv2
import numpy
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
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
        x_iteration = width / self.image_width
        y_iteration = height / self.image_height
        pool = mp.Pool(self.cores)
        x_interval = self.image_width
        y_interval = int(self.image_height / self.cores)
        """print(start_x, start_y, start_x + (x_iteration * self.image_width), start_y + (y_iteration * self.image_height), self.image_height, 0)
        self.array = self.calc_points(self.array, start_x, start_y, x_iteration, y_iteration, self.image_height, 0)"""
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
        for x in range(0, self.image_width):
            progress = (x / self.image_width) * 100
            # print("Progress: " + str(progress) + "%")
            real = start_x + (x_iteration * x)
            for y in range(y_min, y_max):
                imag = start_y + (y_iteration * y)
                t = self.is_out_of_bounds(complex(real, imag))
                if t != 0:
                    colorIndex = t % len(self.colors)
                    color = PIL.ImageColor.getrgb(self.colors[colorIndex])
                    print(color)
                    array[y, x] = color
                else:
                    color = PIL.ImageColor.getrgb('black')
                    array[y, x] = color
        return array

    @staticmethod
    def calc_window(start_x: float, end_x: float, start_y: float):
        width = end_x - start_x
        height = ((width / 16) * 9)
        end_y = start_y + height
        return start_x, end_x, start_y, end_y

    def generate_image(self, start_x: float, end_x: float, start_y: float):
        start_x, end_x, start_y, end_y = self.calc_window(start_x, end_x, start_y)
        self.generate_array(start_x, end_x, start_y, end_y)
        img = cv2.flip(cv2.cvtColor(self.get_image_array(), cv2.COLOR_RGB2BGR), 0)
        cv2.imwrite('test.jpg', img)

    def generate_gpu_image(self, start_x: float, end_x: float, start_y: float):
        start_x, end_x, start_y, end_y = self.calc_window(start_x, end_x, start_y)
        self.generate_gpu_array(start_x, end_x, start_y, end_y)
        img = cv2.flip(cv2.cvtColor(self.get_image_array(), cv2.COLOR_RGB2BGR), 0)
        cv2.imwrite('test.jpg', img)

    def generate_gpu_array(self, start_x: float, end_x: float, start_y: float, end_y: float):
        self.array = np.zeros(shape=(self.image_height, self.image_width, 3), dtype=np.uint8)
        width = end_x - start_x
        height = end_y - start_y
        pool = mp.Pool(self.cores)
        x_iteration = width / self.image_width
        y_iteration = height / self.image_height
        x_interval = self.image_width
        y_interval = int(self.image_height / self.cores)
        result = {}
        for i in range(0, self.cores):
            subarray = self.array[(i * y_interval):((i + 1) * y_interval), 0:x_interval]
            result[i] = pool.apply_async(self.calc_points, [subarray, start_x, start_y + (y_iteration * i * y_interval),
                                                            x_iteration, y_iteration, y_interval, 0])

        final_array = result[0].get()
        for j in range(1, self.cores):
            partial_array = result[j].get()
            final_array = np.vstack((final_array, partial_array))

        self.array = final_array
        pool.close()

    def cuda_points(self, array, start_x, start_y, x_iteration, y_iteration, y_max, y_min):
        shape = array.shape
        image_rgb = gpuarray.empty(shape, dtype=gpuarray.vec.uchar4)
        cuda.memcpy_htod(image_rgb.gpudata, array.data)
        fractal = gpuarray.empty(shape, dtype=np.uint8)
        mod = SourceModule("""
            #include <complex>
            __global__ void bulb(const uchar4* const rgbImage, const unsigned uchar4* const fractal,
                             float start_x, float start_y, float x_iteration, float y_iteration,
                             int y_min, int y_max, int image_width)
            {
                for (int x = 0; x < image_width; x++) {
                    float real = start_x + (x_iteration * x);
                    for (int y = y_min; y < y_max; y++) {
                        float imag = start_y + (y_iteration * y);
                        complex<double> mycomplex(real, imag);
                        int t = is_out_of_bounds(mycomplex);
                        if (t == 0) {
                            
                        }
                    }
                }
            }
            
            __global__ int is_out_of_bounds(complex<double> mycomplex) {
                complex<double> z(0,0);
                int t = 0
                for (int i = 0; i < 256; i++) {
                    z = z * z + mycomplex
                    t = i
                    if (abs(z) > 1000) {
                        break;
                    }
                }
                if (abs(z) > 2) {
                    return t
                }
                return 0
            }
        """)
        bulb = mod.get_function("bulb")
        bulb(image_rgb, fractal, start_x, start_y, x_iteration, y_iteration, y_min, y_max, self.image_width)
        return np.array(fractal.get(), dtype=np.uint8)

    def get_image_array(self):
        return self.array

    @staticmethod
    def is_out_of_bounds(cmplx):
        z = complex(0, 0)
        t = 0
        for i in range(0, 256):
            z = z ** 2 + cmplx
            t = i
            if abs(z) > 1000:
                break
        if abs(z) > 2:
            return t
        return 0


if __name__ == '__main__':
    Mandelbrot(3840, 2160, 16).generate_image(-0.04491, -0.04486, 0.98261)
    # Mandelbrot(3840, 2160, 16).generate_image(-0.08, 0, 0.96)
    # Mandelbrot(3840, 2160, 16).generate_image(-8 / 3, 8 / 3, -1.5)
