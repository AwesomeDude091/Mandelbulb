import multiprocessing as mp
import os
import time

import PIL.ImageColor
import cv2
import numpy
import numpy as np


class Mandelbrot:
    image_height: int

    def __init__(self, image_width: int, image_height: int, cores: int, max_iterations: int, debug: bool = False,
                 color_iteration: int = 8):
        self.image_width = image_width
        self.image_height = image_height
        self.cores = cores
        self.array = np.array([])
        self.color_iteration = color_iteration
        self.max_iterations = max_iterations
        self.debug = debug
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
        """print(start_x, start_y, start_x + (x_iteration * self.image_width), start_y + 
        (y_iteration * self.image_height), self.image_height, 0)
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
            if self.debug:
                print("Progress: " + str(progress) + "%")
            real = start_x + (x_iteration * x)
            for y in range(y_min, y_max):
                imag = start_y + (y_iteration * y)
                t = self.is_out_of_bounds(complex(real, imag))
                if t != 0:
                    colorIndex = int((int(t) / self.color_iteration) % len(self.colors))
                    color = PIL.ImageColor.getrgb(self.colors[colorIndex])
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
        os.environ['CUDA_PATH'] = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8'
        start_x, end_x, start_y, end_y = self.calc_window(start_x, end_x, start_y)
        self.generate_gpu_array(start_x, end_x, start_y, end_y)
        img = cv2.flip(cv2.cvtColor(self.get_image_array(), cv2.COLOR_RGB2BGR), 0)
        cv2.imwrite('test_gpu.jpg', img)

    def generate_gpu_array(self, start_x: float, end_x: float, start_y: float, end_y: float):
        self.array = np.zeros(shape=(self.image_height, self.image_width, 3), dtype=np.uint8)
        width = end_x - start_x
        height = end_y - start_y
        x_iteration = width / self.image_width
        y_iteration = height / self.image_height
        self.array = self.cuda_points(self.array, start_x, start_y, x_iteration, y_iteration, self.image_height, 0)

    def cuda_points(self, array, start_x, start_y, x_iteration, y_iteration, y_max, y_min):
        import pycuda.driver as cuda
        import pycuda.gpuarray as gpuarray
        from pycuda.compiler import SourceModule
        if os.system("cl.exe"):
            os.environ[
                'PATH'] += ';' + r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\
                14.33.31629\bin\Hostx86\x64"
        if os.system("cl.exe"):
            raise RuntimeError("cl.exe still not found, path probably incorrect")
        shape = array.shape
        image_rgb = gpuarray.empty(shape, dtype=gpuarray.vec.uchar4)
        cuda.memcpy_htod(image_rgb.gpudata, array.data)
        fractal = gpuarray.empty(shape, dtype=np.uint8)
        mod = SourceModule("""
            #include <thrust/complex.h>
            
            extern "C" {
                __device__ int is_out_of_bounds(thrust::complex<float> c) 
                {
                    thrust::complex<float> z = thrust::complex<float>(0.0f, 0.0f);
                    int runs = 0;
                    for (int i = 0; i < 256; i++) {
                        z = z * z + c;
                        runs = i;
                        if (thrust::abs(z) > 1000) {
                            break;
                        }
                    }
                    if (thrust::abs(z) > 2) {
                        return runs;
                    }
                    return 0;
                }
                
                __global__ void bulb (const uchar4* const rgbImage, uchar4* const fractal, int numRows, int numCols,
                                 float start_x, float start_y, float x_iteration, float y_iteration,
                                 int y_min, int y_max, int image_width)
                {
                    int x = threadIdx.x + blockIdx.x * blockDim.x;
                    int y = threadIdx.y + blockIdx.y * blockDim.y;
                    if (y < numCols && x < numRows) {
                        int index = numRows*y + x;
                        float real = start_x + (x_iteration * x);
                        float imag = start_y + (y_iteration * y);
                        thrust::complex<float> c = thrust::complex<float>(real, imag);
                        int runs = is_out_of_bounds(c);
                        uchar4 color;
                        if (runs == 0) {
                            color = make_uchar4(0.0f, 0.0f, 0.0f, 0.0f);
                        } else {
                            color = make_uchar4(255.0f, 255.0f, 255.0f, 255.0f);
                        }
                        fractal[index] = color;
                    }    
                }
            }    
        """, no_extern_c=True)
        func = mod.get_function("bulb")
        func(image_rgb, fractal, np.int32(shape[0]), np.int32(shape[1]), numpy.float32(start_x), numpy.float32(start_y),
             numpy.float32(x_iteration), numpy.float32(y_iteration), np.int32(y_min), np.int32(y_max),
             np.int32(self.image_width), block=(1024, 1, 1))
        return np.array(fractal.get(), dtype=np.uint8)

    def get_image_array(self):
        return self.array

    def is_out_of_bounds(self, cmplx):
        z = complex(0, 0)
        t = 0
        for i in range(0, self.max_iterations):
            z = z ** 2 + cmplx
            t = i
            if abs(z) > 1000:
                break

        if abs(z) >= 2:
            return t
        return 0


if __name__ == '__main__':
    clock = time.time()
    # Mandelbrot(3840, 2160, 16).generate_gpu_image(-8 / 3, 8 / 3, -1.5)
    # Mandelbrot(3840, 2160, 16).generate_image(-0.04491, -0.04486, 0.98261)
    # Mandelbrot(3840, 2160, 16).generate_image(-0.08, 0, 0.96)
    Mandelbrot(3840, 2160, 16, 256, debug=True, color_iteration=8).generate_image(-8 / 3, 8 / 3, -1.5)
    print("Time to complete: {}".format(time.time() - clock))
