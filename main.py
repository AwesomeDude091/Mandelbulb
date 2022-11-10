import collections
import fractions
import math
import time
from os import getenv, putenv

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from PIL import Image as im


def get_prime_factors(n):
    # http://stackoverflow.com/a/22808285/5393381
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def has_irrational_square_root(num):
    # http://stackoverflow.com/questions/42302488/identify-a-irrational-or-complex-number
    frac = fractions.Fraction(str(num))
    top_primes = get_prime_factors(frac.numerator)
    bottom_primes = get_prime_factors(frac.denominator)
    all_even_top = all(num % 2 == 0 for num in collections.Counter(top_primes).values())
    all_even_bottom = all(num % 2 == 0 for num in collections.Counter(bottom_primes).values())
    if all_even_top and all_even_bottom:
        return False
    return True


def myRange(start, end, step, round_value=3):
    i = start
    while i < end:
        yield i
        i = i + step
        i = round(i, round_value)
    yield end


def is_out_of_bounds(cmplx):
    z = complex(0, 0)
    t = 0
    for i in range(0, 100):
        z = z ** 2 + cmplx
        t = i
        if abs(z) > 100:
            break
    return abs(z) > 3, t


def calc_points(start_x, end_x, start_y, end_y, step, round_value):
    black_x = np.array([])
    black_y = np.array([])
    red_x = np.array([])
    red_y = np.array([])
    navy_x = np.array([])
    navy_y = np.array([])
    blue_x = np.array([])
    blue_y = np.array([])
    green_x = np.array([])
    green_y = np.array([])
    yellow_x = np.array([])
    yellow_y = np.array([])
    orange_x = np.array([])
    orange_y = np.array([])
    interval = (end_x - start_x) / step
    counter = 0
    for x in myRange(start_x, end_x, step, round_value):
        progress = (counter / interval) * 100
        print("Progress: " + str(progress) + "%")
        counter += 1
        for y in myRange(start_y, end_y, step, round_value):
            w = complex(x, y)
            is_out_of_bound, t = is_out_of_bounds(w)
            if is_out_of_bound:
                if t <= 16:
                    navy_x = np.append(navy_x, x)
                    navy_y = np.append(navy_y, y)
                elif t <= 32:
                    blue_x = np.append(blue_x, x)
                    blue_y = np.append(blue_y, y)
                elif t <= 48:
                    green_x = np.append(green_x, x)
                    green_y = np.append(green_y, y)
                elif t <= 64:
                    yellow_x = np.append(yellow_x, x)
                    yellow_y = np.append(yellow_y, y)
                elif t <= 80:
                    orange_x = np.append(orange_x, x)
                    orange_y = np.append(orange_y, y)
                else:
                    red_x = np.append(red_x, x)
                    red_y = np.append(red_y, y)
            else:
                black_x = np.append(black_x, x)
                black_y = np.append(black_y, y)

    return black_x, black_y, red_x, red_y, navy_x, navy_y, blue_x, blue_y, green_x, green_y, yellow_x, yellow_y, \
           orange_x, orange_y


def parallel_processing(x_start, x_end, y_start, y_end, cpu_cores, depth, clarity, round_value):
    now = time.time()
    x_length = x_end - x_start
    y_length = y_end - y_start
    square = False
    if not has_irrational_square_root(cpu_cores):
        x_interval = int(math.sqrt(cpu_cores)) * x_length / cpu_cores
        y_interval = int(math.sqrt(cpu_cores)) * y_length / cpu_cores
        square = True
    else:
        x_interval = x_length / cpu_cores
        y_interval = y_length

    pool = mp.Pool(cpu_cores)
    result = {}
    black_x = {}
    black_y = {}
    red_x = {}
    red_y = {}
    navy_x = {}
    navy_y = {}
    blue_x = {}
    blue_y = {}
    green_x = {}
    green_y = {}
    yellow_x = {}
    yellow_y = {}
    orange_x = {}
    orange_y = {}
    if square:
        i = 0
        for y in range(0, int(math.sqrt(cpu_cores))):
            for x in range(0, int(math.sqrt(cpu_cores))):
                result[i] = pool.apply_async(calc_points, [x_start + x * x_interval,
                                                           x_start + (x + 1) * x_interval,
                                                           y_start + y * y_interval,
                                                           y_start + (y + 1) * y_interval, depth, round_value])
                i += 1
    else:
        for i in range(0, cpu_cores):
            result[i] = pool.apply_async(calc_points, [x_start + i * x_interval,
                                                       x_start + (i + 1) * x_interval,
                                                       y_start,
                                                       y_start + y_interval, depth, round_value])
    for j in range(0, cpu_cores):
        black_x[j], black_y[j], red_x[j], red_y[j], navy_x[j], navy_y[j], blue_x[j], blue_y[j], green_x[j], green_y[j], \
        yellow_x[j], yellow_y[j], orange_x[j], orange_y[j] = result[j].get()

    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.set_title('Mandelbrot Menge', fontsize=10)
    ax.set_xlabel('Reele Zahlen')
    ax.set_ylabel('Imaginäre Zahlen')

    for k in range(0, cpu_cores):
        for p in range(0, clarity):
            ax.scatter(black_x[k], black_y[k], s=depth, color='black')
            ax.scatter(red_x[k], red_y[k], s=depth, color='red')
            ax.scatter(navy_x[k], navy_y[k], s=depth, color='navy')
            ax.scatter(blue_x[k], blue_y[k], s=depth, color='lightblue')
            ax.scatter(green_x[k], green_y[k], s=depth, color='green')
            ax.scatter(yellow_x[k], yellow_y[k], s=depth, color='hotpink')
            ax.scatter(orange_x[k], orange_y[k], s=depth, color='yellow')

    pool.close()
    print("Time to calculate: {}".format(time.time() - now))
    fig.tight_layout()
    plt.show()


def almond_bread():
    plt.xlabel('Reele Zahlen')
    plt.ylabel('Imaginäre Zahlen')
    for x in myRange(-2.5, 2.5, 0.01):
        for y in myRange(-1.5, 1.5, 0.01):
            w = complex(x, y)
            print(w)
            if is_out_of_bounds(w):
                plt.scatter(w.real, w.imag, s=0.015, color='black', alpha=1)
            else:
                plt.scatter(w.real, w.imag, s=0.015, color='red', alpha=1)

    plt.show()


if __name__ == '__main__':
    # almond_bread()
    parallel_processing(-8/3, 8/3, -1.5, 1.5, cpu_cores=16, depth=0.005, clarity=8, round_value=10)
