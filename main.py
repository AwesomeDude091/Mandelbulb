import collections
import fractions
import math
import time

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp


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
    for i in range(0, 100):
        z = z ** 2 + cmplx
        if abs(z) > 3:
            break
    return abs(z) > 3


def calc_points(start_x, end_x, start_y, end_y, step):
    black_x = np.array([])
    black_y = np.array([])
    red_x = np.array([])
    red_y = np.array([])
    interval = (end_x - start_x) / step
    counter = 0
    for x in myRange(start_x, end_x, step):
        progress = (counter / interval) * 100
        print("Progress: " + str(progress) + "%")
        counter += 1
        for y in myRange(start_y, end_y, step):
            w = complex(x, y)
            if is_out_of_bounds(w):
                red_x = np.append(red_x, x)
                red_y = np.append(red_y, y)
            else:
                black_x = np.append(black_x, x)
                black_y = np.append(black_y, y)

    return black_x, black_y, red_x, red_y


def parallel_processing(x_start, x_end, y_start, y_end, cpu_cores, depth, clarity):
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
    if square:
        i = 0
        for y in range(0, int(math.sqrt(cpu_cores))):
            for x in range(0, int(math.sqrt(cpu_cores))):
                result[i] = pool.apply_async(calc_points, [x_start + x * x_interval,
                                                           x_start + (x+1) * x_interval,
                                                           y_start + y * y_interval,
                                                           y_start + (y+1) * y_interval, depth])
                i += 1
    else:
        for i in range(0, cpu_cores):
            result[i] = pool.apply_async(calc_points, [x_start + i * x_interval,
                                                       x_start + (i + 1) * x_interval,
                                                       y_start,
                                                       y_start + y_interval, depth])
    for j in range(0, cpu_cores):
        black_x[j], black_y[j], red_x[j], red_y[j] = result[j].get()

    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.set_title('Mandelbrot Menge', fontsize=10)
    ax.set_xlabel('Reele Zahlen')
    ax.set_ylabel('Imaginäre Zahlen')

    for k in range(0, cpu_cores):
        for p in range(0, clarity):
            ax.scatter(black_x[k], black_y[k], s=depth, color='black')
            ax.scatter(red_x[k], red_y[k], s=depth, color='red')

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
    parallel_processing(-2.5, 2.5, -1.5, 1.5, cpu_cores=4, depth=0.005, clarity=8)
