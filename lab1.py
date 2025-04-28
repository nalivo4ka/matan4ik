import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def adaptive_partition(func, a, b, n):

    n += 1 # Т.к n точек образуют n - 1 отрезок
    x_dense = np.linspace(a, b, max(5 * n, 1000))
    y_dense = func(x_dense)
    dy_dense = np.abs(np.gradient(y_dense, x_dense))
    weights = dy_dense / np.sum(dy_dense)
    
    cdf = np.cumsum(weights)
    x_adaptive = np.interp(np.linspace(0, 1, n), cdf, x_dense)

    partition = np.vstack((x_adaptive[:-1], x_adaptive[1:])).T

    return partition

def rect_method(func, partition):
    results = {
        'left': 0.0,
        'right': 0.0,
        'middle': 0.0,
        'random': 0.0
    }

    for a, b in partition:
        length = b - a
        results['left'] += func(a) * length
        results['right'] += func(b) * length
        results['middle'] += func((a + b) / 2) * length
        results['random'] += func(np.random.uniform(a, b)) * length

    return results

def trapez_method(func, partition):
    result = 0.0

    for a, b in partition:
        length = b - a
        result += (func(a) + func(b)) * length / 2

    return result

def simpson_method(func, partition):
    result = 0.0

    for a, b in partition:
        length = b - a
        result += (func(a) + 4 * func((a + b) / 2) + func(b)) * length / 6

    return result

def draw_rect_areas(func, a, b, n, ax = None):
    
    if not ax:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
    
    partition = adaptive_partition(func, a, b, n)

    x = np.linspace(a - 0.2, b + 0.2, 1000)
    y = func(x)
    ax.plot(x, y, color='blue', label=r'2^x')
    ax.set_aspect('equal', adjustable='datalim')
    ax.axvline(0, color='black', linestyle='--')
    ax.axhline(0, color='black', linestyle='--')
    ax.vlines(x=[a, b], ymin=[0, 0], ymax=[func(a), func(b)], color='red', linewidth=0.5, alpha=0.5)
    ax.grid()
    ax.set_title(f"Area = {rect_method(func, partition)['middle']:.4f}, n = {n}", fontsize=9)

    for leftBound, rightBound in partition:
        rect = patches.Rectangle((leftBound, 0), rightBound - leftBound, func((leftBound + rightBound) / 2), facecolor='green', alpha=0.3, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)

def task3(func, a, b):
    fig = plt.figure(figsize=(15, 15))

    ns = np.power(2, np.arange(3) + 2)
    row_count = int(np.ceil(np.sqrt(len(ns))))
    for i, n in enumerate(ns):
        ax = fig.add_subplot(row_count, row_count, i + 1)
        draw_rect_areas(func, a, b, n, ax)

    plt.show()

def task4(func, a, b, n):
    partition = adaptive_partition(func, a, b, n)
    real_area = 1 / np.log(2)

    print(f"===== ЗНАЧЕНИЯ ПРИ n = {n} =====\n"
          f"===== РЕАЛЬНОЕ ЗНАЧЕНИЕ: {real_area:.5f} =====\n"
          "1) Метод прямоугольнико:\n"
          f"\t1.1 По центру: значение = {rect_method(func, partition)['middle']:.5f}, MAE = {np.abs(real_area - rect_method(func, partition)['middle']):.5f}\n"
          f"\t1.2 По левой границе: значение = {rect_method(func, partition)['left']:.5f}, MAE = {np.abs(real_area - rect_method(func, partition)['left']):.5f}\n"
          f"\t1.3 По правой границе: значение = {rect_method(func, partition)['right']:.5f}, MAE = {np.abs(real_area - rect_method(func, partition)['right']):.5f}\n"
          f"\t1.4 Рандом: значение = {rect_method(func, partition)['random']:.5f}, MAE = {np.abs(real_area - rect_method(func, partition)['random']):.5f}\n"
          f"2) Метод трапеций: значение = {trapez_method(func, partition):.5f}, MAE = {np.abs(real_area - trapez_method(func, partition)):.5f}\n"
          f"3) Метод Симпсона: значение = {simpson_method(func, partition):.5f}, MAE = {np.abs(real_area - simpson_method(func, partition)):.5f}\n"
    )

def task5(func, a, b):
    ns = np.power(2, np.arange(8))
    print(f"------- REAL: {1 / np.log(2):.5f} -------")
    for n in ns:
        partition = adaptive_partition(func, a, b, n)
        print(f"n = {n}:\n"
              f"rectangle method (on middle): {rect_method(func, partition)['middle']:.5f}\n"
              f"trapez method: {trapez_method(func, partition):.5f}\n"
              f"simpson's method: {simpson_method(func, partition):.5f}\n"
        )

def task6(func, a, b, max_n):
    ns = np.arange(max_n) + 1

    y_mae_rect, y_mse_rect = [], []
    y_mae_trapez, y_mse_trapez = [], []
    y_mae_simpson, y_mse_simpson = [], []
    real_area = 1 / np.log(2)

    for n in ns:
        partition = adaptive_partition(func, a, b, n)

        area_rect_on_middle = rect_method(func, partition)['middle']
        y_mae_rect.append(np.abs(real_area - area_rect_on_middle))
        y_mse_rect.append(np.power(real_area - area_rect_on_middle, 2))

        area_trapez = trapez_method(func, partition)
        y_mae_trapez.append(np.abs(real_area - area_trapez))
        y_mse_trapez.append(np.power(real_area - area_trapez, 2))

        area_simpson = simpson_method(func, partition)
        y_mae_simpson.append(np.abs(real_area - area_simpson))
        y_mse_simpson.append(np.power(real_area - area_simpson, 2))


    plt.figure(figsize=(12, 8))
    plt.plot(ns, y_mae_rect, label='MAE', color='blue')
    plt.plot(ns, y_mse_rect, label='MSE', color='red')
    plt.title('Метод прямоугольник (по центру)')
    plt.xlabel('Количество отрезков')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.grid()

    plt.figure(figsize=(12, 8))
    plt.plot(ns, y_mae_trapez, label='MAE', color='blue')
    plt.plot(ns, y_mse_trapez, label='MSE', color='red')
    plt.title('Метод трапеций')
    plt.xlabel('Количество отрезков')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.grid()

    plt.figure(figsize=(12, 8))
    plt.plot(ns, y_mae_simpson, label='MAE', color='blue')
    plt.plot(ns, y_mse_simpson, label='MSE', color='red')
    plt.title('Метод Симпсона')
    plt.xlabel('Количество отрезков')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.grid()

    plt.show()


def main():
    func = lambda x: np.power(2, x)
    a, b = 0, 1

    task3(func, a, b)
    task4(func, a, b, 10)
    task5(func, a, b)
    task6(func, a, b, 300)

if __name__ == "__main__":
    main()