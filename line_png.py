
from basic_test import Main

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Загружаем и изменяем размер изображения с помощью PIL
target_size = (980, 889)  # (ширина, высота)

import sys
if len(sys.argv[1:]):
    w1, w2 = map(int, sys.argv[1:])
else:
    w1, w2 = 9, 11

# Загружаем изображение
pil_img = Image.open('background.png')
pil_img_resized = pil_img.resize(target_size, Image.Resampling.LANCZOS)  # LANCZOS для лучшего качества
img = np.array(pil_img_resized)

# Получаем точки из вашей функции
points = Main(w1, w2).my_red_line()  # список точек [(x1,y1), (x2,y2), ...]
x_coords, y_coords = zip(*points)

# Отображаем изображение
plt.imshow(img)

# Рисуем линию поверх
plt.plot(x_coords, y_coords, 'r-', linewidth=2)  # 'r-' - красная линия

# Убираем оси
plt.axis('off')

# Сохраняем или показываем
plt.savefig('background_with_line.png', bbox_inches='tight', pad_inches=0)
plt.show()
