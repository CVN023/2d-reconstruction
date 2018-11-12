from PIL import Image, ImageFilter
from skimage import io, img_as_float
import numpy as np
import pickle

image_int = io.imread('Mount_Bromo.jpg')
image = img_as_float(image_int)
img_size = image.shape

green_img = image[:, :, 1].copy()
red_img = image[:, :, 0].copy()
blue_img = image[:, :, 2].copy()
print(green_img.shape)

original_list = [red_img, green_img, blue_img]


A = np.random.rand(200, 245)
with open('A_matrix', 'wb') as file:
    my_pickler = pickle.Pickler(file)
    my_pickler.dump(A)


with open('A_matrix', 'rb') as file:
    my_unpickler = pickle.Unpickler(file)
    A = my_unpickler.load()
    
def f(x):
    global A
    return np.dot(A, x)

def cost_function(x, y):
    return np.linalg.norm(f(x)-y)
    
def grad_cost_function(x, y):
    global A
    return (np.dot(np.transpose(A), f(x)) - np.dot(np.transpose(A), y))

def tv(x):
    var = 0
    for i in range(len(x)-1):
        for j in range(len(x[0])-1):
            var = var + abs(x[i][j] - x[i+1][j]) + abs(x[i][j] - x[i][j+1])
    return var

def cost_function_tv(x, y):
    return np.linalg.norm(f(x)-y) + 0.001 * tv(x)
    
def grad_cost_function_tv(x, y):
    global A
    delta_tv = np.ones(green_img.shape, dtype=np.uint8)
    for i in range(len(x)-1):
        for j in range(len(x[0])-1):
            if i!= 0 and j != 0:
                delta = 0
                if x[i+1][j] > x[i][j]:
                    delta += 1
                else:
                    delta += -1
                if x[i][j+1] > x[i][j]:
                    delta += 1
                else:
                    delta += -1
                if x[i][j] > x[i-1][j]:
                    delta += 1
                else:
                    delta += -1
                if x[i][j] > x[i][j-1]:
                    delta += 1
                else:
                    delta += -1
                delta_tv[i][j] = delta
    for i in range(len(x)-1):
        delta = 0
        j = len(x[0])-1
        if x[i+1][j] > x[i][j]:
            delta += 1
        else:
            delta += -1
        if x[i][j] > x[i-1][j]:
            delta += 1
        else:
            delta += -1
        if x[i][j] > x[i][j-1]:
            delta += 1
        else:
            delta += -1
        delta_tv[i][j] = delta
        
    for j in range(len(x[0])-1):
        delta = 0
        i = len(x)-1
        if x[i][j+1] > x[i][j]:
            delta += 1
        else:
            delta += -1
        if x[i][j] > x[i-1][j]:
            delta += 1
        else:
            delta += -1
        if x[i][j] > x[i][j-1]:
            delta += 1
        else:
            delta += -1
        delta_tv[i][j] = delta
        
    for i in range(1, len(x)-1):
        delta = 0
        j = 0
        if x[i+1][j] > x[i][j]:
            delta += 1
        else:
            delta += -1
        if x[i][j+1] > x[i][j]:
            delta += 1
        else:
            delta += -1
        if x[i][j] > x[i-1][j]:
            delta += 1
        else:
            delta += -1
        delta_tv[i][j] = delta
        
    for j in range(1, len(x[0])-1):
        delta = 0
        i = 0
        if x[i+1][j] > x[i][j]:
            delta += 1
        else:
            delta += -1
        if x[i][j+1] > x[i][j]:
            delta += 1
        else:
            delta += -1
        if x[i][j] > x[i][j-1]:
            delta += 1
        else:
            delta += -1
        delta_tv[i][j] = delta
    
    i = len(x) - 1
    j = len(x[0]) - 1
    delta = 0
    if x[i][j] > x[i-1][j]:
        delta += 1
    else:
        delta += -1
    if x[i][j] > x[i][j-1]:
        delta += 1
    else:
        delta += -1
    delta_tv[i][j] = delta
    
    i = 0
    j = 0
    delta = 0
    if x[i+1][j] > x[i][j]:
        delta += 1
    else:
        delta += -1
    if x[i][j+1] > x[i][j]:
        delta += 1
    else:
        delta += -1
    delta_tv[i][j] = delta
    
    i = 0
    j = len(x[0]) - 1
    delta = 0
    if x[i+1][j] > x[i][j]:
        delta += 1
    else:
        delta += -1
    if x[i][j] > x[i][j-1]:
        delta += 1
    else:
        delta += -1
    delta_tv[i][j] = delta
    
    i = len(x) - 1
    j = 0
    delta = 0

    if x[i][j+1] > x[i][j]:
        delta += 1
    else:
        delta += -1
    if x[i][j] > x[i-1][j]:
        delta += 1
    else:
        delta += -1
        
    delta_tv[i][j] = delta
    
    return ((np.dot(np.transpose(A), f(x)) - np.dot(np.transpose(A), y)) - 0.001 * delta_tv)

hessien = 2 * np.dot(np.transpose(A), A)
step = np.linalg.inv(hessien)

print('matrixes:', hessien, step)
print('matrix Identity:', np.dot(hessien, step))

y_red = f(red_img)
y_green = f(green_img)
y_blue = f(blue_img)

y_list = [y_red, y_green, y_blue]
x_list = []

itr = 10001

# Re use of predent results

# rec_bromo = io.imread('rec_bromo.png')
# rec_bromo = img_as_float(rec_bromo)

#Initialization
rec_bromo = np.ones(img_size, dtype=np.uint8)

x_rgb_prog_list = [[], [], []]

for color in range(3):
    x_color_0 = np.random.rand(245, 326)
    x_color_0 = rec_bromo[:, :, color].copy()
    
    lr = 0.00001
    # lr =step
    save = int(itr/10)
    cur_x = x_color_0
    for i in range(itr+1):
        prev_x = cur_x
        # print(grad_cost_function(prev_x, y))
        cur_x = cur_x - lr * grad_cost_function_tv(prev_x, y_list[color])
        if i%250 == 0:
            print('___________________________')
            print(cost_function_tv(cur_x, y_list[color]))
            print(i / ( itr - 1 ) * 100, '% done')
            # print(cur_x - original_list[color])
        
        if i%save == 0:
            x_rgb_prog_list[color].append((np.rint((255*cur_x)).astype(np.uint8)))
            # print()
        
    x_list.append((np.rint((255*cur_x)).astype(np.uint8)))
    
for i in range(3):
    img = Image.fromarray(x_list[i], 'L')
    # img.show()
print(x_list)
    
reconstructed_img = np.ones(img_size, dtype=np.uint8)
reconstructed_img[:, :, 0] = x_list[0]
reconstructed_img[:, :, 1] = x_list[1]
reconstructed_img[:, :, 2] = x_list[2]

print(reconstructed_img)
print(image_int)

rec_img = Image.fromarray(reconstructed_img, 'RGB')
rec_img.save('rec_bromo.png')
rec_img.show()

image_test = np.ones(img_size, dtype=np.uint8)
image_test[:, :, 0] = np.rint((255*red_img)).astype(np.uint8)
image_test[:, :, 1] = np.rint((255*green_img)).astype(np.uint8)
image_test[:, :, 2] = np.rint((255*blue_img)).astype(np.uint8)

image_test= Image.fromarray(image_test, 'RGB')
image_test.show()

for i in range(10):
    reconstructed_img = np.ones(img_size, dtype=np.uint8)
    reconstructed_img[:, :, 0] = x_rgb_prog_list[0][i]
    reconstructed_img[:, :, 1] = x_rgb_prog_list[1][i]
    reconstructed_img[:, :, 2] = x_rgb_prog_list[2][i]
    rec_img = Image.fromarray(reconstructed_img, 'RGB')
    rec_img.save('rec_bromo_%s.png' % i)
