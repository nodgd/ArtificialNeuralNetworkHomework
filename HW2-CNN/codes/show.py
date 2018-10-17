import numpy as np
from PIL import Image, ImageDraw, ImageFont

def floatToInt(data):
    data_min = np.min(data)
    data_max = np.max(data)
    data = (data - data_min) / (data_max - data_min) * 255.0
    return data.astype('uint8')

def drawOne(i, input, output, file_name):
    input = input.reshape(28, 28, 1)
    output = output.reshape(28, 28, 4)
    tmp = np.concatenate([input, output], axis=2)
    data = np.zeros(tmp.shape, dtype='uint8')
    for j in range(data.shape[2]):
        data[ : , : , j] = floatToInt(tmp[ : , : , j])
    data = data.transpose(0, 2, 1).reshape(28, 140)

if __name__ == '__main__':
    input = np.load('first_layer_input.npy')
    output = np.load('first_layer_output.npy')
    print('input.shape = ', input.shape)
    print('output.shape = ', output.shape)

    st = 20
    en = st + 20
    im = Image.new('RGB', (140, 28 * (en - st)), 'white')
    dw = ImageDraw.Draw(im)
    for i in range(st, en):
        tmp = np.concatenate([input[i], output[i]], axis=2)
        data = np.zeros(tmp.shape, dtype='uint8')
        for j in range(data.shape[2]):
            data[ : , : , j] = floatToInt(tmp[ : , : , j])
        data = data.transpose(2, 1, 0).reshape(140, 28)
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                color_value = data[x, y]
                color_str = '#%02x%02x%02x' % (color_value, color_value, color_value)
                dw.rectangle((x, y + (i - st) * 28, x, y + (i - st) * 28), color_str, color_str)

    file_name = 'image/mix-[%d-%d].png' % (st, en, )
    im.save(file_name)

