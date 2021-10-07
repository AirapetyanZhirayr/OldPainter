import numpy as np
import cv2
import random
import utils
import os
from KukaLogJSON import KukaLog

from loss import PixelLoss

import matplotlib.pyplot as plt


def _random_floats(low, high, size):
    return [random.uniform(low, high) for _ in range(size)]


def _normalize(x, width):
    return int(x * (width - 1) + 0.5)


class Renderer:

    def __init__(self, args):

        self.CANVAS_WIDTH = args['canvas_size']
        self.renderer = args['renderer']
        self.stroke_params = None
        self.canvas_color = args['canvas_color']
        self.canvas = None
        self.create_empty_canvas()
        self.make_log = args['KukaLog']
        self.batch_dir = args['batch_dir']
        self.img_path = args['img_path']
        self.img_name = os.path.split(self.img_path)[-1]
        self.img_name, self.img_extension = self.img_name.split('.')

        self.train = args['train']

        self.kuka_width = args['kuka_width']
        self.kuka_height = args['kuka_height']
        self.x_shift = args['x_shift']
        self.y_shift = args['y_shift']
        self.compressor = None

        if self.renderer in ['markerpen']:
            self.d = 12  # x0, y0, x1, y1, x2, y2, radius0, radius2, R, G, B, A
            self.d_shape = 8
            self.d_color = 3
            self.d_alpha = 1
        elif self.renderer in ['watercolor']:
            self.d = 15  # x0, y0, x1, y1, x2, y2, radius0, radius2, R0, G0, B0, R2, G2, B2, A
            self.d_shape = 8
            self.d_color = 6
            self.d_alpha = 1
        elif self.renderer in ['oilpaintbrush']:
            self.d = 9  # xc, yc, w, h, theta, R0, G0, B0, R2, G2, B2, A
            self.d_shape = 5
            self.d_color = 3
            self.d_alpha = 1
            self.brush_small_vertical = cv2.imread(
                'brushes/brush_fromweb2_small_vertical-Copy1.png', cv2.IMREAD_GRAYSCALE)
            self.brush_small_horizontal = cv2.imread(
                'brushes/brush_fromweb2_small_horizontal.png', cv2.IMREAD_GRAYSCALE)
            self.brush_large_vertical = cv2.imread(
                'brushes/brush_fromweb2_large_vertical.png', cv2.IMREAD_GRAYSCALE)
            self.brush_large_horizontal = cv2.imread(
                'brushes/brush_fromweb2_large_horizontal.png', cv2.IMREAD_GRAYSCALE)
        elif self.renderer in ['rectangle']:
            self.d = 9  # xc, yc, w, h, theta, R, G, B, A
            self.d_shape = 5
            self.d_color = 3
            self.d_alpha = 1
        else:
            raise NotImplementedError(
                'Wrong renderer name %s (choose one from [watercolor, markerpen, oilpaintbrush, rectangle] ...)'
                % self.renderer)

    # def create_log(self, batch_id):
    #     self.log = KukaLog(batch_id)
    #     self.log.addChangeBrush(0)

    def create_log(self, batch_id):
        self.log = KukaLog(self.batch_dir, self.img_name, batch_id)
        self.log.addChangeBrush(0)

    def end_log(self):
        self.log.EndWrite()

    def create_empty_canvas(self):
        if self.canvas_color == 'white':
            self.canvas = np.ones(
                [self.CANVAS_WIDTH, self.CANVAS_WIDTH, 3]).astype('float32')
        else:
            self.canvas = np.zeros(
                [self.CANVAS_WIDTH, self.CANVAS_WIDTH, 3]).astype('float32')

    def random_stroke_params(self):
        self.stroke_params = np.array(_random_floats(0, 1, self.d), dtype=np.float32)

    def random_stroke_params_sampler(self, err_map, img):

        map_h, map_w, c = img.shape

        err_map = cv2.resize(err_map, (self.CANVAS_WIDTH, self.CANVAS_WIDTH))
        err_map[err_map < 0] = 0
        if np.all((err_map == 0)):
            err_map = np.ones_like(err_map)
        err_map = err_map / (np.sum(err_map) + 1e-99)

        index = np.random.choice(range(err_map.size), size=1, p=err_map.ravel())[0]

        cy = (index // self.CANVAS_WIDTH) / self.CANVAS_WIDTH
        cx = (index % self.CANVAS_WIDTH) / self.CANVAS_WIDTH

        if self.renderer in ['markerpen']:
            # x0, y0, x1, y1, x2, y2, radius0, radius2, R, G, B, A
            x0, y0, x1, y1, x2, y2 = cx, cy, cx, cy, cx, cy
            x = [x0, y0, x1, y1, x2, y2]
            r = _random_floats(0.1, 0.5, 2)
            color = img[int(cy * map_h), int(cx * map_w), :].tolist()
            alpha = _random_floats(0.8, 0.98, 1)
            self.stroke_params = np.array(x + r + color + alpha, dtype=np.float32)
        elif self.renderer in ['watercolor']:
            # x0, y0, x1, y1, x2, y2, radius0, radius2, R0, G0, B0, R2, G2, B2, A
            x0, y0, x1, y1, x2, y2 = cx, cy, cx, cy, cx, cy
            x = [x0, y0, x1, y1, x2, y2]
            r = _random_floats(0.1, 0.5, 2)
            color = img[int(cy * map_h), int(cx * map_w), :].tolist()
            color = color + color
            alpha = _random_floats(0.98, 1.0, 1)
            self.stroke_params = np.array(x + r + color + alpha, dtype=np.float32)
        elif self.renderer in ['oilpaintbrush']:
            # xc, yc, w, h, theta, R0, G0, B0, R2, G2, B2, A
            x = [cx, cy]
            wh = _random_floats(0.1, 0.5, 2)
            theta = _random_floats(0, 1, 1)
            color = img[int(cy * map_h), int(cx * map_w), :].tolist()
            color = color
            alpha = _random_floats(0.98, 1.0, 1)
            self.stroke_params = np.array(x + theta + wh + color + alpha, dtype=np.float32)
        elif self.renderer in ['rectangle']:
            # xc, yc, w, h, theta, R, G, B, A
            x = [cx, cy]
            wh = _random_floats(0.1, 0.5, 2)
            theta = [0]
            color = img[int(cy * map_h), int(cx * map_w), :].tolist()
            alpha = _random_floats(0.8, 0.98, 1)
            self.stroke_params = np.array(x + wh + theta + color + alpha, dtype=np.float32)

    def check_stroke(self):
        r_ = 1.0
        if self.renderer in ['markerpen', 'watercolor']:
            r_ = max(self.stroke_params[6], self.stroke_params[7])
        elif self.renderer in ['oilpaintbrush']:
            r_ = max(self.stroke_params[3], self.stroke_params[4])
        elif self.renderer in ['rectangle']:
            r_ = max(self.stroke_params[2], self.stroke_params[3])
        if r_ > 0.025:
            return True
        else:
            return False

    def draw_stroke(self):

        if self.renderer == 'watercolor':
            return self._draw_watercolor()
        elif self.renderer == 'markerpen':
            return self._draw_markerpen()
        elif self.renderer == 'oilpaintbrush':
            return self._draw_oilpaintbrush()
        elif self.renderer == 'rectangle':
            return self._draw_rectangle()

    def _draw_oilpaintbrush(self):

        # https://hysy.org/color/00beff
        # colors = [[0.0, 0.0, 0.0], [0.39, 0.26, 0.07], [0.0, 0.74, 1.0]]

        x0, y0, theta, w, h, = self.stroke_params[0:5]
        R0, G0, B0, ALPHA = self.stroke_params[5:]
        x0 = _normalize(x0, self.CANVAS_WIDTH)
        y0 = _normalize(y0, self.CANVAS_WIDTH)
        w = int(1 + w * self.CANVAS_WIDTH)
        h = int(1 + h * self.CANVAS_WIDTH)
        theta = np.pi * theta

        # brush = self.choose_brush(h, w)
        brush = self.brush_large_horizontal
        color_index, [_R0, _G0, _B0] = self.choose_color([R0, G0, B0])
        if self.make_log:
            self.log.addColorBrush(color_index)
            # self.log.addTestStroke()
            self.send_kuka_coords([x0, y0], h, w, theta)

        brush_widths = np.array([2., 4., 5., 6., 8., 10.])/self.kuka_width  # in mm
        w_idx = np.argmin((brush_widths -  w)**2)
        w = brush_widths[w_idx]


        self.foreground, self.stroke_alpha_map = utils.create_transformed_brush(
            brush, self.CANVAS_WIDTH, self.CANVAS_WIDTH,
            x0, y0, w, h, theta,
            # R0, G0, B0,
            _R0, _G0, _B0
        )

        if not self.train:
            self.foreground = cv2.dilate(self.foreground, np.ones([2, 2]))
            self.stroke_alpha_map = cv2.erode(self.stroke_alpha_map, np.ones([2, 2]))

        self.foreground = np.array(self.foreground, dtype=np.float32) / 255.
        self.stroke_alpha_map = np.array(self.stroke_alpha_map, dtype=np.float32) / 255.
        self.canvas = self._update_canvas()
        # plt.imshow(self.canvas)
        # plt.show()

    def _update_canvas(self):
        return self.foreground * self.stroke_alpha_map + \
               self.canvas * (1 - self.stroke_alpha_map)

    def choose_brush(self, h, w):

        if w * h / (self.CANVAS_WIDTH**2) > 0.1:
            if h > w:
                brush = self.brush_large_vertical
            else:
                brush = self.brush_large_horizontal
        else:
            if h > w:
                brush = self.brush_small_vertical
            else:
                brush = self.brush_small_horizontal

    def send_kuka_coords(self, mid_point, h, w, theta):
        brush_widths = np.array([2., 4., 5., 6., 8., 10.])
        mid_point = np.array(mid_point)
        _h = np.array([0, h/2])
        _w = np.array([w/2, 0])

        rotation_M = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        height_step = rotation_M @ _h
        width_step = rotation_M @ _w

        if h > w:
            right_point = mid_point + height_step
            left_point = mid_point - height_step
            brush_idx = int(np.argmin((brush_widths - self.kuka_width*w)**2))

        else:
            right_point = mid_point + width_step
            left_point = mid_point - width_step
            brush_idx = int(np.argmin((brush_widths - self.kuka_width*h)**2))


        # normalization : from pixels to mm
        normalization = np.array([self.kuka_width, self.kuka_height])/self.CANVAS_WIDTH
        to_float = lambda point: [float(el) for el in point]
        shift_coords = np.array([self.x_shift, self.y_shift])  # in mm

        left_point = to_float(left_point*normalization + shift_coords)
        mid_point = to_float(mid_point*normalization + shift_coords)
        right_point = to_float(left_point*normalization + shift_coords)
        if self.make_log:
            self.log.addChangeBrush(brush_idx)
            self.log.addSplineStroke(*left_point, *mid_point, *right_point)

    def choose_color(self, color):
        color = np.array(color)[None,:]
        color_index, color = self.compressor.get_closest_color(color)
        color_index = int(color_index[0])
        color = color[0]
        return color_index, color

