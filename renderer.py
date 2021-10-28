import numpy as np
import cv2
import random
import utils
import os
from KukaLogJSON import KukaLog
import base64
import pickle
from loss import PixelLoss

import matplotlib.pyplot as plt

ERROR_THRESHOLD = 2000
def _random_floats(low, high, size):
    return [random.uniform(low, high) for _ in range(size)]


def _normalize(x, width):
    return int(x * (width - 1) + 0.5)

# def cast_vector(row):
#     return np.array(list(map(lambda x: x.astype('float32'), row)))


class Renderer:

    def __init__(self, args):

        self.compressor = None
        self.args = args

        self.kuka_interaction = args['camera_interaction']
        if self.kuka_interaction:
            self.interaction_dir = args['interaction_dir']
            self.c_canvas_dir = args['camera_canvas_dir']

        self.logs_dir = args['logs_dir']
        self.img_path = args['img_path']
        self.img_name = args['img_name']

        self.train = args['train']
        self.canvas_size = args['canvas_size']
        self.renderer = args['renderer']
        self.canvas_color = args['canvas_color']
        self.create_empty_canvas()

        self.n_colors = args['n_colors']
        self.cc_white = np.array(args['cc_white'])
        self.color_indices = np.arange(self.n_colors)
        self.brush_idx = -1
        self.color_idx = -1
        self.brush_widths = np.array(args['brush_widths'])
        self.n_without_dipping = args['n_without_dipping']
        self.kuka_width = args['kuka_width']
        self.kuka_height = args['kuka_height']
        self.x_shift = args['x_shift']
        self.y_shift = args['y_shift']
        self.x_dir = args['x_dir']
        self.y_dir = args['y_dir']

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
            self.d = 9  # x, y, w, h, theta, R, G, B, alpha
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


    def create_log(self, batch_id):
        if self.kuka_interaction:
            self.log = KukaLog(self.interaction_dir, self.logs_dir, batch_id)
        else:
            self.log = KukaLog(self.logs_dir, self.logs_dir, batch_id)


    def end_log(self):
        self.log.EndWrite()

    def create_empty_canvas(self):
        if self.canvas_color == 'white':
            self.canvas = np.ones(
                [self.canvas_size, self.canvas_size, 3]).astype('float32')
        else:
            self.canvas = np.zeros(
                [self.canvas_size, self.canvas_size, 3]).astype('float32')

    def random_stroke_params(self):
        self.stroke_params = np.array(_random_floats(0, 1, self.d), dtype=np.float32)

    def random_stroke_params_sampler(self, err_map, img):

        map_h, map_w, c = img.shape

        err_map = cv2.resize(err_map, (self.canvas_size, self.canvas_size))
        err_map[err_map < 0] = 0
        overall_err = np.sum(err_map)
        if np.all((err_map == 0)):
            err_map = np.ones_like(err_map)
        err_map = err_map / (np.sum(err_map) + 1e-99)

        index = np.random.choice(range(err_map.size), size=1, p=err_map.ravel())[0]

        cy = (index // self.canvas_size) / self.canvas_size
        cx = (index % self.canvas_size) / self.canvas_size

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
            alpha = _random_floats(0.98, 1.0, 1)
            if overall_err < ERROR_THRESHOLD:
                x = [float('nan'), float('nan')]
                wh = [float('nan'), float('nan')]
                theta = [float('nan')]
                color = [float('nan'), float('nan'), float('nan')]
                alpha = [float('nan')]
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
            print('Stroke skipped')
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

    def _draw_oilpaintbrush(self, make_log=True):

        x0, y0, theta, w, h, = self.stroke_params[0:5]
        brush = self.brush_large_horizontal
        R0, G0, B0, ALPHA = self.stroke_params[5:]
        color_index, [_R0, _G0, _B0] = self.choose_color([R0, G0, B0])
        brush_idx = int(np.argmin(np.abs(self.brush_widths - w*self.kuka_width)))
        w = self.brush_widths[brush_idx] / self.kuka_width  # [0, 1]
        theta = np.pi * theta

        if make_log:
            if self.color_idx == -1 and self.brush_idx == -1:
                # first command
                self.log.addChangeBrush(brush_idx)
                self.brush_idx = brush_idx
                self.log.addColorBrush(color_index)
                self.log.addTestStroke()
                self.color_idx = brush_idx
                self.same_color_counts = 1
            else:
                color_changed = (self.color_idx != color_index)
                brush_changed = (self.brush_idx != brush_idx)
                if color_changed and brush_changed:
                    self.log.addClearBrush()
                    self.log.addChangeBrush(brush_idx)
                    self.brush_idx = brush_idx
                    self.log.addColorBrush(color_index)
                    self.log.addTestStroke()
                    self.color_idx = color_index
                    self.same_color_counts = 1
                elif color_changed and not brush_changed:
                    self.log.addClearBrush()
                    self.log.addColorBrush(color_index)
                    self.log.addTestStroke()
                    self.color_idx = color_index
                    self.same_color_counts = 1
                elif not color_changed and brush_changed:
                    self.log.addClearBrush()
                    self.log.addChangeBrush(brush_idx)
                    self.brush_idx = brush_idx
                    self.log.addColorBrush(color_index)
                    self.log.addTestStroke()
                    self.same_color_counts = 1
                elif not color_changed and not brush_changed:
                    if self.same_color_counts < self.n_without_dipping:
                        self.same_color_counts += 1
                    else:
                        self.log.addColorBrush(color_index)
                        self.log.addTestStroke()
                        self.same_color_counts = 1
            self.send_kuka_coords([x0, y0], h, w, theta)

        x0 = _normalize(x0, self.canvas_size)  # [0, 1] to {0, 1, .. canvas_size-1}
        y0 = _normalize(y0, self.canvas_size)
        # x0, y0, color_index
        # w = int(1 + w * self.canvas_size)
        # h = int(1 + h * self.canvas_size)
        w = max(int(0.5 + w * self.canvas_size), 1)  # [0, 1 ] to {1, 2, .. canvas_size}
        h = max(int(0.5 + h * self.canvas_size), 1)


        self.foreground, self.stroke_alpha_map = utils.create_transformed_brush(
            brush, self.canvas_size, self.canvas_size,
            x0, y0, w, h, theta,
            _R0, _G0, _B0
        )

        if not self.train:
            self.foreground = cv2.dilate(self.foreground, np.ones([2, 2]))
            self.stroke_alpha_map = cv2.erode(self.stroke_alpha_map, np.ones([2, 2]))

        self.foreground = np.array(self.foreground, dtype=np.float32) / 255.
        self.stroke_alpha_map = np.array(self.stroke_alpha_map, dtype=np.float32) / 255.
        self.canvas = self._update_canvas()

    def _update_canvas(self):
        return self.foreground * self.stroke_alpha_map + \
               self.canvas * (1 - self.stroke_alpha_map)


    def choose_brush(self, h, w):

        if w * h / (self.canvas_size**2) > 0.1:
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
        # [0,1] to kuka_coords

        mid_point = np.array(mid_point)
        _h = np.array([0, h/2])
        # _w = np.array([w/2, 0])

        rotation_M = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        height_step = rotation_M @ _h
        # width_step = rotation_M @ _w

        right_point = mid_point + height_step
        left_point = mid_point - height_step

        # normalization : from [0,1] to mm
        normalization = np.array([self.kuka_width, self.kuka_height])
        to_float = lambda point: [float(el) for el in point]
        shift_coords = np.array([self.x_shift, self.y_shift])  # in mm
        axis_dir = np.array([self.x_dir, self.y_dir])

        left_point = to_float(axis_dir*left_point*normalization + shift_coords)
        mid_point = to_float(axis_dir*mid_point*normalization + shift_coords)
        right_point = to_float(axis_dir*right_point*normalization + shift_coords)

        _round = lambda point: [round(el)/1. for el in point]
        left_point = _round(left_point)
        mid_point = _round(mid_point)
        right_point = _round(right_point)
        self.log.addSplineStroke(*left_point, *mid_point, *right_point)

    def choose_color(self, color):
        # color = cast_vector(color)[None, :]
        color = np.array(color, dtype=np.float32)[None, :]
        color_index, color = self.compressor.get_closest_color(color)
        color_index = int(color_index[0])
        color = color[0]
        return color_index, color

    def from_spline_to_params(self, left_point, mid_point, right_point, w):
        left_point, mid_point, right_point = map(np.array, [left_point, mid_point, right_point])
        normalization = np.array([self.kuka_width, self.kuka_height])
        shift_coords = np.array([self.x_shift, self.y_shift])  # in mm
        axis_dir = np.array([self.x_dir, self.y_dir])

        # kuka to [0, 1]
        left_point = (axis_dir * (left_point - shift_coords) / normalization)
        mid_point = (axis_dir * (mid_point - shift_coords)/normalization)
        right_point = (axis_dir * (right_point - shift_coords)/normalization)

        dx, dy = left_point - mid_point
        theta = np.arctan2(dx, -dy)
        h = 2*np.sqrt(dx**2 + dy**2)

        x, y = mid_point
        return x, y, h, w/self.kuka_width, theta/np.pi

    def preproc_camera_canvas(self, data, batch_id):

        jpg_as_text = base64.b64decode(data['canvas_image'])
        buffer = np.frombuffer(jpg_as_text, dtype=np.uint8)
        img = cv2.imdecode(buffer, flags=1)  # BGR
        img_path = os.path.join(self.c_canvas_dir, f'batch_{batch_id}_out.jpg')
        cv2.imwrite(img_path, img)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.
        s = 15
        corners_c = np.vstack((img[:s, :s], img[-s:, :s], img[:s, -s:], img[-s:, -s:])).reshape(-1, 3).mean(0)
        a = 1
        cc_white = (a*corners_c + (1-a)*self.cc_white)
        img_compressed = self.compressor.compress_camera_canvas(img, cc_white, self.color_indices)
        img_c_path = os.path.join(self.c_canvas_dir, f'batch_{batch_id}_in.jpg')
        cv2.imwrite(img_c_path, (img_compressed[:,:,[2,1,0]]*255.).astype(np.uint8))
        return img_compressed






