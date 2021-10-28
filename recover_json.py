import json
import os
import numpy as np
import utils
import cv2
from renderer import Renderer, _normalize
import matplotlib.pyplot as plt

experiment_path = '/Users/jiji/Desktop/OldPainter/experiments/cat3/0a6e2df2-9e7f-48cd-8dd7-955565a2fa38'
# loading args from analyzed experiment
with open(os.path.join(experiment_path, 'description.txt'), 'r') as f:
    args = json.load(f)

n_batches = len(args['grid_div'])
n_colors = args['n_colors']
colors = np.load(os.path.join(args['colors_dir'], f'{n_colors}colorsRGB.npy'))
brush = cv2.imread('brushes/brush_fromweb2_large_horizontal.png', cv2.IMREAD_GRAYSCALE)
brush_widths = args['brush_widths']

# parsing kuka logs
rderr = Renderer(args)
brush_idx = None
color_idx = None
strokes = []
for batch_id in range(n_batches):
    batch_path = os.path.join(os.path.join(rderr.logs_dir, f'batch_{batch_id}.json'))
    with open(batch_path, 'r') as f:
        curr_batch = json.load(f)
    for command in curr_batch['commands']:
        if command['action'] == 'change_brush':
            brush_idx = command['data']['brush_num']
        elif command['action'] == 'color_brush':
            color_idx = command['data']['color_num']
        elif command['action'] == 'spline_stroke':
            w = rderr.args['brush_widths'][brush_idx]  # in mm
            r, g, b = colors[color_idx]

            data = command['data']
            left_point = [data['p0']['x'], data['p0']['y']]
            mid_point = [data['p1']['x'], data['p1']['y']]
            right_point = [data['p2']['x'], data['p2']['y']]

            x, y, h, w, theta = rderr.from_spline_to_params(
                left_point, mid_point, right_point, w)

            x = _normalize(x, rderr.canvas_size)  # [0, 1] to {0, 1, .. canvas_size-1}
            y = _normalize(y, rderr.canvas_size)
            w = max(int(0.5 + w * rderr.canvas_size), 1)  # [0, 1 ] to {1, 2, .. canvas_size}
            h = max(int(0.5 + h * rderr.canvas_size), 1)
            theta = np.pi*theta

            rderr.foreground, rderr.stroke_alpha_map = utils.create_transformed_brush(
                brush, rderr.canvas_size, rderr.canvas_size,
                x, y, w, h, theta,
                r, g, b
            )
            rderr.foreground = np.array(rderr.foreground, dtype=np.float32) / 255.
            rderr.stroke_alpha_map = np.array(rderr.stroke_alpha_map, dtype=np.float32) / 255.
            rderr.canvas = rderr._update_canvas()



init_canvas = cv2.imread(os.path.join(args['virtual_canvas_dir'], f'batch_{n_batches-1}.jpg'), cv2.IMREAD_COLOR)
init_canvas = cv2.cvtColor(init_canvas, cv2.COLOR_BGR2RGB)

cv2.imshow('init', init_canvas)
print(init_canvas.shape)
h, w  = init_canvas.shape[:2]
cv2.imshow('recovered', cv2.resize(rderr.canvas, (w, h)))
cv2.waitKey()





