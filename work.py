from sklearn import cluster
import torch
import torch.optim as optim
import pickle
import os
import time
import numpy as np
import cv2
import uuid
import json

import utils
from painter import *

def _mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device.type}')

FAST_TESTING = False

args = {
    'img_path' : 'test_images/cat3.jpeg',

    'camera_interaction': False,
    # sub_experiment :: if True: a separate experiment is not created, uuid defined below for camera_interaction
    # experiment is used, init_strokes from camera_interaction experiment are optimized using only virtual canvas
    'sub_experiment': False,


    'grid_div': sorted([5, 6, 7, 8]) if FAST_TESTING else sorted([4, 6, 8]),
    'max_m_strokes': 174*1 if FAST_TESTING else 500,
    'iters_per_stroke': 2 if FAST_TESTING else 1,
    'batch_start_id': 0,
    # KUKA
    'clamp' : True,
    'suppression_freq': 25,  # clamp params every `suppression_freq'th` iteration
    'brush_widths': [10., 21.],  # in mm
    'kuka_width': 300.,  # in mm
    'kuka_height': 400.,  # in mm
    # number of consecutive strokes of same color without dipping into paint cup
    'n_without_dipping': 2,
    'max_h': 40.,  # in mm
    'min_h': 10.,  # in mm

    # COLORS
    'n_colors': 5,  # n of quantized colors
    'use_compressed_ref': True,  # whether to use color-compressed reference(input imp) or not

    'experiments_dir' : 'experiments',
    'renderer_checkpoint_dir' : 'checkpoints_G_fix_w',
    'checkpoint_dir' : 'checkpoints_G_fix_w',

    'canvas_size': 512,
    'canvas_color': 'white',
    'with_ot_loss': True,
    'beta_L1': 1.0,
    'beta_ot': 0.1,
    'lr': 0.005,
    'keep_aspect_ratio': True,
    'save_video': True,
    'video': 'MP4V',

    'train': False,
    'net_G': 'zou-fusion-net',
    'max_num_epochs': 400,
    'vis_dir': 'val_out_G',
    'batch_size': 64,
    'print_models': False,
    'renderer': 'oilpaintbrush',
}
if FAST_TESTING: print('FAST TESTING IS ENABLED')

args['total_blocks'] = int(np.sum(np.array(args['grid_div'])**2))
args['m_strokes_per_block'] = args['max_m_strokes'] // args['total_blocks']
args['m_strokes'] = args['total_blocks']*args['m_strokes_per_block']
print(f'TOTAL STROKES : {args["m_strokes"]}')
print(f'STROKES PER BLOCK : {args["m_strokes_per_block"]}')

# params to shift origin from left upper corner (img origin) to right lower corner (kuka origin)
args['x_shift'] = args['kuka_width']
args['y_shift'] = args['kuka_height']
args['x_dir'] = -1
args['y_dir'] = -1

args['max_w'] = max(args['brush_widths'])
args['min_w'] = min(args['brush_widths'])


assert not (args['camera_interaction'] and args['sub_experiment']), \
    'both cannot be True :: choose between camera_interaction and sub_experiment'

assert (not args['camera_interaction']) or args['clamp'], \
    '''you are logging to KUKA clamped unrealistic drawing 
    :: turn off camera_interaction or turn on clamping'''

assert args['batch_start_id'] == 0 or args['camera_interaction'], \
    '''batch_start_id is implemented only for camera_interaction..
    do not use non-zero batch_start_id if camera_interaction is False'''

# EXPERIMENT DIR
experiment_uuid = 'experiment_3c00c72a-ed59-477c-9ea7-3d36e37d5fca'  # cam cat
# experiment_uuid = 'experiment_56ee63dd-4f75-4e23-ad3a-d1654282adb0'. # cam peach
interaction_dir = f'/media/files/experiments/hse_experiments/{experiment_uuid}'

if not (args['camera_interaction'] or args['sub_experiment']):  # if experiment is local
    experiment_uuid = str(uuid.uuid4())  # generating random uuid

print(f'EXPERIMENT UUID : {experiment_uuid}')
_mkdir(args['experiments_dir'])
img_name, img_ext = args['img_path'].split('.')
img_name = img_name.split('/')[-1]
args['img_name'] = img_name
img_dir = os.path.join(args['experiments_dir'], img_name)
_mkdir(img_dir)
exp_dir = os.path.join(img_dir, experiment_uuid)
_mkdir(exp_dir)

if args['sub_experiment']:
    # loading head_experiment description
    with open(f'{exp_dir}/description.txt', 'r') as f:
        args = json.load(f)
        args['sub_experiment'] = True
        args['camera_interaction'] = False
        args['batch_start_id'] = 0

if args['camera_interaction']:
    args['interaction_dir'] = interaction_dir
    args['batch_dir'] = os.path.join(exp_dir, 'cf_batches')  # cf for camera feedback
    args['camera_canvas_dir'] = os.path.join(args['batch_dir'], 'c_canvas')
    _mkdir(args['batch_dir'])
    _mkdir(args['interaction_dir'])
    _mkdir(args['camera_canvas_dir'])
else:
    args['batch_dir'] = os.path.join(exp_dir, 'vf_batches')  # vf for virtual feedback
    _mkdir(args['batch_dir'])



# SAVE DIRS
args['init_strokes_dir'] = os.path.join(exp_dir, 'init_strokes')
args['colors_dir'] = os.path.join(exp_dir, 'colors')
args['virtual_canvas_dir'] = os.path.join(args['batch_dir'], 'v_canvas')
args['video_dir'] = args['virtual_canvas_dir']
args['logs_dir'] = os.path.join(args['batch_dir'], 'logs')
args['strokes_dir'] = os.path.join(args['batch_dir'], 'strokes')

_mkdir(args['init_strokes_dir'])
_mkdir(args['colors_dir'])
_mkdir(args['virtual_canvas_dir'])
_mkdir(args['logs_dir'])

args['compressor_sample_size'] = 5000 if FAST_TESTING else None
args['cc_white'] = [.63, .64, .62]  # cc_white for camera_canvas white


if not args['sub_experiment']:
    # saving experiment description
    with open(f'{exp_dir}/description.txt', 'w') as f:
        json.dump(args, f, indent='\t')


def optimize_x(pt):

    pt._load_checkpoint()
    pt.net_G.eval()

    PARAMS = np.zeros([1, 0, pt.rderr.d], np.float32)

    if args['canvas_color'] == 'white':
        CANVAS_tmp = torch.ones([min(args['grid_div']) ** 2, 3, pt.net_G.out_size, pt.net_G.out_size],
                                dtype=torch.float32).to(device)
    else:
        CANVAS_tmp = torch.zeros([min(args['grid_div']) ** 2, 3, pt.net_G.out_size, pt.net_G.out_size],
                                 dtype=torch.float32).to(device)

    if args['batch_start_id'] and args['camera_interaction']:
        img_path = os.path.join(interaction_dir, f'batch_{args["batch_start_id"]-1}_out.pkl')
        print(f'READING START batch_{args["batch_start_id"]}_out.pkl')
        with open(img_path, 'rb') as f:
            data = pickle.load(f)
        CANVAS_tmp = pt.rderr.preproc_camera_canvas(data, args["batch_start_id"]-1)
        CANVAS_tmp = utils.img2patches(CANVAS_tmp, args['grid_div'][args['batch_start_id']],
                                       pt.net_G.out_size, adder=0.0).to(device)

    pt.batch_id = -1
    for j, pt.m_grid in enumerate(args['grid_div']):
        pt.batch_id += 1
        if (pt.batch_id < args['batch_start_id']) and args['camera_interaction']:
            continue
        pt.img_batch = utils.img2patches(pt.img_, pt.m_grid, pt.net_G.out_size, adder=0.0).to(device)
        pt.G_final_pred_canvas = CANVAS_tmp

        pt.initialize_params()
        pt.set_params_requires_grad(True)
        utils.set_requires_grad(pt.net_G, False)

        pt.optimizer_x = optim.RMSprop([pt.x_ctt, pt.x_color, pt.x_alpha, pt.x_w, pt.x_h], lr=pt.lr, centered=True)
        pt.step_id = 0

        if args['sub_experiment']:
            init_strokes_path = os.path.join(args['init_strokes_dir'],
                                             f'batch_{pt.batch_id}.npy')
            init_strokes = np.load(init_strokes_path).astype(np.float32)
        else:
            init_strokes = np.zeros((pt.m_grid**2, pt.m_strokes_per_block, pt.rderr.d), dtype=np.float32)

        for pt.anchor_id in range(0, pt.m_strokes_per_block):
            if args['sub_experiment']:
                # LOADING strokes from past head_experiment for current sub_experiment
                for grid_id in range(pt.m_grid * pt.m_grid):
                    stroke_params = init_strokes[grid_id, pt.anchor_id, :]
                    pt._insert_stroke_params(grid_id, pt.anchor_id, stroke_params)
            else:
                pt.stroke_sampler(pt.anchor_id)
                # SAVING strokes from current head_experiment for future sub_experiment
                stroke_params = np.c_[
                    pt.x_ctt[:, pt.anchor_id, :].detach().cpu().numpy(),
                     pt.x_w[:, pt.anchor_id, :].detach().cpu().numpy(),
                     pt.x_h[:, pt.anchor_id, :].detach().cpu().numpy(),
                     pt.x_color[:, pt.anchor_id, :].detach().cpu().numpy(),
                     pt.x_alpha[:, pt.anchor_id, :].detach().cpu().numpy()
                ]
                init_strokes[:, pt.anchor_id, :] = stroke_params

            for i in range(args['iters_per_stroke']):
                pt.G_pred_canvas = CANVAS_tmp

                # update x
                pt.optimizer_x.zero_grad()

                pt._clamp_params()  # all params to valid [0, 1] range

                pt._forward_pass()  # pt.x is created  #pt.G_final_pred_canvas is updated
                pt._backward_x()
                pt._drawing_step_states()

                pt._clamp_params()

                pt.optimizer_x.step()
                pt.step_id += 1
                if args['clamp'] and (i % args['suppression_freq'] == 0 or i == args['iters_per_stroke'] - 1):
                    max_w = min(args['max_w'] * pt.m_grid / args['kuka_width'], 1)
                    min_w = args['min_w'] * pt.m_grid / args['kuka_width']
                    pt.x_w.data = torch.clamp(pt.x_w.data, min_w, max_w)  # w*w_k/m_grid < w_max
                    max_h = min(args['max_h'] * pt.m_grid / args['kuka_height'], 1)
                    min_h = args['min_h'] * pt.m_grid / args['kuka_height']
                    pt.x_h.data = torch.clamp(pt.x_h.data, min_h, max_h)

        # all batch pt.x is already trained!
        v = pt._normalize_strokes(pt.x)  # from patch coords to img coords
        v = pt._shuffle_strokes_and_reshape(v)
        v = pt._sort_strokes(v, by='width_color')
        PARAMS = np.concatenate([PARAMS, v], axis=1)
        CANVAS_tmp = pt._render(PARAMS, PARAMS.shape[1] - v.shape[1])

        if args['camera_interaction']:
            img_path = os.path.join(interaction_dir, f'batch_{pt.batch_id}_out.pkl')
            while not os.path.exists(img_path):
                time.sleep(1)
            print(f'READING batch_{pt.batch_id}_out.pkl')
            with open(img_path, 'rb') as f:
                data = pickle.load(f)
            CANVAS_tmp = pt.rderr.preproc_camera_canvas(data, pt.batch_id)

        if (j + 1) < len(args['grid_div']):
            CANVAS_tmp = utils.img2patches(CANVAS_tmp, args['grid_div'][j + 1], pt.net_G.out_size, adder=0.0).to(device)

        if not args['sub_experiment']:
            init_strokes_path = os.path.join(args['init_strokes_dir'],
                                             f'batch_{pt.batch_id}.npy')
            np.save(init_strokes_path, init_strokes, allow_pickle=False)

        print(CANVAS_tmp.shape)



if __name__ == '__main__':
    pt = ProgressivePainter(args=args)
    optimize_x(pt)




