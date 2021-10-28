import os
import cv2
import random

import matplotlib.pyplot as plt
import numpy as np

import loss
from networks import *
import morphology
import renderer
import torch
from image_compressor import ImgCompressor

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PainterBase():

    def __init__(self, args):
        self.checkpoint_dir = args['renderer_checkpoint_dir']

        self.args = args

        self.rderr = renderer.Renderer(args=args)

        # define G
        self.net_G = define_G(rdrr=self.rderr, netG=args['net_G']).to(device)

        self.lr = args['lr']

        self._pxl_loss = loss.PixelLoss(p=1)
        self._sinkhorn_loss = loss.SinkhornLoss(epsilon=0.01, niter=5, normalize=False)
    def _load_checkpoint(self):
        ckpt_path = os.path.join(self.checkpoint_dir, 'last_ckpt.pt')
        # load renderer G
        if os.path.exists(ckpt_path):
            print('loading renderer from pre-trained checkpoint...')
            # load the entire checkpoint
            checkpoint = torch.load(ckpt_path,
                                    map_location=None if torch.cuda.is_available() else device)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.net_G.to(device)
            self.net_G.eval()
        else:
            print('pre-trained renderer does not exist...')
            exit()

    def _compute_acc(self):

        target = self.img_batch.detach()
        canvas = self.G_pred_canvas.detach()
        psnr = utils.cpt_batch_psnr(canvas, target, PIXEL_MAX=1.0)

        return psnr

    def _sort_strokes(self, v, by):
        v = v[0, :, :]
        width_idx = 3
        if by=='width':
            sort_idxs = np.argsort(-v[:, width_idx])
        elif by=='width_color':
            c1 = v[:, [width_idx]]
            brush_idx = (np.argmin(np.abs(self.rderr.brush_widths - c1 * self.rderr.kuka_width), axis=1))
            c1 = - self.rderr.brush_widths[brush_idx] / self.rderr.kuka_width

            c2 = np.zeros_like(c1)
            for i, stroke in enumerate(v):
                R0, G0, B0, _ = stroke[5:]
                color_index, _ = self.rderr.choose_color([R0, G0, B0])
                c2[i] = color_index
            sort_idxs = np.lexsort((c2, c1))

        v = v[sort_idxs]
        v = v[None, ...]
        return v

    def _shuffle_strokes_and_reshape(self, v):
        # v.shape = [-1, d]

        # grid_idx = list(range(self.m_grid ** 2))
        grid_idx = list(range(len(v)))
        random.shuffle(grid_idx)
        v = v[grid_idx, :]
        # v = np.reshape(np.transpose(v, [1, 0, 2]), [-1, self.rderr.d])
        v = np.expand_dims(v, axis=0)

        return v

    def _render(self, v, index):

        v = v[0, :, :]
        out_h, out_w = self.out_h, self.out_w

        print('rendering canvas...')
        self.rderr.create_log(self.batch_id)

        for i in range(index, v.shape[0]):  # for each stroke
            self.rderr.stroke_params = v[i, :]
            if self.rderr.check_stroke():
                self.rderr.draw_stroke()  # updates rderr.canvas

            this_frame = self.rderr.canvas
            this_frame = cv2.resize(this_frame, (out_w, out_h), cv2.INTER_AREA)

            self.video_writer.write((this_frame[:, :, ::-1] * 255.).astype(np.uint8))

        final_rendered_image = np.copy(this_frame)
        print('saving final rendered result...')
        file_path = os.path.join(self.v_canvas_dir, f'batch_{self.batch_id}.jpg')
        cv2.imwrite(file_path, (final_rendered_image[:,:,::-1]*255.).astype(np.uint8))

        self.rderr.end_log()

        return final_rendered_image

    def _normalize_strokes(self, v):

        # from patch [0, 1] to image [0, 1]
        if self.rderr.renderer in ['watercolor', 'markerpen']:
            # x0, y0, x1, y1, x2, y2, radius0, radius2, ...
            xs = np.array([0, 4])
            ys = np.array([1, 5])
            rs = np.array([6, 7])
        elif self.rderr.renderer in ['oilpaintbrush', 'rectangle']:
            # xc, yc, w, h, theta ...
            xs = np.array([0])
            ys = np.array([1])
            rs = np.array([3, 4])
        else:
            raise NotImplementedError('renderer [%s] is not implemented' % self.rderr.renderer)

        # v.shape = [m_grid**2, m_strokes_per_block, d]
        v = v.detach().cpu().numpy()
        _, _, d = v.shape
        v[:, :, rs] /= self.m_grid
        v[np.isnan(v)] = 0.


        x_id = np.arange(self.m_grid)
        y_id = np.arange(self.m_grid)
        x_id, y_id = np.meshgrid(x_id, y_id)
        grid_id = np.c_[x_id.ravel(), y_id.ravel()]
        grid_id = grid_id.reshape(self.m_grid, self.m_grid, 1, 2)

        v = v.reshape(self.m_grid, self.m_grid, -1, d)
        v[:, :, :, np.hstack([xs, ys])] += grid_id
        v[:, :, :, np.hstack([xs, ys])] /= self.m_grid

        v = v.reshape(-1, d)
        v = v[self.keep_mask.detach().cpu().numpy()]
        return v



    def initialize_params(self):
        # uniform init
        self.x_ctt = np.random.rand(
            self.m_grid * self.m_grid, self.m_strokes_per_block,
            self.rderr.d_shape - 2).astype(np.float32)
        self.x_ctt = torch.tensor(self.x_ctt).to(device)

        self.x_w = np.random.rand(
            self.m_grid * self.m_grid, self.m_strokes_per_block,
            1).astype(np.float32)
        self.x_w = torch.tensor(self.x_w).to(device)

        self.x_h = np.random.rand(
            self.m_grid * self.m_grid, self.m_strokes_per_block,
            1).astype(np.float32)
        self.x_h = torch.tensor(self.x_h).to(device)

        self.x_color = np.random.rand(
            self.m_grid * self.m_grid, self.m_strokes_per_block,
            self.rderr.d_color).astype(np.float32)
        self.x_color = torch.tensor(self.x_color).to(device)

        self.x_alpha = np.random.rand(
            self.m_grid * self.m_grid, self.m_strokes_per_block,
            self.rderr.d_alpha).astype(np.float32)
        self.x_alpha = torch.tensor(self.x_alpha).to(device)

    def stroke_sampler(self, anchor_id):

        if anchor_id == self.m_strokes_per_block:
            return
        print(self.img_batch.size())
        print(self.G_final_pred_canvas.size())
        err_maps = torch.sum(
            torch.abs(self.img_batch - self.G_final_pred_canvas),
            dim=1, keepdim=True).detach()  # summation along color dim

        for i in range(self.m_grid * self.m_grid):
            this_err_map = err_maps[i, 0, :, :].cpu().numpy()
            ks = int(this_err_map.shape[0] / 8)
            this_err_map = cv2.blur(this_err_map, (ks, ks))
            this_err_map = this_err_map ** 4
            this_img = self.img_batch[i, :, :, :].detach().permute([1, 2, 0]).cpu().numpy()

            self.rderr.random_stroke_params_sampler(
                err_map=this_err_map, img=this_img)

            self._insert_stroke_params(i, anchor_id, self.rderr.stroke_params)

    def _insert_stroke_params(self, grid_id, stroke_id, stroke_params):

        self.x_ctt.data[grid_id, stroke_id, :] = torch.tensor(
            stroke_params[0:self.rderr.d_shape - 2], dtype=torch.float32)

        self.x_w.data[grid_id, stroke_id, :] = torch.tensor(
            stroke_params[self.rderr.d_shape - 2:self.rderr.d_shape - 1], dtype=torch.float32)

        self.x_h.data[grid_id, stroke_id, :] = torch.tensor(
            stroke_params[self.rderr.d_shape - 1:self.rderr.d_shape], dtype=torch.float32)

        self.x_color.data[grid_id, stroke_id, :] = torch.tensor(
            stroke_params[self.rderr.d_shape:self.rderr.d_shape + self.rderr.d_color], dtype=torch.float32)
        self.x_alpha.data[grid_id, stroke_id, :] = torch.tensor(stroke_params[-1], dtype=torch.float32)

    def _backward_x(self):

        self.G_loss = 0
        self._G_loss = 0
        self.G_loss += self.args['beta_L1'] * self._pxl_loss(
            canvas=self.G_final_pred_canvas, gt=self.img_batch)
        if self.args['with_ot_loss']:
            self.G_loss += self.args['beta_ot'] * self._sinkhorn_loss(
                self.G_final_pred_canvas, self.img_batch)
        self.G_loss.backward()


    def _forward_pass(self):

        self.x = torch.cat([self.x_ctt, self.x_w, self.x_h, self.x_color, self.x_alpha], dim=-1)

        # v - all already sampled in current batch strokes
        v = torch.reshape(self.x[:, 0:self.anchor_id + 1, :],
                          [self.m_grid * self.m_grid * (self.anchor_id + 1), -1])

        keep_mask = ~torch.all(torch.isnan(v), dim=-1)
        self.keep_mask = keep_mask
        keep_idx = torch.where(keep_mask)[0]
        self.keep_idx = keep_idx

        v = v[keep_mask][..., None, None]
        # print(v.shape)
        # v[0,:,0,0] = torch.tensor([0.5, 0.5, 0/10**10, 1., 1., 1., 1., 1., 1.], dtype=torch.double)
        # v[1,:,0,0] = torch.tensor([0.5, 0.5, 1/8., 1., 1., 1., 1., 1., 1.], dtype=torch.double)

        # v = torch.from_numpy(np.array([0.5, 0.5, 0., 1., 1., 1., 1., 1., 1.], dtype=np.float64))[None, ..., None, None]
        self.G_pred_foregrounds, self.G_pred_alphas = self.net_G(v)
        # plt.imshow(self.G_pred_foregrounds[0].detach().numpy().transpose([1, 2, 0]))
        # plt.show()
        # plt.imshow(self.G_pred_foregrounds[1].detach().numpy().transpose([1, 2, 0]))
        # plt.show()
        # plt.imshow(self.G_pred_foregrounds[0].detach().numpy().transpose([1, 2, 0]))
        # plt.show()

        self.G_pred_foregrounds = morphology.Dilation2d(m=1)(self.G_pred_foregrounds)
        self.G_pred_alphas = morphology.Erosion2d(m=1)(self.G_pred_alphas)

        G_pred_foregrounds = torch.zeros(self.m_grid**2*(self.anchor_id+1), 3, self.net_G.out_size, self.net_G.out_size).to(device)
        G_pred_alphas = torch.zeros_like(G_pred_foregrounds).to(device)
        #
        for i in range(len(self.G_pred_foregrounds)):
            G_pred_foregrounds[keep_idx[i]] = self.G_pred_foregrounds[i]
            G_pred_alphas[keep_idx[i]] = self.G_pred_alphas[i]


        G_pred_foregrounds = torch.reshape(
            G_pred_foregrounds, [self.m_grid * self.m_grid, self.anchor_id + 1, 3,
                                      self.net_G.out_size, self.net_G.out_size])

        G_pred_alphas = torch.reshape(
            G_pred_alphas, [self.m_grid * self.m_grid, self.anchor_id + 1, 3,
                                 self.net_G.out_size, self.net_G.out_size])

        for i in range(self.anchor_id + 1):
            # G_pred_foreground = self.G_pred_foregrounds[:, i]
            # G_pred_alpha = self.G_pred_alphas[:, i]
            G_pred_foreground = G_pred_foregrounds[:, i]
            G_pred_alpha = G_pred_alphas[:, i]
            self.G_pred_canvas = G_pred_foreground * G_pred_alpha \
                                 + self.G_pred_canvas * (1 - G_pred_alpha)


        self.G_final_pred_canvas = self.G_pred_canvas


class ProgressivePainter(PainterBase):

    def __init__(self, args):
        super(ProgressivePainter, self).__init__(args=args)

        self.canvas_size = args['canvas_size']
        self.clamped = args['clamp']
        self.grid_div = args['grid_div']
        self.max_divide = max(self.grid_div)
        self.max_m_strokes = args['max_m_strokes']

        self.use_compressed_ref = args['use_compressed_ref']
        self.keep_aspect_ratio = args['keep_aspect_ratio']
        self.m_strokes_per_block = args['m_strokes_per_block']
        self.m_strokes = args['m_strokes']

        self.img_path = args['img_path']
        self.img_name = args['img_name']
        self.img_ = cv2.imread(args['img_path'], cv2.IMREAD_COLOR)
        self.img_ = cv2.cvtColor(self.img_, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.

        self.colors_dir = args['colors_dir']
        self.n_colors = args['n_colors']

        self.kuka_interaction = args['camera_interaction']
        self.v_canvas_dir = args['virtual_canvas_dir']

        compressor = ImgCompressor(self.img_, self.n_colors, sample_size=args['compressor_sample_size'])
        compressor.save_colors(self.colors_dir)
        compressor.save_palette(self.colors_dir)

        if self.kuka_interaction:
            self.c_canvas_dir = args['camera_canvas_dir']

        cv2.imwrite(os.path.join(self.v_canvas_dir, 'reference.jpg'),
                    (self.img_[:, :, [2,1,0]]*255.).astype(np.uint8))
        cv2.imwrite(os.path.join(self.v_canvas_dir, 'reference_compressed.jpg'),
                    (compressor.img_compressed[:, :, [2,1,0]]*255.).astype(np.uint8))


        if self.use_compressed_ref:
            self.img_ = compressor.img_compressed
        self.rderr.compressor = compressor

        self.input_aspect_ratio = self.img_.shape[0] / self.img_.shape[1]
        self.define_out_size()  # self.out_h, self.out_w


        self.img_ = cv2.resize(self.img_, (self.net_G.out_size * self.max_divide,
                                           self.net_G.out_size * self.max_divide), cv2.INTER_AREA)
        self.video_dir = args['video_dir']
        video_path = os.path.join(self.video_dir, 'animation' + '.mp4')
        if args['batch_start_id']:
            try:
                frames = []
                cap = cv2.VideoCapture(video_path)
                while cap.isOpened():
                    r, frame = cap.read()
                    if not r:
                        break
                    frames.append(frame)
                self.video_writer = cv2.VideoWriter(
                    video_path, cv2.VideoWriter_fourcc(*self.args['video']), 40,
                    (self.out_w, self.out_h))
                for frame in frames:
                    self.video_writer.write(frame)
            except SystemError:
                self.video_writer = cv2.VideoWriter(
                    video_path, cv2.VideoWriter_fourcc(*self.args['video']), 40,
                    (self.out_w, self.out_h))
        else:
            self.video_writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*self.args['video']), 40,
            (self.out_w, self.out_h))

    def define_out_size(self):
        if self.keep_aspect_ratio:
            if self.input_aspect_ratio < 1:
                self.out_h = int(self.canvas_size * self.input_aspect_ratio)
                self.out_w = self.canvas_size
            else:
                self.out_h = self.canvas_size
                self.out_w = int(self.canvas_size / self.input_aspect_ratio)
        else:
            self.out_h = self.canvas_size
            self.out_w = self.canvas_size

    def set_params_requires_grad(self, requires_grad=True):
        self.x_ctt.requires_grad = requires_grad
        self.x_color.requires_grad = requires_grad
        self.x_alpha.requires_grad = requires_grad
        self.x_w.requires_grad = requires_grad
        self.x_h.requires_grad = requires_grad

    def _clamp_params(self):
        # clamping params to valid range
        self.x_ctt.data = torch.clamp(self.x_ctt.data, 0.1, 1 - 0.1)
        self.x_color.data = torch.clamp(self.x_color.data, 0, 1.)
        self.x_alpha.data = torch.clamp(self.x_alpha.data, 0, 1.)
        self.x_w.data = torch.clamp(self.x_w.data, 0, .5)
        self.x_h.data = torch.clamp(self.x_h.data, 0, .5)

    def _drawing_step_states(self):
        acc = self._compute_acc().item()
        print('iteration step %d, G_loss: %.5f, step_acc: %.5f, grid_scale: %d / %d, strokes: %d / %d'
              % (self.step_id, self.G_loss.item(), acc,
                 self.m_grid, self.max_divide,
                 self.anchor_id + 1, self.m_strokes_per_block))

    # def _forward_pass(self):
    #     self.x = torch.cat([self.x_ctt, self.x_w, self.x_h, self.x_color, self.x_alpha], dim=-1)
    #     v = torch.reshape(self.x[:, 0:self.anchor_id + 1, :],
    #                       [self.m_grid * self.m_grid * (self.anchor_id + 1), -1, 1, 1])
    #
    #     self.G_pred_foregrounds, self.G_pred_alphas = self.net_G(v)
    #     # G_pred_foregrounds.shape = G_pred_alphas.shape = [m_grid**2*(anchor_id+1), 3, out, out]
    #
    #
    #     self.G_pred_foregrounds = torch.reshape(
    #         self.G_pred_foregrounds, [self.m_grid * self.m_grid, self.anchor_id + 1, 3,
    #                                   self.net_G.out_size, self.net_G.out_size])
    #
    #     self.G_pred_alphas = torch.reshape(
    #         self.G_pred_alphas, [self.m_grid * self.m_grid, self.anchor_id + 1, 3,
    #                              self.net_G.out_size, self.net_G.out_size])
    #
    #     for i in range(self.anchor_id + 1):
    #         G_pred_foreground = self.G_pred_foregrounds[:, i]
    #         G_pred_alpha = self.G_pred_alphas[:, i]
    #         self.G_pred_canvas = G_pred_foreground * G_pred_alpha \
    #                              + self.G_pred_canvas * (1 - G_pred_alpha)
    #
    #     self.G_final_pred_canvas = self.G_pred_canvas





