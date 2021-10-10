import os
import json


class KukaLog():

    # def __init__(self, batch_id):
    #     result_filepath = os.path.join('./', 'batch_{}.txt'.format(batch_id))
    #     self.f = open(result_filepath, 'w')
    #     self.commands = []

    def __init__(self, batch_dir, img_name, batch_id):
        if os.path.exists(batch_dir) is False:
            os.mkdir(batch_dir)
        result_filepath = os.path.join(batch_dir, img_name)
        if os.path.exists(result_filepath) is False:
            os.mkdir(result_filepath)
        result_filepath = os.path.join(result_filepath, f'batch_{batch_id}.txt')
        self.f = open(result_filepath, 'w')
        self.commands = []

    def addTestStroke(self):
        com = {'action': 'test_stroke'}
        self.commands.append(com)

    def addSplineStroke(self, x0, y0, x1, y1, x2, y2):
        p0 = {'x': x0, 'y': y0}
        p1 = {'x': x1, 'y': y1}
        p2 = {'x': x2, 'y': y2}
        data = {'p0': p0, 'p1': p1, 'p2': p2}
        com = {'action': 'spline_stroke', 'data': data}
        self.commands.append(com)

    def addChangeBrush(self, num):
        com = {'action': 'change_brush', 'data': {'brush_num': num}}
        self.commands.append(com)

    def addClearBrush(self):
        com = {"action": "clear_brush"}
        self.commands.append(com)

    def addColorBrush(self, num):
        com = {'action': 'color_brush', 'data': {'color_num': num}}
        self.commands.append(com)

    def EndWrite(self):
        self.json = {'common_data': {"brush_tilt_threshold": 30}, 'commands': self.commands}
        json.dump(self.json, self.f)
        self.f.close()


