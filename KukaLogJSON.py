import os
import json


class KukaLog:

    def __init__(self, log_dir1, log_dir2, batch_id):
        # log_dir[1, 2] for local and interaction directories
        self.log_dir1 = log_dir1
        self.log_dir2 = log_dir2
        self.log_path1 = os.path.join(log_dir1, f'batch_{batch_id}.json')
        self.log_path2 = os.path.join(log_dir2, f'batch_{batch_id}.json')
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

        with open(self.log_path1, 'w') as f:
            json.dump(self.json, f, indent=' ')

        with open(self.log_path2, 'w') as f:
            json.dump(self.json, f, indent=' ')
