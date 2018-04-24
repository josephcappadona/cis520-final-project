import pickle
from GUI import Application, View, Document, Window, Cursor, rgb
from GUI.Geometry import pt_in_rect, offset_rect, rects_intersect
from GUI.StdColors import black, red
from GUI.Files import FileType
from GUI import Image
from GUI.FileDialogs import request_old_file
from GUI.Colors import rgb
import imghdr
from math import sqrt
import csv
import numpy as np

TESSERACT_CACHE_DIR = '../tesseract'

WINDOW_SIZE = (800, 1000)
MODELED_DOCUMENT_VIEW_SIZE = (750, 800)
MODELED_DOCUMENT_VIEW_PADDING = 50
CUSTOM_PARAMS = [(7, (True,False,False), ''), # link tokens to form words
                 (40, (True,False,False), ' '), # link tokens in line
                 (25, (False,True,False), None),
                 (1, (True,True,True), None),
                 (1, (True,True,True), None)]

CUSTOM_PARAMS_2 = [(7, (True,False,False), ''), # link tokens to form words
                 (200, (True,False,False), ' '), # link tokens in line
                 (50, (False,True,False), None),
                 (1, (True,True,True), None),
                 (1, (True,True,True), None)]
class AppImportError(ValueError):
    def __init__(self, msg):
        super(ValueError, self).__init__()
        self.args += (msg, )


########################
# GENERIC DATA IMPORTS #
########################
from pandas import read_csv

BLOCKS_COLUMNS = ['x',
                  'y',
                  'w',
                  'h',
                  'text']
def import_blocks(filepath, img_filepath=None):

    if filepath and 'ghega' in filepath and 'blocks' in filepath:
        blocks_df = import_ghega_blocks(filepath)
    elif not filepath or 'tesseract' in filepath:
        blocks_df = import_tesseract_blocks(filepath, img_filepath=img_filepath)
    else:
        blocks_df = read_csv(filepath, sep=',', header=0)

        if any(col not in blocks_df for col in BLOCKS_COLUMNS):
            error = ValueError()
            error.args += ('Blocks CSV {} is missing field(s)'.format(filepath), )
            raise error

    for column_name in list(blocks_df.columns.values):
        if column_name not in BLOCKS_COLUMNS:
            blocks_df = blocks_df.drop(column_name, axis=1)

    # filter nonsense chars
    blocks_df = blocks_df.drop(blocks_df[blocks_df['text'] == '~'].index)

    blocks = blocks_df.T.to_dict().values()
    for block in blocks:
        block['texts'] = [[block['text']]]
    return blocks


GROUNDTRUTH_COLUMNS = ['element_type',
                       'label_exists',
                       'label_x',
                       'label_y',
                       'label_w',
                       'label_h',
                       'label_text',
                       'value_x',
                       'value_y',
                       'value_w',
                       'value_h',
                       'value_text']
def import_groundtruth(filepath):

    if 'ghega' in filepath and 'groundtruth' in filepath:
        gt_df = import_ghega_groundtruth(filepath)
    else:
        gt_df = read_csv(filepath, sep=',', header=0)

        if any(col not in gt_df for col in GROUNDTRUTH_COLUMNS):
            error = ValueError()
            missing = set(GROUNDTRUTH_COLUMNS).difference(set(gt_df.columns))
            error.args += ("Groundtruth CSV {} is missing field(s):  {}".format(filepath, ', '.join(missing)), )
            raise error

    for column_name in list(gt_df.columns.values):
        if column_name not in GROUNDTRUTH_COLUMNS:
            gt_df = gt_df.drop(column_name, axis=1)

    groundtruth = gt_df.T.to_dict().values()
    return groundtruth




##########################
# GHEGA DATA SET IMPORTS #
##########################
GHEGA_DPI = 300 # since ghega location data is in inches, we need to convert units to pixels
GHEGA_BLOCKS_COLUMNS = ['block_type',
                        'page',
                        'x',
                        'y',
                        'w',
                        'h',
                        'text',
                        'very_useless_serialized_data']
def import_ghega_blocks(filepath):
    blocks_df = read_csv(filepath, sep=',', header=None, names=GHEGA_BLOCKS_COLUMNS)

    return process_ghega_blocks(blocks_df)

GHEGA_GROUNDTRUTH_COLUMNS = ['element_type',
                             'label_page',
                             'label_x',
                             'label_y',
                             'label_w',
                             'label_h',
                             'label_text',
                             'value_page',
                             'value_x',
                             'value_y',
                             'value_w',
                             'value_h',
                             'value_text']
def import_ghega_groundtruth(filepath):
    gt_df = read_csv(filepath, sep=',', header=None, names=GHEGA_GROUNDTRUTH_COLUMNS)
    
    gt_df['label_exists'] = np.where(gt_df['label_page'] != -1, True, False)

    return process_ghega_blocks(gt_df, prefixes=['label_', 'value_'])

def process_ghega_blocks(blocks_df, prefixes=['']):
    blocks_df = blocks_df.copy()

    # convert inches to pixels
    resize = lambda x: x * GHEGA_DPI
    fields = ['x', 'y', 'w', 'h']
    for field in fields:
        for prefix in prefixes:
            blocks_df[prefix+field] = blocks_df[prefix+field].apply(resize)
    return blocks_df


import io
TESSERACT_BLOCKS_COLUMNS = ['text',
                            'l',
                            'b',
                            'r',
                            't',
                            'null']
def import_tesseract_blocks(filepath, img_filepath=None):
    if filepath:
        blocks_df = read_csv(filepath, sep=' ', header=None, names=TESSERACT_BLOCKS_COLUMNS, encoding='utf-8', quoting=csv.QUOTE_NONE)
    elif img_filepath:
        blocks_df = infer_tesseract_blocks(img_filepath)

    return process_tesseract_blocks(blocks_df, PIL.Image.open(img_filepath).size)

def infer_tesseract_blocks(img_filepath):
    boxes_ssv = get_pytesseract_boxes(img_filepath)
    blocks_df = read_csv(io.StringIO(unicode(boxes_ssv)), sep=' ', header=None, names=TESSERACT_BLOCKS_COLUMNS, encoding='utf-8', quoting=csv.QUOTE_NONE)

    return blocks_df

def process_tesseract_blocks(blocks_df, img_size):
    blocks_df = blocks_df.copy()

    # invert pytesseract vertical coords
    blocks_df['b'] = img_size[1]-1 - blocks_df['b']
    blocks_df['t'] = img_size[1]-1 - blocks_df['t']

    # rename columns/transpose
    blocks_df['x'] = blocks_df['l']
    blocks_df['y'] = blocks_df['t']
    blocks_df['w'] = blocks_df['r'] - blocks_df['l']
    blocks_df['h'] = blocks_df['b'] - blocks_df['t']

    return blocks_df



class MyApp(Application):

    def __init__(self):
        Application.__init__(self)

        self.file_type = FileType(name="App Document", suffix="app")
        # self.app_cursor = Cursor("app.tiff")

    def open_app(self):
        self.new_cmd()

    def make_document(self, fileref):
        # if fileref
        #     open file and return App(info)
        return AppDoc()

    def make_window(self, document):
        win = Window(size=WINDOW_SIZE, document=document)
        view = AppView(model=document)
                       #extent=(1000,1000)) # cursor = self.app_cursor, scorlling='hv'
        win.place(view, left=0, top=0, right=0, bottom=0, sticky='nsew')
        win.show()


class AppView(View):

    def draw(self, canvas, update_rect):
        self.model.modeled_doc.draw(canvas)

    def mouse_down(self, event):
        x, y = event.position
        if self.model.modeled_doc.contains(x,y):
            boxes = self.model.modeled_doc.find_boxes(x, y)
            i = -1
            for i, box in enumerate(boxes):
                id_, type_, l_or_v = box
                # print data associated w box
                print("The clicked element is the {} for {} ({})".format(l_or_v, type_, id_))
            if i == -1:
                self.model.toggle_clusters()


class AppDoc(Document):

    filepath = None
    modeled_doc = None
    img_filepath = None
    blocks_filepath = None
    gt_filepath = None

    def new_contents(self):
        print("Select an image file:")
        img_fr = request_old_file(prompt="LOAD IMG FILE PLZ", default_dir=None)
        try:
            self.img_filepath = '{}/{}'.format(img_fr.dir.path, img_fr.name)
        except AttributeError:
            raise AppImportError('You must choose an image file to import!')
        if not imghdr.what(self.img_filepath):
            raise AppImportError('File {} does not appear to be an image file'.format(self.img_filepath))
        
        print("Select a blocks file (or cancel to infer blocks from the image):")
        blocks_fr = request_old_file(prompt="LOAD BLOCKS FILE PLZ", default_dir=None)
        try:
            self.blocks_filepath = '{}/{}'.format(blocks_fr.dir.path, blocks_fr.name)
            if 'csv' not in blocks_fr.name.split('.'):
                raise AppImportError('File {} does not appear to be a CSV file'.format(self.blocks_filepath))
            print("{}\n".format(self.blocks_filepath))
        except AttributeError:
            print("No blocks file chosen. Blocks will be inferred using pytesseract OCR.\n")
            self.infer_blocks = True
        
        print("Select a groundtruth file (or cancel to leave it blank):")
        gt_fr = request_old_file(prompt="LOAD GROUNDTRUTH FILE PLZ", default_dir=None)
        try:
            self.gt_filepath = '{}/{}'.format(gt_fr.dir.path, gt_fr.name)
            if 'csv' not in gt_fr.name.split('.'):
                raise AppImportError('File {} does not appear to be a CSV file'.format(self.gt_filepath))
            print("{}\n".format(self.gt_filepath))
        except AttributeError:
            print("No groundtruth file chosen.\n")
        

        self.init_doc(self.img_filepath, blocks_filepath=self.blocks_filepath, gt_filepath=self.gt_filepath)

    def toggle_clusters(self):
        self.modeled_doc.toggle_clusters()
        self.changed()
        self.notify_views()

    def read_contents(self, file_):
        self.filepath, self.modeled_doc = pickle.load(file_)
        self.changed()
        self.notify_views()

    def write_contents(self, file_):
        pickle.dump((self.filepath, self.modeled_doc), file_)

    def init_doc(self, filepath, blocks_filepath=None, gt_filepath=None):
        self.filepath = filepath
        self.modeled_doc = ModeledDocument(filepath, MODELED_DOCUMENT_VIEW_SIZE, blocks_filepath=blocks_filepath, gt_filepath=gt_filepath)
        self.changed()
        self.notify_views()



class ModeledDocument(Image):

    blocks = []
    clustered_blocks = []
    groundtruth = []
    always_show_groundtruth = True
    current_view_type = None
    current_view = None
    params = []

    def __init__(self, img_filepath, size, blocks_filepath=None, gt_filepath=None, params=None):
        self.img_filepath = img_filepath
        Image.__init__(self, file=img_filepath)

        self.gt_filepath = gt_filepath
        self.blocks_filepath = blocks_filepath

        # get image location and size information
        l, t, r, b = self.get_bounds() # get actual image bounds
        w, h = (r - l), (b - t) # compute image width and height

        max_x, max_y = size
        resize_ratio = min(max_x / w, max_y / h)
        resize = lambda x: x * resize_ratio

        # scale down image to fit window
        new_l = l + MODELED_DOCUMENT_VIEW_PADDING
        new_t = t + MODELED_DOCUMENT_VIEW_PADDING
        new_w = resize(w)
        new_h = resize(h)
        self.rect = (new_l, new_t, new_l+new_w, new_t+new_h)

        self.blocks_views = []
        # scale down data values
        if blocks_filepath or img_filepath:
            blocks = import_blocks(blocks_filepath, img_filepath=img_filepath)
            hs = []
            for block in blocks:
                x, y, w, h = block['x'], block['y'], block['w'], block['h']
                hs.append(h)
            print(sorted(hs)[:30])
            resize_fields = set(['x', 'y', 'w', 'h']) # fields to resize
            self.blocks = [{k: (resize(v) if k in resize_fields else v) for k, v in block.iteritems()} for block in blocks]

            clustered_blocks = blocks[:]
            self.params = params if params else CUSTOM_PARAMS
            self.eps, self.hvd, self.seps = zip(*self.params)
            print('Iteratively finding block clusters...')
            clustered_blocks = get_clustered_blocks(clustered_blocks, CUSTOM_PARAMS)

            # resize data to fit image's new size in window
            resized_cluster_blocks = []
            for cluster in clustered_blocks:
                resized_cluster = []
                for block in cluster:
                    resized_block = {k: (resize(v) if k in resize_fields else v) for k, v in block.iteritems()}
                    resized_cluster.append(resized_block)
                resized_cluster_blocks.append(resized_cluster)
            self.clustered_blocks = resized_cluster_blocks
            print('')

            self.blocks_views.append(self.blocks)
            self.blocks_views.extend(self.clustered_blocks)

        if gt_filepath:
            self.gt_filepath = gt_filepath
            groundtruth = import_groundtruth(self.gt_filepath) if self.gt_filepath else None
            # resize data to fit image's new size in window
            fields = set(['label_x', 'label_y', 'label_w', 'label_h', 'value_x', 'value_y', 'value_w', 'value_h']) # fields to resize
            self.groundtruth = [{k: (resize(v) if k in fields else v) for k, v in gt.iteritems()} for gt in groundtruth]
            
            self.blocks_views.append(self.groundtruth)
        
        self.num_views = len(self.blocks_views)
        self.current_view_index = 0
        self.current_view = self.blocks_views[self.current_view_index]
        self.current_view_type = 'blocks' if self.blocks else 'groundtruth' if self.groundtruth else None
        print('\nmodel view is {}'.format(self.get_complex_view_type()))
#

    def toggle_clusters(self):

        self.current_view_index = (self.current_view_index + 1) % self.num_views
        self.current_view = self.blocks_views[self.current_view_index]
        count = 0
        if self.blocks:
            count += 1
            if self.current_view_index == count-1:
                self.current_view_type = 'blocks'
        if self.clustered_blocks and self.current_view_index != 0:
            for cluster in self.clustered_blocks:
                count += 1
                if self.current_view_index == count-1:
                    self.current_view_type = 'clusters'
        if self.groundtruth:
            count += 1
            if self.current_view_index == count-1:
                self.current_view_type = 'groundtruth'

        print('\nmodel view is now {}'.format(self.get_complex_view_type()))

    def get_complex_view_type(self):
        return self.current_view_type + ('(eps={},hvd={})'.format(self.eps[self.current_view_index-1],''.join([str(int(b)) for b in self.hvd[self.current_view_index-1]])) if self.current_view_type=='clusters' else '') + ('+groundtruth' if self.groundtruth and self.always_show_groundtruth and self.current_view_type != 'groundtruth' else '')

    def contains(self, x, y):
        return pt_in_rect((x, y), self.rect)

#

    def find_boxes(self, x, y):
        boxes = []
        if self.current_view_type == 'blocks':
            for id_, block in enumerate(self.current_view):
                if self.block_contains(block, x, y):
                    yield block['texts'], 'Block_'.format(id_), 'TextBlock'

        if self.current_view_type == 'clusters':
            for id_, block in enumerate(self.current_view):
                if self.block_contains(block, x, y):
                    yield block['texts'], 'Cluster_{}'.format(id_), 'Cluster'

        if self.current_view_type == 'groundtruth' or self.always_show_groundtruth:
            for id_, gt in enumerate(self.groundtruth):
                if gt['label_exists'] == 1:
                    if self.block_contains(gt, x, y, prefix='label_'):
                        yield gt['label_text'], gt['element_type'], 'LABEL'

                if self.block_contains(gt, x, y, prefix='value_'):
                    yield gt['value_text'], gt['element_type'], 'VALUE'

    def block_contains(self, block, x, y, prefix=''):
        l, t, r, b = self.rect
        bx, by, bw, bh = block[prefix+'x'], block[prefix+'y'], block[prefix+'w'], block[prefix+'h']
        return l+bx <= x <= l+bx+bw and t+by <= y <= t+by+bh
#

    def draw(self, canvas):
        Image.draw(self, canvas, self.get_bounds(), self.rect)

        if self.current_view_type == 'blocks':
            self.draw_raw_blocks(canvas, self.current_view)
        if self.current_view_type == 'clusters':
            self.draw_clustered_blocks(canvas, self.current_view)
        if self.current_view_type == 'groundtruth' or self.always_show_groundtruth:
            self.draw_groundtruth(canvas, self.groundtruth)


    # DRAW HELPER METHODS
    def draw_raw_blocks(self, canvas, blocks):
        l, t, r, b = self.rect
        for block in blocks:
            x, y, w, h = block['x'], block['y'], block['w'], block['h']
            self.draw_box(canvas, l+x, t+y, w, h, rgb_=(0,0,0))
    
    def draw_clustered_blocks(self, canvas, clustered_blocks):
        l, t, r, b = self.rect
        for block in clustered_blocks:
            x, y, w, h = block['x'], block['y'], block['w'], block['h']
            self.draw_box(canvas, l+x, t+y, w, h, rgb_=(0.45,0,0.45))


    def draw_groundtruth(self, canvas, groundtruth):
        l, t, r, b = self.rect
        for gt in groundtruth:
            labels_coords = None
            if gt['label_exists']:
                x, y, w, h = gt['label_x'], gt['label_y'], gt['label_w'], gt['label_h']
                self.draw_box(canvas, l+x, t+y, w, h, rgb_=(0,0,1))
                labels_coords = (l+x+w/2.0, t+y+h/2.0)

            x, y, w, h = gt['value_x'], gt['value_y'], gt['value_w'], gt['value_h']
            self.draw_box(canvas, l+x, t+y, w, h, rgb_=(1,0,0))
            value_coords = (l+x+w/2.0, t+y+h/2.0)

            if labels_coords:
                self.draw_line(canvas, labels_coords, value_coords, rgb_=(0,0.75,0), alpha=0.5, pensize=2)


    # DRAW STATIC METHODS
    @staticmethod
    def draw_box(canvas, x, y, w, h, rgb_=(1,1,1)):
        canvas.pencolor = rgb(*rgb_, alpha=0.50)
        canvas.fillcolor = rgb(*rgb_, alpha=0.25)
        canvas.pensize = 1
        canvas.newpath()
        canvas.moveto(x, y)
        canvas.lineto(x+w, y)
        canvas.lineto(x+w, y+h)
        canvas.lineto(x, y+h)
        canvas.closepath()
        canvas.fill_stroke()

    @staticmethod
    def draw_line(canvas, pt1, pt2, rgb_=(1,1,1), alpha=1.0, pensize=1):
        canvas.pencolor = rgb(*rgb_, alpha=alpha)
        canvas.pensize = pensize
        canvas.newpath()
        canvas.moveto(*pt1)
        canvas.lineto(*pt2)
        canvas.stroke()



########################
### BLOCK CLUSTERING ###
########################
def get_clustered_blocks(blocks, params):
    
    euclidean_distance = lambda (x1,y1),(x2,y2): sqrt((x2-x1)**2 + (y2-y1)**2)
    def make_DBSCAN_dist(horizontal, vertical, diag):
        def DBSCAN_dist(b1, b2):
            if blocks_do_intersect(b1, b2) and vertical and horizontal:
                return 0
            elif blocks_h_overlap(b1, b2) and vertical:
                _, t_1, _, b_1 = b1
                _, t_2, _, b_2 = b2
                if b_1 < t_2: # if b1 is closer to 0
                    return t_2 - b_1
                else:
                    return t_1 - b_2
            elif blocks_v_overlap(b1, b2) and horizontal:
                l_1, _, r_1, _ = b1
                l_2, _, r_2, _ = b2
                if r_1 < l_2: # if b1 is closer to 0
                    return l_2 - r_1
                else:
                    return l_1 - r_2
            elif diag:
                l_1, t_1, r_1, b_1 = b1
                l_2, t_2, r_2, b_2 = b2
                c_1 = x_1, y_1 = (l_1 + (r_1-l_1)/2, t_1 + (b_1-t_1)/2)
                c_2 = x_2, y_2 = (l_2 + (r_2-l_2)/2, t_2 + (b_2-t_2)/2)
                center_dist = euclidean_distance(c_1, c_2)
                border_dists = []
                for pt_1 in [(l_1, t_1), (l_1, b_1), (r_1, t_1), (r_1, b_1), (x_1, t_1), (x_1, b_1)]:
                    for pt_2 in [(l_2, t_2), (l_2, b_2), (r_2, t_2), (r_2, b_2), (x_2, t_2), (x_2, b_2)]:
                        border_dists.append(euclidean_distance(pt_1, pt_2))
                dist = min(center_dist, min(border_dists))
                return dist
            else:
                return 10**10
        return DBSCAN_dist

    clustered_blocks_list = []
    clustered_blocks = blocks[:]
    for eps, (h,v,d), sep in params:
        DB = DB_from_blocks(clustered_blocks)
        min_pts = 1

        DBSCAN_dist = make_DBSCAN_dist(h, v, d)
        clusters = DBSCAN(DB, DBSCAN_dist, eps, min_pts)
        clustered_blocks = process_DBSCAN_clusters(clusters, clustered_blocks, horizontal=h, vertical=v, diag=d, sep=sep)
        clustered_blocks_list += [clustered_blocks]
    return clustered_blocks_list


def DB_from_blocks(blocks):
    return [(block['x'], block['y'], block['x']+block['w'], block['y']+block['h']) for block in blocks]

def process_DBSCAN_clusters(clusters, blocks, horizontal=False, vertical=False, diag=False, sep=None):
    clustered_blocks = []
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        elif len(cluster) == 1:
            block_id = cluster.pop()
            clustered_blocks.append(blocks[block_id])
        else: # len(cluster) >= 2
            cluster_blocks = [blocks[i] for i in cluster]
            if horizontal and not vertical and not diag:
                # sort left to right
                sorted_cluster_blocks = sorted(cluster_blocks, key=lambda b: b['x'])
            elif vertical and not horizontal and not diag:
                # sort top to bottom
                sorted_cluster_blocks = sorted(cluster_blocks, key=lambda b: b['y'])
            elif (horizontal and vertical) or diag:
                # sort left to right, then top to bottom
                sorted_cluster_blocks = sorted(cluster_blocks, key=lambda b: (b['y'], b['x']))
            cluster_block = sorted_cluster_blocks.pop(0)
            for block in sorted_cluster_blocks:
                # combine blocks first left to right, then top to bottom, in order to minimize complexity of 'texts' field
                cluster_block = combine_blocks(cluster_block, block, horizontal=horizontal, vertical=vertical, diag=diag, sep=sep)
            clustered_blocks.append(cluster_block)
    return clustered_blocks


def combine_blocks(block_1, block_2, horizontal=False, vertical=False, diag=False, sep=None):
    l_1, t_1, w_1, h_1 = block_1['x'], block_1['y'], block_1['w'], block_1['h']
    r_1, b_1 = l_1+w_1, t_1+h_1
    l_2, t_2, w_2, h_2 = block_2['x'], block_2['y'], block_2['w'], block_2['h']
    r_2, b_2 = l_2+w_2, t_2+h_2

    # compute new bounding box
    min_l = min(l_1, l_2)
    min_t = min(t_1, t_2)
    max_r = max(l_1 + w_1, l_2 + w_2)
    max_b = max(t_1 + h_1, t_2 + h_2)

    new_block = {}
    new_block['x'] = min_l
    new_block['y'] = min_t
    new_block['w'] = max_r - min_l
    new_block['h'] = max_b - min_t
    
    new_block['texts'] = []
    h_overlap = blocks_h_overlap((l_1,t_1,r_1,b_1), (l_2,t_2,r_2,b_2))
    v_overlap = blocks_v_overlap((l_1,t_1,r_1,b_1), (l_2,t_2,r_2,b_2))
    if vertical or (h_overlap and not v_overlap):
        top_block = block_1 if t_1 < t_2 else block_2
        bottom_block = block_2 if top_block == block_1 else block_1
        if len(top_block['texts']) == 1 and len(bottom_block['texts']) == 1:
            new_block['texts'] = [top_block['texts'][0] + ['\n'] + bottom_block['texts'][0]]
        else:
            new_block['texts'] = top_block['texts'] + bottom_block['texts']
    elif horizontal or (v_overlap and not h_overlap):
        left_block = block_1 if l_1 < l_2 else block_2
        right_block = block_2 if left_block == block_1 else block_1
        if (len(left_block['texts']) == 1 and '\n' in left_block['texts'][0]) or (len(right_block['texts']) == 1 and '\n' in right_block['texts'][0]):
            new_block['texts'] = [left_block['texts'][0] + right_block['texts'][0]]
        else:
            new_block['texts'] = left_block['texts'] + right_block['texts']
    else:
        first_block = block_1 if t_1 < t_2 else block_2
        second_block = block_2 if first_block == block_1 else block_1
        new_block['texts'] = first_block['texts'] + second_block['texts']
    new_block['text'] = '\n'.join([' '.join(sub_block) for sub_block in new_block['texts']])

    return new_block


def blocks_do_intersect(b1, b2):
    return blocks_h_overlap(b1, b2) and blocks_v_overlap(b1, b2)

def blocks_h_overlap(b1, b2):
    l_1, t_1, r_1, b_1 = b1
    l_2, t_2, r_2, b_2 = b2
    return (l_1 <= r_2) and (r_1 >= l_2)

def blocks_v_overlap(b1, b2):
    l_1, t_1, r_1, b_1 = b1
    l_2, t_2, r_2, b_2 = b2
    return (t_1 <= b_2) and (b_1 >= t_2)


##########
# DBSCAN #
##########
def DBSCAN(DB, dist, eps, min_pts):
    print("Running DBSCAN (eps = {})....".format(eps))
    C = -1
    labels = {P:None for P in DB}
    for P in DB:
        if labels[P] != None: continue
        
        N = RangeQuery(DB, dist, P, eps)
        if len(N) < min_pts:
            labels[P] = -1
            continue
        
        C += 1
        labels[P] = C
        S = N.difference(set([P]))
        S_list = list(S)
        i = 0
        while i < len(S_list):
            Q = S_list[i]
            if labels[Q] == -1: labels[Q] = C
            if labels[Q] != None: i += 1; continue
            labels[Q] = C
            N = RangeQuery(DB, dist, Q, eps)
            if len(N) >= min_pts:
                new_pts = N.difference(S)
                S_list.extend(list(new_pts))
            i += 1
    
    block_labels = {i:labels[P] for i, P in enumerate(DB)}

    # cluster[0] is noise
    clusters = [set() for i in range(C+2)]
    for block_id, cluster_id in block_labels.iteritems():
        clusters[cluster_id+1].add(block_id)
    return clusters

def RangeQuery(DB, dist, Q, eps):
    neighbors = set()
    for P in DB:
        if dist(Q, P) <= eps:
            neighbors.add(P)
    return neighbors



###############
# PYTESSERACT #
###############
import PIL
import pytesseract
import os.path

def get_pytesseract_boxes(img_filepath):
    # cache filepath
    img_filename = img_filepath.split('/')[-1]
    cache_filename = img_filename + '.tesseract'
    cache_filepath = TESSERACT_CACHE_DIR + '/' + cache_filename

    if os.path.exists(cache_filepath):
        print('Found {} in cache!'.format(cache_filepath))
        with open(cache_filepath, 'r') as cache_file:
            res = cache_file.read()
    else:
        print("Running pytesseract...")
        img = PIL.Image.open(img_filepath)
        res = pytesseract.image_to_boxes(img, config='--oem 1', lang='eng')

        with open(cache_filepath, 'w') as cache_file:
            cache_file.write(res)
    return res



#################
# BUILD LEXICON #
#################
import re, Levenshtein

DATE_REGEX = r"^(?:(?:31(\/|-|\.)(?:0?[13578]|1[02]))\1|(?:(?:29|30)(\/|-|\.)(?:0?[1,3-9]|1[0-2])\2))(?:(?:1[6-9]|[2-9]\d)?\d{2})$|^(?:29(\/|-|\.)0?2\3(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00))))$|^(?:0?[1-9]|1\d|2[0-8])(\/|-|\.)(?:(?:0?[1-9])|(?:1[0-2]))\4(?:(?:1[6-9]|[2-9]\d)?\d{2})$" # https://stackoverflow.com/questions/15491894/regex-to-validate-date-format-dd-mm-yyyy#15504877
TELE_REGEX = r"^(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?$" # https://stackoverflow.com/questions/123559/a-comprehensive-regex-for-phone-number-validation#123681
EMAIL_REGEX = r"[^@\s]+@[^@\s]+\.[^@\s.]+$"
ALPHABET = set('abcdefghijklmnopqrstuvwxyz')
NUMERALS = set('1234567890')

def clean_word(word):
    word = word.lower()
    if '$' in word:
        word = '__MONEY__'
    elif re.match(DATE_REGEX, word):
        word = '__DATE__'
    elif re.match(TELE_REGEX, word):
        word = '__TELE__'
    elif re.match(EMAIL_REGEX, word):
        word = '__EMAIL__'
    return word

def clean_sentence(sentence):
    return ' '.join([clean_word(word) for word in sentence.split()])

# run on final set of cluster blocks
# use as input to w2v
def extract_text(blocks):
    sentences = []
    for block in blocks:
        if len(block['texts']) == 1:
            sentences.append(' '.join(block['texts'][0])) # should we join on something other than ' '?
        elif len(block['texts']) > 1:
            for text_block in block['texts']:
                sentences.append(' '.join(text_block))
    return [clean_sentence(sentence) for sentence in sentences]

import sys
reload(sys)
sys.setdefaultencoding('utf8')
from sparray import sparray # http://www.janeriksolem.net/sparray-sparse-n-dimensional-arrays-in.html


#############
# BUILD X,y #
#############
# TODO: build ability to mask around particular block
# run on first set of cluster blocks
# use as input to CNN
def build_ghega_x(blocks, img_shape, w2v, scaling_factor=1.0):
    w, h = img_shape
    w, h = int(round(scaling_factor*w)), int(round(scaling_factor*h))
    model = sparray((h, w, W2V_DIM))

    for block in blocks:
        word = clean_word(block['texts'][0][0])
        if word not in w2v:
            vocab = w2v.vocab.keys()
            word = vocab[np.argmin([Levenshtein.distance(w, word.encode('utf-8', errors='replace')) for w in vocab])]
        for x in range(block['x'], block['x']+block['w']+1):
            x = int(round(scaling_factor * x))
            for y in range(block['y'], block['y']+block['h']+1):
                y = int(round(scaling_factor * y))
                model[h-1-y,x] = w2v[word]
    return model

def get_ghega_labels(groundtruth_blocks):
    y = {}
    for block in groundtruth_blocks:
        element_type = block['element_type'].lower()
        y[element_type + '_label'] = block['label_exists']
        y[element_type + '_value'] = True
    return y

def build_gt_classes(gt_blocks_list):
    ys = [get_ghega_labels(gt_blocks) for gt_blocks in gt_blocks_list]
    classes = set()
    for y in ys:
        for k in y.keys():
            classes.add(k)
    return sorted(classes)

def build_ghega_X(img_filepaths, gt_filepaths, w2v, scaling_factor=1.0):
    gt_blocks_list, blocks_list, clustered_blocks_list = [], [], []
    for i, (img_fp, gt_fp) in enumerate(zip(img_filepaths, gt_filepaths)):
        print('\n\nProcessing {} ({} of {})'.format(img_fp, i+1, len(img_filepaths)))
        
        blocks = import_blocks(None, img_filepath=img_fp)
        clustered_blocks = get_clustered_blocks(blocks[:], CUSTOM_PARAMS)
        blocks_list += [clustered_blocks]

        gt_blocks_list += [import_groundtruth(gt_fp)]
    sorted_classes = build_gt_classes(gt_blocks_list)



    img_shape = w, h = PIL.Image.open(img_filepaths[0]).size 
    w, h = int(round(scaling_factor*w)), int(round(scaling_factor*h))
    yield ((h,w,W2V_DIM+1))

    Xs = []; x_file = open("../data/preprocessed/patents_X.pickle", "wb")
    Ys = []; y_file = open("../data/preprocessed/patents_Y.pickle", "wb")
    # sparray((len(gt_filepaths), len(sorted_classes)+1))
    for i, (blocks, gt_blocks) in enumerate(zip(blocks_list, gt_blocks_list)):
        X = build_ghega_x(blocks, img_shape, w2v, scaling_factor=scaling_factor)
        X_bag = {}
        Y = np.zeros((len(sorted_classes)+1))
        ROI_neg = sparray((h, w, 1))
        for x in range(w):
            for y in range(h):
                ROI_neg[y,x] = 1



        for gt_block in gt_blocks:
            if gt_block['label_exists']:
                ROI_l = sparray((h, w, 1))
                l_x, l_y, l_w, l_h = gt_block['label_x'], gt_block['label_y'], gt_block['label_w'], gt_block['label_h']
                l_x, l_y, l_w, l_h = int(round(scaling_factor*l_x)), int(round(scaling_factor*l_y)), int(round(scaling_factor*l_w)), int(round(scaling_factor*l_h))
                # mark "region-of-interest" as 1, negative ROI as 0
                for x in range(l_x, l_x+l_w+1):
                    for y in range(l_y, l_y+l_h+1):
                        ROI_l[y,x] = 1
                        ROI_neg[y,x] = 0
                #X_l = (X, ROI_l)
                X_l = np.c_[X.dense(), ROI_l.dense()]
                Y_l = Y.copy()
                Y_l[sorted_classes.index(gt_block['element_type'].lower()+'_label')] = 1

                yield X_l
                #pickle.dump(X_l, x_file)# Xs.append(nparray_to_sparray_3d(X_l))
                #pickle.dump(Y_l, y_file)# Ys.append(Y_l)

            if gt_block['value_text']:
                ROI_v = sparray((h, w, 1))
                v_x, v_y, v_w, v_h = gt_block['value_x'], gt_block['value_y'], gt_block['value_w'], gt_block['value_h']
                v_x, v_y, v_w, v_h = int(round(scaling_factor*v_x)), int(round(scaling_factor*v_y)), int(round(scaling_factor*v_w)), int(round(scaling_factor*v_h))
                # mark "region-of-interest" as 1
                for x in range(v_x, v_x+v_w+1):
                    for y in range(v_y, v_y+v_h+1):
                        ROI_v[y,x] = 1
                        ROI_neg[y,x] = 1
                #X_v = (X, ROI_v)
                X_v = np.c_[X.dense(), ROI_v.dense()]
                Y_v = Y.copy()
                Y_v[sorted_classes.index(gt_block['element_type'].lower()+'_value')] = 1

                yield X_v
                #pickle.dump(X_v, x_file) #Xs.append(nparray_to_sparray_3d(X_v))
                #pickle.dump(Y_v, y_file) #Ys.append(Y_v)

        #X_neg = (X, ROI_neg) 
        X_neg = np.c_[X.dense(), ROI_neg.dense()]
        Y_neg = Y.copy()
        Y_neg[-1] = 1

        yield X_neg
        #pickle.dump(X_neg, x_file) #Xs.append(nparray_to_sparray_3d(X_neg))
        #pickle.dump(Y_neg, y_file) #Ys.append(Y_neg)


def build_ghega_Y(img_filepaths, gt_filepaths, w2v, scaling_factor=1.0):
    gt_blocks_list, blocks_list, clustered_blocks_list = [], [], []
    for i, (img_fp, gt_fp) in enumerate(zip(img_filepaths, gt_filepaths)):
        print('\n\nProcessing {} ({} of {})'.format(img_fp, i+1, len(img_filepaths)))
        
        blocks = import_blocks(None, img_filepath=img_fp)
        params = CUSTOM_PARAMS[0]
        eps, (h,v,d), sep = get_clustered_blocks(blocks[:], (eps, (h,v,d), sep))
        blocks_list += [clustered_blocks]

        for eps, (h,v,d), sep in CUSTOM_PARAMS[1:]:
            clustered_blocks = get_clustered_blocks(clustered_blocks, (eps, (h,v,d), sep))
            clustered_blocks_list += [clustered_blocks]

        gt_blocks_list += [import_groundtruth(gt_fp)]
    sorted_classes = build_gt_classes(gt_blocks_list)



    img_shape = w, h = PIL.Image.open(img_filepaths[0]).size 
    w, h = int(round(scaling_factor*w)), int(round(scaling_factor*h))

    Xs = []; x_file = open("../data/preprocessed/patents_X.pickle", "wb")
    Ys = []; y_file = open("../data/preprocessed/patents_Y.pickle", "wb")
    # sparray((len(gt_filepaths), len(sorted_classes)+1))
    for i, (blocks, gt_blocks) in enumerate(zip(blocks_list, gt_blocks_list)):
        X = build_ghega_x(blocks, img_shape, w2v, scaling_factor=scaling_factor)
        X_bag = {}
        Y = np.zeros((len(sorted_classes)+1))
        ROI_neg = sparray((h, w, 1))
        for x in range(w):
            for y in range(h):
                ROI_neg[y,x] = 1



        for gt_block in gt_blocks:
            if gt_block['label_exists']:
                ROI_l = sparray((h, w, 1))
                l_x, l_y, l_w, l_h = gt_block['label_x'], gt_block['label_y'], gt_block['label_w'], gt_block['label_h']
                l_x, l_y, l_w, l_h = int(round(scaling_factor*l_x)), int(round(scaling_factor*l_y)), int(round(scaling_factor*l_w)), int(round(scaling_factor*l_h))
                # mark "region-of-interest" as 1, negative ROI as 0
                for x in range(l_x, l_x+l_w+1):
                    for y in range(l_y, l_y+l_h+1):
                        ROI_l[y,x] = 1
                        ROI_neg[y,x] = 0
                #X_l = (X, ROI_l)
                X_l = np.c_[X.dense(), ROI_l.dense()]
                Y_l = Y.copy()
                Y_l[sorted_classes.index(gt_block['element_type'].lower()+'_label')] = 1

                #yield X_l
                yield Y_l
                #pickle.dump(X_l, x_file)# Xs.append(nparray_to_sparray_3d(X_l))
                #pickle.dump(Y_l, y_file)# Ys.append(Y_l)

            if gt_block['value_text']:
                ROI_v = sparray((h, w, 1))
                v_x, v_y, v_w, v_h = gt_block['value_x'], gt_block['value_y'], gt_block['value_w'], gt_block['value_h']
                v_x, v_y, v_w, v_h = int(round(scaling_factor*v_x)), int(round(scaling_factor*v_y)), int(round(scaling_factor*v_w)), int(round(scaling_factor*v_h))
                # mark "region-of-interest" as 1
                for x in range(v_x, v_x+v_w+1):
                    for y in range(v_y, v_y+v_h+1):
                        ROI_v[y,x] = 1
                        ROI_neg[y,x] = 1
                #X_v = (X, ROI_v)
                X_v = np.c_[X.dense(), ROI_v.dense()]
                Y_v = Y.copy()
                Y_v[sorted_classes.index(gt_block['element_type'].lower()+'_value')] = 1

                #yield X_v
                yield Y_v
                #pickle.dump(X_v, x_file) #Xs.append(nparray_to_sparray_3d(X_v))
                #pickle.dump(Y_v, y_file) #Ys.append(Y_v)

        #X_neg = (X, ROI_neg) 
        X_neg = np.c_[X.dense(), ROI_neg.dense()]
        Y_neg = Y.copy()
        Y_neg[-1] = 1

        #yield X_neg
        yield Y_neg
        #pickle.dump(X_neg, x_file) #Xs.append(nparray_to_sparray_3d(X_neg))
        #pickle.dump(Y_neg, y_file) #Ys.append(Y_neg)





from gensim.models.word2vec import Word2Vec
W2V_DIM = 100
def build_word2vec_model(blocks_filepaths):
    sentences = []
    for fp in blocks_filepaths:
        blocks = import_blocks(fp)
        clustered_blocks = blocks[:]
        for (eps, (h,v,d), sep) in CUSTOM_PARAMS:
            clustered_blocks = get_clustered_blocks(clustered_blocks, eps=eps, horizontal=h, vertical=v, diag=d, sep=sep)
        split_sentences = [sentence.split() for sentence in extract_text(clustered_blocks)]
        sentences.extend(split_sentences)
    model = Word2Vec(sentences, size=W2V_DIM, window=5, min_count=5, compute_loss=True)
    model.save('../models/w2v.model')
    return model

def load_word2vec_model(model_filepath):
    return Word2Vec.load(model_filepath)

import fnmatch
import os
def get_files(dir_path, ext):
    matches = []
    for root, dirnames, filenames in os.walk(dir_path):
        for filename in fnmatch.filter(filenames, ext):
            matches.append(os.path.join(root, filename))
    return matches

def get_ghega_img_files(datasheets_or_patents):
    dir_path = '../ghega-dataset/{}'.format(datasheets_or_patents)
    return get_files(dir_path, '*.out.000.png')

def get_ghega_blocks_files(datasheets_or_patents):
    dir_path = '../ghega-dataset/{}'.format(datasheets_or_patents)
    return get_files(dir_path, '*.blocks.csv')

def get_ghega_gt_files(datasheets_or_patents):
    dir_path = '../ghega-dataset/{}'.format(datasheets_or_patents)
    return get_files(dir_path, '*.groundtruth.csv')
    

iii = -1
def nparray_to_sparray_3d(arr):
    global iii
    iii += 1
    print("Converting array {}".format(iii))
    sparr = sparray(arr.shape)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                if arr[i,j,k] != 0:
                    sparr[i,j,k] = arr[i,j,k]
    return sparr



def process_ghega_data(b, e):
    img_filepaths = sorted(get_ghega_img_files('patents'))
    blocks_filepaths = sorted(get_ghega_blocks_files('patents'))
    gt_filepaths = sorted(get_ghega_gt_files('patents'))

    w2v_model = load_word2vec_model('../models/w2v.model')
    return build_ghega_X(img_filepaths[b:e], gt_filepaths[b:e], w2v_model.wv, scaling_factor=0.1), build_ghega_Y(img_filepaths[b:e], gt_filepaths[b:e], w2v_model.wv, scaling_factor=0.1)



def main():
    MyApp().run()


if __name__ == "__main__":
    main()






def extract_features_(blocks):
    for block in blocks:
        label_words = []
        if block['label_exists']:
            feature_name = block['element_type'].lower() + '_label'
            feature_dict[feature_name] = label_words = block['label_text'].lower().split()
        feature_name = block['element_type'].lower() + '_value'
        feature_dict[feature_name] = value_words = block['value_text'].lower().split()
        feature_dict['words'] = set(value_words).union(label_words)
    return feature_dict

def get_tesseract_analysis(img_filepath):
    # cache filepath
    img_filename = img_filepath.split('/')[-1]
    cache_filename = img_filename + '.tesseract-la'
    cache_filepath = TESSERACT_CACHE_DIR + '/' + cache_filename

    if os.path.exists(cache_filepath):
        print('Found {} in cache!'.format(cache_filepath))
        with open(cache_filepath, 'r') as cache_file:
            res = cache_file.read()
    else:
        print("Running pytesseract layout analysis for {}...".format(img_filepath))
        img = PIL.Image.open(img_filepath)
        res = pytesseract.image_to_data(img, config='--psm 1 tsv', lang='eng')

        with open(cache_filepath, 'w') as cache_file:
            cache_file.write(res)
    res = unicode(res)
    return io.StringIO(res)

def find_matching_gt_block(block, gt_blocks):
    b1 = block['x'], block['y'], block['x']+block['w'], block['y']+block['h']
    print('\n\n>>>>>')
    print(block['texts'])
    for gt_block in gt_blocks:
        if gt_block['label_exists']:
            gt_block_label = {}
            gt_block_label['element_type'] = gt_block['element_type']
            gt_block_label['block_type'] = 'label'
            gt_block_label['x'] = gt_block['label_x']
            gt_block_label['y'] = gt_block['label_y']
            gt_block_label['w'] = gt_block['label_w']
            gt_block_label['h'] = gt_block['label_h']
            gt_block_label['text'] = gt_block['label_text']
            gt_block_label['words'] = set([word.lower() for word in gt_block_label['text'].split()])
            b2 = gt_block_label['x'], gt_block_label['y'], gt_block_label['x']+gt_block_label['w'], gt_block_label['y']+gt_block_label['h']
            #print(gt_block_label['words'])
            if all([any([label_word in text for sub_block in block['texts'] for text in sub_block]) for label_word in gt_block_label['words']]):
                return gt_block_label
        gt_block_value = {}
        gt_block_value['element_type'] = gt_block['element_type']
        gt_block_value['block_type'] = 'value'
        gt_block_value['x'] = gt_block['value_x']
        gt_block_value['y'] = gt_block['value_y']
        gt_block_value['w'] = gt_block['value_w']
        gt_block_value['h'] = gt_block['value_h']
        gt_block_value['text'] = gt_block['value_text']
        gt_block_value['words'] = set([word.lower() for word in gt_block_label['text'].split()])
        b2 = gt_block_value['x'], gt_block_value['y'], gt_block_value['x']+gt_block_value['w'], gt_block_value['y']+gt_block_value['h']
        #print(gt_block_value['words'])
        if all([any([value_word in text for sub_block in block['texts'] for text in sub_block]) for value_word in gt_block_value['words']]):
            return gt_block_value
    return None


# features[word_1]:
#   hAlign_word_2
#   sameBlock_word_2
#   len_line
#   len_block
def process_tesseract_analysis(tsv, gt_blocks, vAlignThresh=15):
    word_dicts = read_csv(tsv, sep='\t', header=0).T.to_dict().values()
    words = set()
    prev_word_num = -1
    prev_line_num = -1
    prev_block_num = -1
    line = []
    block = []
    line_blocks = []
    line_clusters = []
    text_blocks = []
    clustered_blocks = []

    features = {}
    # iterate over word blocks
    for i, word_dict in enumerate(word_dicts):

        # if block is textual
        if word_dict['text']:
            if str(word_dict['text']).lower() == 'nan':
                continue
            word_dict['w'] = int(word_dict['width'])
            word_dict['h'] = int(word_dict['height'])
            word_dict['x'] = int(word_dict['left'])
            word_dict['y'] = int(word_dict['top'])
            word_dict['texts'] = [[word_dict['text'].lower()]]
            text_blocks += [word_dict]

            word = word_dict['text'] = word_dict['text'].lower()
            features[word] = {}
            features[word]['w'] = word_dict['w']
            features[word]['h'] = word_dict['h']
            features[word]['x'] = word_dict['x']
            features[word]['y'] = word_dict['y']
            words.add(word)

            line_num = word_dict['line_num']
            if line_num == prev_line_num or prev_line_num == -1:
                line += [word]
                line_blocks += [word_dict]
            if line_num != prev_line_num or i == len(word_dicts)-1:
                # end of prev line
                line_words = set(line)
                for word_1 in line_words:
                    for word_2 in line_words:
                        if word_1 != word_2:
                            features[word_1]['hAlign_'+word_2] = True
                            features[word_2]['hAlign_'+word_1] = True
                    features[word_1]['line_len'] = len(line)

                if line_blocks:
                    clustered_line = line_blocks[0]
                    for word_block in line_blocks[1:]:
                        clustered_line = combine_blocks(clustered_line, word_block, horizontal=True, sep=' ')
                    line_clusters += [clustered_line]
                    line = []
                    line_blocks = []
            prev_line_num = line_num

            block_num = word_dict['block_num']
            if block_num == prev_block_num or prev_block_num == -1:
                block += [word]
            if block_num != prev_block_num or i == len(word_dicts)-1:
                # end of prev block
                
                if line_clusters:
                    clustered_block = line_clusters[0]
                    for clustered_line in line_clusters[1:]:
                        clustered_block = combine_blocks(clustered_block, clustered_line, vertical=True, sep='\n')

                    clustered_block_words = set(block)
                    for word_1 in clustered_block_words:
                        for word_2 in clustered_block_words:
                            if word_1 != word_2:
                                features[word_1]['sameBlock_'+word_2] = True
                                features[word_2]['sameBlock_'+word_1] = True
                        features[word_1]['block_len'] = len(block)

                clustered_blocks += [clustered_block]
                block = []
                line_clusters = []
            prev_block_num = block_num

    for w_1, word_1 in features.iteritems():
        x_1 = word_1['x']+word_1['w']/2
        for w_2, word_2 in features.iteritems():
            if w_1 != w_2:
                x_2 = word_2['x']+word_2['w']/2
                if abs(x_1 - x_2) < vAlignThresh:
                    features[w_1]['vAlign_'+w_2] = True
                    features[w_2]['vAlign_'+w_1] = True
    return features

from collections import defaultdict
def get_clusters(blocks, clustered_blocks):
    clusters = defaultdict(set)
    cluster_of_block = {}
    for block in blocks:
        b1 = block['x'], block['y'], block['x']+block['w'], block['y']+block['h']
        for c, clustered_block in enumerate(clustered_blocks):
            b2 = clustered_block['x'], clustered_block['y'], clustered_block['x']+clustered_block['w'], clustered_block['y']+clustered_block['h']
            if blocks_do_intersect(b1, b2):
                clusters[c].add(b1)
                cluster_of_block[b1] = c
    return clusters, cluster_of_block

def is_text(text):
    return any(all(c in ALPHABET for c in word) for word in text.split())

def is_numeric(text):
    try:
        _ = float(text)
        return True
    except ValueError:
        return False

def is_alphanumeric(text):
    return all(c in ALPHABET or c in NUMERALS for c in text)

VALIGN_THRESH = 10
def process_blocks(blocks, gt_blocks, clustered_blocks, vAlignThresh=15):
    for block in blocks:
        block['text'] = clean_sentence(block['text'])
        block['texts'] = [[block['text']]]
    clusters, cluster_of_block = get_clusters(blocks, clustered_blocks)
    features = []
    for block in blocks:
        block_features = {'text':None, 'block_type':None, 'element_type':None, 'hAlign':set(), 'vAlign':set(), 'sameCluster':set(), 'num_words':None, 'num_chars':None}
        block_features['text'] = text = block['text']
        block_features['num_words'] = len(text.split())
        block_features['num_chars'] = len(text)
        block_features['is_text'] = is_text(text)
        block_features['is_num'] = is_numeric(text)
        block_features['is_alphanumeric'] = is_alphanumeric(text)
        for gt_block in gt_blocks:
            if gt_block['label_exists']:
                if block['text'] == gt_block['label_text']:
                    block_features['block_type'] = 'label'
                    block_features['element_type'] = gt_block['element_type']
            if block['text'] == gt_block['value_text']:
                block_features['block_type'] = 'value'
                block_features['element_type'] = gt_block['element_type']
        for block_ in blocks:
            # hAlign
            if blocks_v_overlap(b1, b2):
                hAlign_words = set([word for sub_block in block_['texts'] for line in sub_block for word in line.split()])
                block_features['hAlign'].update(hAlign_words)

            # vAlign
            b1 = x_1,_,w_1,_ = block['x'], block['y'], block['x']+block['w'], block['y']+block['h']
            b2 = x_2,_,w_2,_ = block_['x'], block_['y'], block_['x']+block_['w'], block_['y']+block_['h']
            if abs(x_1 - x_2) < VALIGN_THRESH or abs((x_1+w_1) - (x_2+w_2)) < VALIGN_THRESH:
                    vAlign_words = set([word for sub_block in block_['texts'] for line in sub_block for word in line.split()])
                    block_features['vAlign'].update(vAlign_words)

            # sameCluster
            if cluster_of_block[b1] == cluster_of_block[b2]:
                cluster_words = set([word for sub_block in block_['texts'] for line in sub_block for word in line.split()])
                block_features['sameCluster'].update(cluster_words)
        features += [block_features]
    return features



def process_ghega_data_2(l=None):
    img_filepaths = sorted(get_ghega_img_files('patents'))
    blocks_filepaths = sorted(get_ghega_blocks_files('patents'))
    gt_filepaths = sorted(get_ghega_gt_files('patents'))

    feature_list = []
    feature_labels = set()
    gt_labels = set()
    for blocks_fp, gt_fp in zip(blocks_filepaths[:l], gt_filepaths[:l]):
        blocks = import_blocks(blocks_fp)
        gt_blocks = import_groundtruth(gt_fp)

        clustered_blocks = get_clustered_blocks(blocks[:], CUSTOM_PARAMS[1:])

        features = process_blocks(blocks, gt_blocks, clustered_blocks[-1])
        feature_list += [features]
    return feature_list
    



# old stuff
'''
def get_clustered_blocks_scored(blocks):
    blocks_d = {i:block for i, block in enumerate(blocks)}
    scores = {i:get_block_score(block) for i, block in blocks_d.iteritems()}
    block_min_score = {i:(set([i]), score, blocks_d[i]) for i, score in scores.iteritems()}
    combined_blocks = {}

    clustered_blocks = []
    prev_blocks = block_min_score.values()
    print('starting with {} blocks'.format(len(prev_blocks)))

    while True:
        pair_indexes = None
        combined_info = None  
        next_blocks = []      
        for i, (cluster_i, score_i, block_i) in enumerate(prev_blocks):
            for j, (cluster_j, score_j, block_j) in enumerate(prev_blocks):
                if i < 2:
                    combined_block = combine_blocks(block_i, block_j)
                    combined_score = get_block_score(combined_block)
                    if combined_score < score_i + score_j:
                        progress = True
                        combined_cluster = cluster_i.union(cluster_j)
                        pair_indexes = (i, j)
                        combined_info = (combined_cluster, combined_score, combined_block)
                        break
            if pair_indexes:
                break
        if not pair_indexes:
            for _, score, block in prev_blocks:
                block['score'] = score
                clustered_blocks.append(block)
            break
        i, j = pair_indexes
        next_blocks = [combined_info] + [block_info for k, block_info in enumerate(prev_blocks) if k not in pair_indexes]
        prev_blocks = next_blocks
        print('now {} blocks'.format(len(prev_blocks)))

    return clustered_blocks


def get_block_score(block, k=2, q=0.5, p=2, C=2*10**4, cache={}):
    w, h = block['w'], block['h']

    if (w,h) in cache:
        return cache[(w,h)]

    perimeter = 2*w + 2*h
    area = w*h
    score = area**q + perimeter**k + C
    # TODO: try making vertical space more costly than horizontal space

    cache[(w,h)] = score
    return score


def get_clustered_blocks_top_down(blocks):
    # combine blocks into single cluster
    document_cluster_block = blocks[0]
    for block in blocks[1:]:
        document_cluster_block = combine_blocks(document_cluster_block, block)
    document_cluster_block_ids = set(range(len(blocks)))
    clusters = [document_cluster_block_ids]

    document_cluster_density = get_cluster_density(document_cluster_block, document_cluster_block_ids, blocks)

def get_cluster_density(cluster_block, cluster_block_ids, blocks):
    
    cluster_area = cluster_block['w'] * cluster_block['h']
    
    blocks_area_sum = 0
    for block_id, block in enumerate(blocks):
        if block_id in cluster_block_ids:
            blocks_area_sum += blocks[block_id]['w'] * blocks[block_id]['h']

    return blocks_area_sum / cluster_area
    
def get_clustered_blocks_horizontal(blocks):
    pass


def get_clustered_blocks_nearby(blocks):
    euclidean_distance = lambda (x1,y1),(x2,y2): sqrt((x2-x1)**2 + (y2-y1)**2)
    closest_block = {}
    closest_block_dist = {}
    for i, block_i in enumerate(blocks):
        l_i, t_i, w_i, h_i = block_i['x'], block_i['y'], block_i['w'], block_i['h']
        r_i, b_i = l_i+w_i, t_i+h_i
        x_i, y_i = l_i+w_i/2, t_i+h_i/2

        min_block = None
        min_dist = 10**10
        for j, block_j in enumerate(blocks):
            l_j, t_j, w_j, h_j = block_j['x'], block_j['y'], block_j['w'], block_j['h']
            r_j, b_j = l_j+w_j, t_j+h_j
            x_j, y_j = l_j+w_j/2, t_j+h_j/2

            center_dist = euclidean_distance((x_i,y_i), (x_j,y_j))
            corners_dists = []
            for corner_i in [(l_i, t_i), (l_i, b_i), (r_i, t_i), (r_i, b_i)]:
                for corner_j in [(l_j, t_j), (l_j, b_j), (r_j, t_j), (r_j, b_j)]:
                    corner_dists.append(euclidean_distance(corner_i, corner_j))
            dist = min(center_dist, min(corner_dists))

            if dist < min_dist:
                min_dist = dist
                min_block = j

        closest_block[i] = min_block
        closest_block_dist[i] = min_dist
'''

