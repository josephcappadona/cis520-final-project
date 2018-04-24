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


CUSTOM_PARAMS_PATENTS = [(40, (True,False,False), ' '), # form lines
                         (25, (False,True,False), None), # form blocks
                         (1, (True,True,True), None), # link overlapping boxes
                         (1, (True,True,True), None)] # link overlapping boxes
CUSTOM_PARAMS_DATASHEETS = [(200, (True,False,False), ' '),
                            (50, (False,True,False), None),
                            (1, (True,True,True), None),
                            (1, (True,True,True), None)]
DEFAULT_PARAMS = [(10, (True,False,False), ''), # form words
                  (75, (True,False,False), ' '), # form lines
                  (25, (False,True,False), None),
                  (1, (True,True,True), None),
                  (1, (True,True,True), None)]
def get_DBSCAN_params(patents_or_datasheets):
    if patents_or_datasheets == 'patents':
        return CUSTOM_PARAMS_PATENTS
    elif patents_or_datasheets == 'datasheets':
        return CUSTOM_PARAMS_DATASHEETS
    else:
        return DEFAULT_PARAMS


########################
# GENERIC DATA IMPORTS #
########################
from pandas import read_csv

class AppImportError(ValueError):
    def __init__(self, msg):
        super(ValueError, self).__init__()
        self.args += (msg, )


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
        if type(block['text']) != str:
            block['text'] = ''
        else:
            block['text'] = clean_sentence(block['text'])
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
    for gt_block in groundtruth:
        if gt_block['label_exists']:
            gt_block['label_text'] = clean_sentence(gt_block['label_text'])
        gt_block['value_text'] = clean_sentence(gt_block['value_text'])
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
        doc_type = 'patents' if 'patents' in filepath else 'datasheets' if 'datasheets' in filepath else None
        params = get_DBSCAN_params(doc_type)
        self.modeled_doc = ModeledDocument(filepath, MODELED_DOCUMENT_VIEW_SIZE, blocks_filepath=blocks_filepath, gt_filepath=gt_filepath, params=params)
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
            self.params = get_DBSCAN_params(params)
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
    #print("Running DBSCAN (eps = {})....".format(eps))
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


def get_tesseract_layout_analysis(img_filepath):
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


# features[word_1]:
#   hAlign_word_2
#   sameBlock_word_2
#   len_line
def get_tesseract_layout_analysis_blocks(tsv, gt_blocks):
    word_dicts = read_csv(tsv, sep='\t', header=0).T.to_dict().values()

    # iterate over word blocks
    for i, word_dict in enumerate(word_dicts):
        block_dict = {}
        # if block is textual
        if word_dict['text']:
            if str(word_dict['text']).lower() == 'nan':
                continue
            block_dict['w'] = int(word_dict['width'])
            block_dict['h'] = int(word_dict['height'])
            block_dict['x'] = int(word_dict['left'])
            block_dict['y'] = int(word_dict['top'])
            word = block_dict['text'] = clean_sentence(word_dict['text'].lower())
            block_dict['texts'] = [[word_dict['text']]]

            yield block_dict


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
    word = word.lower().strip('()[]"\':;.!?,@`').replace(',', '')
    if '$' in word:
        word = '__MONEY__'
    elif re.match(DATE_REGEX, word):
        word = '__DATE__'
    elif re.match(TELE_REGEX, word):
        word = '__TELE__'
    elif re.match(EMAIL_REGEX, word):
        word = '__EMAIL__'
    elif is_numeric(word):
        word = '__NUM__'
    elif is_alphanumeric(word):
        word = '__ALPHANUM__'
    elif any(c in word for c in '*_|\\!@#~[]{}()`?\"=+'):
        word = '__GARB__'
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


##################
# WORD2VEC MODEL #
##################
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



######################
# FEATURE GENERATION #
######################
from collections import defaultdict
def get_spatial_clusters(blocks, clustered_blocks):
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
    return any(c in ALPHABET for c in text) and any(c in NUMERALS for c in text) and all(c in ALPHABET or c in NUMERALS for c in text)

def get_all_nltk_stopwords():
    from nltk.corpus import stopwords
    all_stopwords = set()
    for language in stopwords.fileids():
        all_stopwords.update(stopwords.words(language))
    return all_stopwords
STOPWORDS = get_all_nltk_stopwords()

    
# taken from https://stackoverflow.com/questions/27889873/clustering-text-documents-using-scikit-learn-kmeans-in-python#27890107
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
def get_document_clusters(documents, n_clusters=50):
    vectorizer = TfidfVectorizer(stop_words=STOPWORDS)
    X = vectorizer.fit_transform(documents)
    model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=10)
    return model.fit(X), vectorizer

def predict_document_cluster(model, vectorizer, document):
    return model.predict(vectorizer.transform(document))

VALIGN_THRESH = 10
HALIGN_THRESH = 10
def process_blocks(blocks, gt_blocks, clustered_blocks, kmeans=(None, None), vocab=set()):

    # cluster blocks spatially (DBSCAN)
    clusters, cluster_of_block = get_spatial_clusters(blocks, clustered_blocks)
    
    # get document clusters
    kmeans_model, kmeans_vectorizer = kmeans

    features = []
    labels = []
    for block in blocks:
        block_features = {}

        # textual features
        text = block['text']
        block_features['num_words'] = len(text.split())
        block_features['num_chars'] = len(text)
        block_features['is_text'] = int(is_text(text))
        block_features['is_num'] = int(is_numeric(text))
        block_features['is_alphanumeric'] = int(is_alphanumeric(text))
        block_features['cluster'] = predict_document_cluster(kmeans_model, kmeans_vectorizer, [text])[0]
        
        # shape+location features
        x_1 = block['x']
        y_1 = block['y']
        block_features['w'] = w_1 = block['w']
        block_features['h'] = h_1 = block['h']
        block_features['x'] = x_1 + w_1/2
        block_features['y'] = y_1 + h_1/2

        # inter-block features
        for block_ in blocks:
            r_1, b_1 = x_1+w_1, y_1+h_1
            b1 = x_1, y_1, r_1, b_1

            x_2, y_2, w_2, h_2 = block_['x'], block_['y'], block_['w'], block_['h']
            r_2, b_2 = x_2+w_2, y_2+h_2
            b2 = x_2, y_2, r_2, b_2

            # hAlign
            if abs((y_1+h_1/2) - (y_2+h_2/2)) < HALIGN_THRESH or abs(y_1 - y_2) < HALIGN_THRESH or abs((y_1+h_1) - (y_2+h_2)) < HALIGN_THRESH: # if center aligned, or top aligned, or bottom aligned
                hAlign_words = set([word for sub_block in block_['texts'] for line in sub_block for word in line.split() if word not in STOPWORDS and (not vocab or (vocab and word in vocab))])
                for word in hAlign_words:
                    block_features['hAlign_'+word] = 1

            # vAlign
            if abs((y_1+h_1/2) - (y_2+h_2/2)) < VALIGN_THRESH or abs(x_1 - x_2) < VALIGN_THRESH or abs((x_1+w_1) - (x_2+w_2)) < VALIGN_THRESH: # if center aligned, or left aligned, or right aligned
                vAlign_words = set([word for sub_block in block_['texts'] for line in sub_block for word in line.split() if word not in STOPWORDS and (not vocab or (vocab and word in vocab))])
                for word in vAlign_words:
                    block_features['vAlign_'+word] = 1

            # cluster features
            if cluster_of_block[b1] == cluster_of_block[b2]:
                cluster_words = set([word for sub_block in block_['texts'] for line in sub_block for word in line.split() if word not in STOPWORDS and (not vocab or (vocab and word in vocab))])
                for word in cluster_words:
                    # sameCluster
                    block_features['sameCluster_'+word] = 1

                    # vecTo
                    vec_x, vec_y = x_2-x_1, y_2-y_1
                    normalizing_factor = sqrt(vec_x**2 + vec_y**2)
                    vec_x_normalized, vec_y_normalized = vec_x*normalizing_factor, vec_y*normalizing_factor
                    block_features['vecTo_'+word+'_x'] = vec_x_normalized
                    block_features['vecTo_'+word+'_y'] = vec_y_normalized
        features += [block_features]


        # groundtruth labels
        block_type = None
        element_type = None
        if gt_blocks:
            groundtruth = {}
            for gt_block in gt_blocks:
                if gt_block['label_exists']:
                    if block['text'] == gt_block['label_text']:
                        block_type = 'label'
                        element_type = gt_block['element_type']
                if block['text'] == gt_block['value_text']:
                    block_type = 'value'
                    element_type = gt_block['element_type']
        label = element_type+'_'+block_type if block_type and element_type else ''
        labels += [ { label : 1 } ]

    return features, labels


def get_top_words(blocks, k=1000):
    count_dict = defaultdict(int)
    for block in blocks:
        for word in block['text'].split():
            count_dict[word] += 1
    sorted_words = sorted(count_dict.keys(), key=lambda k: count_dict[k], reverse=True)
    return sorted_words[:k]


def process_ghega_data(patents_or_datasheets, l=None):
    img_filepaths = sorted(get_ghega_img_files(patents_or_datasheets))
    blocks_filepaths = sorted(get_ghega_blocks_files(patents_or_datasheets))
    gt_filepaths = sorted(get_ghega_gt_files(patents_or_datasheets))
    params = get_DBSCAN_params(patents_or_datasheets)

    blocks_list = [import_blocks(blocks_fp) for blocks_fp in blocks_filepaths[:l]]
    gt_blocks_list = [import_groundtruth(gt_fp) for gt_fp in gt_filepaths[:l]]

    # cluster docs by content
    documents = [block['text'] for blocks in blocks_list for block in blocks]
    kmeans_docs = get_document_clusters(documents)

    # get top words
    top_words = set(get_top_words([block for blocks in blocks_list for block in blocks]))

    feature_list = []
    label_list = []
    gt_labels = set()
    for i, (blocks, gt_blocks) in enumerate(zip(blocks_list, gt_blocks_list)):
        print('Processing {}... ({} of {})'.format(''.join(blocks_filepaths[i].split('.')[:-2]), i, len(blocks_filepaths)))
        clustered_blocks = get_clustered_blocks(blocks[:], params)

        features, labels = process_blocks(blocks, gt_blocks, clustered_blocks[-1], kmeans=kmeans_docs, vocab=top_words)
        feature_list += features
        label_list += labels
    return feature_list, label_list
    

from sklearn.feature_extraction import DictVectorizer
def vectorize(feature_or_label_list):
    vectorizer = DictVectorizer(sparse=True)
    return vectorizer.fit_transform(feature_or_label_list), vectorizer.get_feature_names()

from scipy.stats import entropy
def select_features(X, Y):
    count_f = np.zeros((X.shape[1], 2), dtype=float)
    count_l = np.zeros((Y.shape[1]), dtype=float)
    count_joint = np.zeros((X.shape[1], 2, Y.shape[1]), dtype=float)
    for i, (x, y) in enumerate(zip(X,Y)):
        if i % 100 == 0:
            print('{} of {}'.format(i, X.shape[0]))
        count_f[:,0] += 1 # assume f_j is 0, correct later (need to do this because of sparse representation of X)
        for (_, i), f in x.todok().iteritems():
            if f:
                count_f[i,0] -= 1
                count_f[i,1] += 1

        for (_, i), l in y.todok().iteritems():
            if l:
                count_l[i] += 1

        count_joint[:][0] += 1 # assume f_j is 0, correct later (need to do this because of sparse representation of X)
        for (_, i), f in x.todok().iteritems():
            for (_, j), l in y.todok().iteritems():
                if f and l:
                    count_joint[i,0,j] -= 1
                    count_joint[i,1,j] += 1
    
    f_counts = np.array([f[0]+f[1] for f in count_f])
    total_l_count = sum(count_l)
    total_joint_count = sum(count_joint)

    p_f = count_f / np.vstack((f_counts, f_counts)).T
    p_l = count_l / total_l_count
    p_joint = count_joint / total_joint_count

    S = np.zeros(p_f.shape[0])
    for i, f_dist in enumerate(p_f):
        p_fl = np.multiply(f_dist.reshape((-1,1)), p_l.reshape((1, -1)))
        p_joint_f = p_joint[i]
        p_fl += 1e-5
        p_joint_f += 1e-5
        S[i] = sum(entropy(p_fl, qk=p_joint_f))
    return S

from sklearn import preprocessing
def prepare_labels(label_list):
    labels = set()
    for label in label_list:
        labels.update(label.keys())
    sorted_labels = sorted(labels)

    le = preprocessing.LabelEncoder()
    le.fit(sorted_labels)
    Y = np.array([le.transform(label.keys())[0] for label in label_list])
    return Y, sorted_labels

def get_ghega_train(patents_or_datasheets):
    feature_list, label_list = process_ghega_data(patents_or_datasheets)

    X, feature_names = vectorize(feature_list)
    Y, label_names = prepare_labels(label_list)

    return (X, Y), (feature_names, label_names)

import sklearn.model_selection as ms


#######################
# LEARNING ALGORITHMS #
#######################

# Decision Trees
from sklearn.tree import DecisionTreeClassifier
def train_DT(depth, X, Y, classifier_results=[]):
    X_train, X_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.1)

    dt_model = DecisionTreeClassifier(max_depth=depth)
    dt_model.fit(X_train, y_train)
    y_pred_test = dt_model.predict(X_test)
    train_score = dt_model.score(X_train, y_train)
    test_score = dt_model.score(X_test, y_test)

    classifier_results.append({'Classifier': 'DecTree',
                               'Depth': depth,
                               'Score': test_score})
    return dt_model, train_score, test_score

# kNN
from sklearn.neighbors import KNeighborsClassifier
def train_kNN(k, X, Y, classifier_results=[]):
    X_train, X_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.1)
    
    knn_model = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree')
    knn_model.fit(X_train, y_train)
    y_pred_test = knn_model.predict(X_test)
    train_score = knn_model.score(X_train, y_train)
    test_score = knn_model.score(X_test, y_test)

    classifier_results.append({'Classifier': 'kNN', 'k':k, 'Score': test_score})
    return knn_model, train_score, test_score

# Logistic Regression
from sklearn.linear_model import LogisticRegression
def train_LR(penalty, X, Y, classifier_results=[]):
    X_train, X_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.1)

    lr_model = LogisticRegression(solver='liblinear', penalty=penalty)
    lr_model.fit(X_train, y_train)
    y_pred_test = lr_model.predict(X_test)
    train_score = lr_model.score(X_train, y_train)
    test_score = lr_model.score(X_test, y_test)
    
    classifier_results.append({'Classifier': 'LogReg-{}'.format(penalty.upper()), 'Score': test_score})
    return lr_model, train_score, test_score

# SVM
from sklearn.svm import SVC
def train_SVM(X, Y, classifier_results=[]):
    X_train, X_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.1)

    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    y_pred_test = svm_model.predict(X_test)
    train_score = svm_model.score(X_train, y_train)
    test_score = svm_model.score(X_test, y_test)
    
    classifier_results.append({'Classifier': 'SVM', 'Score': test_score})
    return svm_model, train_score, test_score

# RF
from sklearn.ensemble import RandomForestClassifier
def train_RF(n_estimators, X, Y, classifier_results=[]):
    X_train, X_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.1)

    rfc_model = RandomForestClassifier(n_estimators=n_estimators)
    rfc_model.fit(X_train, y_train)
    y_pred_test = rfc_model.predict(X_test)
    train_score = rfc_model.score(X_train, y_train)
    test_score = rfc_model.score(X_test, y_test)

    classifier_results.append({'Classifier': 'RandomForest', 'Count': n_estimators, 'Score': test_score})
    return rfc_model, train_score, test_score

# Bagging
from sklearn.ensemble import BaggingClassifier
def train_bagging(base_estimator, label, n_estimators, X, Y, classifier_results=[]):
    X_train, X_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.1)

    bag_model = BaggingClassifier(base_estimator=base_estimator, n_estimators=n_estimators)
    bag_model.fit(X_train, y_train)
    y_pred_test = bag_model.predict(X_test)
    train_score = bag_model.score(X_train, y_train)
    test_score = bag_model.score(X_test, y_test)
    
    classifier_results.append({'Classifier': 'Bag-{}'.format(label), 'Count': n_estimators, 'Score': test_score})
    return bag_model, train_score, test_score

# Boosting
from sklearn.ensemble import AdaBoostClassifier
def train_boosting(base_estimator, label, n_estimators, X, Y, classifier_results=[]):
    X_train, X_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.1)

    boost_model = AdaBoostClassifier(base_estimator=base_estimator, algorithm='SAMME', n_estimators=n_estimators)
    boost_model.fit(X_train, y_train)
    y_pred_test = boost_model.predict(X_test)
    train_score = boost_model.score(X_train, y_train)
    test_score = boost_model.score(X_test, y_test)
    
    classifier_results.append({'Classifier': 'Boost-{}'.format(label), 'Count': n_estimators, 'Score': test_score})
    return boost_model, train_score, test_score

# Neural Network
from sklearn.neural_network import MLPClassifier
def train_DNN(hidden, X, Y, classifier_results=[]):
    X_train, X_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.1)

    mlp_model = MLPClassifier(hidden)
    mlp_model.fit(X_train, y_train)
    y_pred_test = mlp_model.predict(X_test)
    train_score = mlp_model.score(X_train, y_train)
    test_score = mlp_model.score(X_test, y_test)

    classifier_results.append({'Classifier': 'DNN', 'Hidden': hidden, 'Score': test_score})
    return mlp_model, train_score, test_score


def train_all_models(X, Y):
    classifier_results = []
    '''
    print('Training DT models...')
    depths = range(10,31,2)
    for depth in depths:
        dt_model, dt_train_score, dt_test_score = train_DT(depth, X, Y, classifier_results=classifier_results)
        print('DT-{}_train_accuracy={}'.format(depth, dt_train_score))
        print('DT-{}_test_accuracy={}'.format(depth, dt_test_score))
    print('\n')
    '''
    print('Training kNN model...')
    ks = [1, 5, 10, 15]
    for k in ks:
        knn_model, knn_train_score, knn_test_score = train_kNN(k, X, Y, classifier_results=classifier_results)
        print('kNN-{}_train_accuracy={}'.format(k, knn_train_score))
        print('kNN-{}_test_accuracy={}'.format(k, knn_test_score))
    print('\n')
    
    print('Training LR models...')
    penalties = ['l1', 'l2']
    for penalty in penalties:
        lr_model, lr_train_score, lr_test_score = train_LR(penalty, X, Y, classifier_results=classifier_results)
        print('LR-{}_train_accuracy={}'.format(penalty, lr_train_score))
        print('LR-{}_test_accuracy={}'.format(penalty, lr_test_score))
    print('\n')

    print('Training SVM model...')
    svm_model, svm_train_score, svm_test_score = train_SVM(X, Y, classifier_results=classifier_results)
    print('SVM_train_accuracy={}'.format(svm_train_score))
    print('SVM_test_accuracy={}'.format(svm_test_score))
    print('\n')

    print('Training DNN model...')
    hidden = (20,10,5)
    dnn_model, dnn_train_score, dnn_test_score = train_DNN(hidden, X, Y, classifier_results=classifier_results)
    print('DNN-{}_train_accuracy={}'.format('-'.join([str(l) for l in hidden]), dnn_train_score))
    print('DNN-{}_test_accuracy={}'.format('-'.join([str(l) for l in hidden]), dnn_test_score))
    print('\n')
    

    n_estimators = 31
    base_estimator, label = (DecisionTreeClassifier(), 'DecTree')

    print('Training RF model...')
    rfc_model, rfc_train_score, rfc_test_score = train_RF(n_estimators, X, Y, classifier_results=classifier_results)
    print('RF_train_accuracy={}'.format(rfc_train_score))
    print('RF_test_accuracy={}'.format(rfc_test_score))
    print('\n')
    
    print('Training Bagging model...')
    bag_model, bag_train_score, bag_test_score = train_bagging(base_estimator, label, n_estimators, X, Y, classifier_results=classifier_results)
    print('Bag-{}_train_accuracy={}'.format(label, bag_train_score))
    print('Bag-{}_test_accuracy={}'.format(label, bag_test_score))
    print('\n')

    print('Training Boosting model...')
    boost_model, boost_train_score, boost_test_score = train_boosting(base_estimator, label, n_estimators, X, Y, classifier_results=classifier_results)
    print('Boost-{}_train_accuracy={}'.format(label, bag_train_score))
    print('Boost-{}_test_accuracy={}'.format(label, bag_test_score))
    print('\n')
    

    return classifier_results

def main():
    MyApp().run()


if __name__ == "__main__":
    main()







