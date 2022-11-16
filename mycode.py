from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import unary_union
from tqdm.notebook import tqdm
from PIL import Image

import nibabel as nib
import numpy as np

import datetime
import pydicom
import pathlib
import skimage.measure
import typing
import json
import copy
import cv2
import os


from collections import defaultdict
import shutil
import regex as re


def load_json(fp):
    '''
    Загрузить json
    Parameters
    ==========
    fp: Pathlike
        Путь до json'a
    '''
    with open(fp, 'r') as f:
        return json.load(f)


def dump_json(obj, fp):
    '''
    Сохранить json
    Parameters
    ==========
    obj: dict
        Словарь для сохранения
    fp: Pathlike
        Путь до json'a
    '''
    with open(fp, 'w') as f:
        json.dump(obj, f)


def _create_sub_masks(mask_array: np.array):
    '''
    Извлечение подмасок (для каждого уникального значения), mask_array двумерная
    На выходе получаем словарь вида {значение: матрица mask_array == значение, с добавленным по бокам паддингом}
    '''
    assert mask_array.ndim == 2

    unique_vals = np.unique(mask_array)

    sub_masks = {}
    for val in unique_vals:
        if val == 0:
            continue

        sub_mask = (mask_array == val)
        # Note: we add 1 pixel of padding in each direction
        # because the contours module doesn't handle cases
        # where pixels bleed to the edge of the image
        sub_mask = np.pad(sub_mask, pad_width=1,
                          mode='constant', constant_values=False)
        sub_masks[int(val)] = sub_mask

    return sub_masks


def _create_sub_mask_annotation(sub_mask: np.array,
                                filename: str,
                                category_id: int,
                                is_crowd: int,
                                tolerance: float = 2.0,
                                tolerance_scaling: bool = True):
    '''
    Аннотация для одной под-маски

    tolerance: float
        Насколько сильно мы упрощаем форму полигонов (0 - вообще не упрощаем)
    tolerance_scaling: bool
        Для маленьких фигур tolerance будет больше (и фигура более упращена)
    '''

    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)

    full_area = sub_mask.shape[0] * sub_mask.shape[1]

    contours = skimage.measure.find_contours(
        sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    areas = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        area = poly.area

        cur_tolerance = tolerance
        if tolerance_scaling:
            scale = 1 - area / full_area
            cur_tolerance = tolerance * scale * 2
        poly = poly.simplify(cur_tolerance, preserve_topology=False)

        if isinstance(poly, MultiPolygon):
            polygons.extend(list(poly.geoms))
        else:
            polygons.append(poly)
        if getattr(poly, 'exterior', None) is None:
            continue
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        if segmentation:
            areas.append(area)
            segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    if not multi_poly.bounds:
        return {}
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    area = multi_poly.area

    bbox = (x, y, width, height)

    unary_seg = mapping(unary_union(polygons))['coordinates']
    annotation = {
        'segmentation': [np.array(seg).flatten().tolist() for seg in unary_seg],
        'iscrowd': is_crowd,
        'filename': filename,  # Будет заменено на image_id в convert_images_to_coco
        'category_id': category_id,
        'bbox': bbox,
        'area': area
    }

    return annotation


def _get_annotation(mask_dict: dict):
    '''
    Аннотация для одной маски

    mask_dict: dict
        либо содержит ключи mask_path (Path), либо img_name (str) и mask (array)
        содержит tolerance (насколько сильно мы упрощаем форму полигонов)
    '''

    annotations = []

    if 'mask_path' in mask_dict:
        mask_path = mask_dict['mask_path']
        mask_dict.pop('mask_path')

        mask_image = Image.open(mask_path)
        mask_image = np.array(mask_image)
        if mask_image.ndim == 3:
            mask_image = mask_image[..., 0]

        filename = mask_path.name.replace('png', 'jpg')
    else:

        filename = mask_dict['img_name']
        mask_image = mask_dict['mask']
        mask_dict.pop('img_name')
        mask_dict.pop('mask')

    is_crowd = 0

    num_channels = 1 if mask_image.ndim == 2 else mask_image.shape[2]

    for channel in range(num_channels):

        if mask_image.ndim == 2:
            mask_one_dim = mask_image
        else:
            mask_one_dim = mask_image[..., channel]

        sub_masks = _create_sub_masks(mask_one_dim)
        for category_id, sub_mask in sub_masks.items():

            annotation = _create_sub_mask_annotation(
                sub_mask, filename, category_id, is_crowd, **mask_dict)
            if annotation and annotation['segmentation']:
                annotations.append(annotation)

    return annotations


def convert_images_to_vgg(output_json: os.PathLike,
                          img_dir: pathlib.Path,
                          mask_path: typing.Optional[pathlib.Path] = None,
                          filenames_to_masks_dict: typing.Optional[typing.Dict[pathlib.Path, np.array]] = None,
                          supercategory_name: str = '',
                          description: str = 'Some COCO file',
                          tolerance: float = 2.0,
                          tolerance_scaling: bool = False):
    '''
    Создает на основе масок json в формате VGG

    Parameters
    ==========
    output_json: str or pathlib.Path
        Путь для сохранения json'а
    mask_path: Optional, pathlib.Path
        Путь до папки с масками
    filenames_to_masks_dict: Optional, dict[str, numpy.array]
        Словарь с путями до фото и масками (маски в каждом канале имеют значения для соответствующего инстанса)
    supercategory_name: str
        Тип классов в мета-данных
    description: str
        Описание в мета-данных
    tolerance: float
        Насколько сильно мы упрощаем форму полигонов (0 - вообще не упрощаем)
        https://shapely.readthedocs.io/en/stable/manual.html#object.simplify
    tolerance_scaling: bool
        Для маленьких фигур tolerance будет больше (и фигура более упращена)
    '''

    assert mask_path is not None or filenames_to_masks_dict is not None

    if mask_path is not None:
        ann_input = [{'mask_path': fp}
                     for fp in mask_path.glob('*.png')]
        filenames = [fp.name.replace('png', 'jpg')
                     for fp in mask_path.glob('*.png')]
        photo_paths = list(mask_path.glob('*.png'))

    else:
        ann_input = [{'img_name': fp.name,
                      'mask': mask}
                     for fp, mask in filenames_to_masks_dict.items()]
        filenames = [fp.name for fp in filenames_to_masks_dict]
        photo_paths = list(filenames_to_masks_dict.keys())

    for d in ann_input:
        d['tolerance'] = tolerance
        d['tolerance_scaling'] = tolerance_scaling

    list_of_annotations = [_get_annotation(fp) for fp in tqdm(ann_input)]

    # Присваиваем уникальные annotation_id и image_id
    filename_to_id = {name: i
                      for i, name in enumerate(filenames)}

    all_annotations = []
    annotations_counter = 0
    unique_categories = set()

    for annotations in list_of_annotations:
        for annotation in annotations:
            for seg in annotation['segmentation']:
                ann_new = copy.deepcopy(annotation)
                ann_new['segmentation'] = seg
                ann_new['id'] = annotations_counter
                annotations_counter += 1
                ann_new['image_id'] = filename_to_id[ann_new['filename']]
                ann_new.pop('filename')
                all_annotations.append(ann_new)
            unique_categories.add(annotation['category_id'])

    categories = [{'id': category,
                   'name': str(category),
                   'supercategory': supercategory_name} for category in unique_categories]

    images = []
    for fp in photo_paths:
        photo = Image.open(fp)
        filename = fp.name.replace('png', 'jpg')

        images.append({
            "date_captured": "",
            "file_name": filename,
            "height": photo.size[1],
            "id": filename_to_id[filename],
            "license": 1,
            "width": photo.size[0]})

    cur_time = datetime.datetime.now()

    info = {
        "contributor": "aboba",
        "date_created": str(cur_time),
        "description": description,
        "url": "",
        "version": "0.1.0",
        "year": str(cur_time.year)
    }

    licenses = [
        {
            "id": 1,
            "name": "aboba",
            "url": ""
        }
    ]

    coco = {'annotations': all_annotations,
            'categories': categories,
            'images': images,
            'info': info,
            'licenses': licenses}

    vgg = convert_coco_to_vgg(ann_dict=coco, img_dir=img_dir)

    dump_json(vgg, output_json)


def _vgg_to_color_mask(ann_path: pathlib.Path,
                       photos_path: typing.Optional[pathlib.Path] = None,
                       class_color_func: typing.Callable[[
                           int], tuple] = lambda cls: (cls, 0, 0),
                       cls_for_anns_without_name: typing.Optional[int] = None,
                       is_panoptic: bool = False):
    """
    Преобразует json в цветные маски

    Parameters
    ----------
    ann_path : pathlib.Path
        Путь к аннотациям
    photos_path : typing.Optional[pathlib.Path], optional
        Путь до папки с изображениями, by default None
    class_color_func : _type_, optional
        Функция закрашивания класса, by default lambda cls:(cls, 0, 0)
    cls_for_anns_without_name : typing.Optional[int], optional
        Классы без названия, by default None
    is_panoptic : bool, optional
        Является ли это паноптической моделью, by default False
    """
    
    if photos_path is None:
        photos_path = ann_path.parent
    ann = load_json(ann_path)
    masks = []
    image_fps = []
    # Каждая итерация - отдельное фото
    for data in tqdm(ann.values()):
        # Номер класса -> cписок координат ломаных
        area_dict = defaultdict(lambda: [])
        # Чекаем каждую ломаную
        lines = data['regions']
        if isinstance(lines, dict):
            lines = lines.values()
        for val in lines:
            # Название класса ломаной : номер класса
            # (может быть несколько классов, хз почему)
            cls_names = [name for name,
                         num in val['region_attributes'].items() if num]
            cls_numbers = [
                num for num in val['region_attributes'].values() if num]

            cls = np.nan

            if not cls_numbers:
                pass
            elif not cls_numbers[0]:
                if len(cls_numbers) == 1:
                    cls_match = re.search(r'(\d+)_', cls_names[0])
                    if cls_match:
                        cls = int(cls_match.group(1))

                else:
                    cls = int(cls_numbers[1])
            else:
                cls = int(cls_numbers[0])

            if np.isnan(cls) and cls_for_anns_without_name is None:
                continue
            elif np.isnan(cls):
                cls = cls_for_anns_without_name
            if 'shape_attributes' not in val:
                continue
            val = val['shape_attributes']
            if 'all_points_x' not in val or 'all_points_y' not in val:
                continue
            x = val['all_points_x']
            y = val['all_points_y']
            area = np.stack((x, y)).T
            area_dict[cls].append(area)
        # Открываем фото чтобы знать его размер
        img_path = photos_path / data['filename']
        if not img_path.exists():
            print(f'Not found: {img_path}')
            continue
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = np.array(img)
        mask = img.copy()
        mask[...] = (0, 0, 0)

        if is_panoptic:
            assert len(
                area_dict) == 1, 'Мультикласс еще не поддерживается для is_panoptic'

        cls_and_areas = list(area_dict.items())

        for cls, areas in cls_and_areas:
            for i, area in enumerate(areas):
                if is_panoptic:
                    color = (i + 1, 0, 0)
                else:
                    color = class_color_func(cls)

                mask = cv2.fillPoly(
                    mask.copy(), pts=np.int32([area]), color=color)

        masks.append(mask)
        image_fps.append(img_path)
    return masks, image_fps


def _save_masks_and_images(masks: typing.List[np.array],
                           image_fps: typing.List[pathlib.Path],
                           output_dir: pathlib.Path,
                           save_originals=True):
    '''
    Сохраняет все в формате output_dir/{filename}.jpg, output_dir/{filename}.png (для фото и маски соответственно)
    '''
    output_dir.mkdir(exist_ok=True)
    for i, (mask, image_path) in enumerate(zip(masks, image_fps)):
        filename = re.search(r'(.+)\.', image_path.name).group(1)
        new_image_path = output_dir / f'{filename}.jpg'
        if save_originals:
            shutil.copy(image_path, new_image_path)
        if len(mask.shape) == 3:
            mask = mask.sum(axis = 2)
        mask = Image.fromarray(np.uint8(mask))
        mask.save(output_dir / f'{filename}.png')


def convert_vgg_to_images(ann_paths: typing.List[pathlib.Path],
                          output_path: pathlib.Path,
                          photos_path: typing.Optional[pathlib.Path] = None,
                          class_color_func: typing.Callable[[
                              int], tuple] = lambda cls: (cls, cls, cls),
                          is_panoptic: bool = False,
                          ignore_empty_masks: bool = False,
                          save_originals: bool = False,
                          cls_for_anns_without_name: typing.Optional[int] = None):
    '''
    Создать маски на основе json'ов в формате VGG

    Parameters
    ==========
    ann_paths: pathlib.Path[]
        Список путей до json'ов
    output_path: pathlib.Path
        папка в которую сохраняются маски
    photos_path: pathlib.Path
        Папка с фото (если не указана, то это папки, в которых находятся ann_paths)
    class_color_func: Callable
        Функция, принимающая номер класса и выдающая значения пикселя в маске
    is_panoptic: bool
        Временный способ делать мульти-инстанс маски: тогда class_color_func игнорится и пиксели будут вида (instance_id, instance_id, instance_id)
    ignore_empty_masks: bool
        Не создавать масок, если разметка пустая
    save_originals: bool
        Сохранять ли фото в photos_path
    cls_for_anns_without_name: int
        Если у аннотации нет названия класса, использовать это значений (по умолчанию игнорировать такие аннотации)
    '''
    masks, image_fps = [], []
    for i, ann_path in enumerate(ann_paths):
        new_masks, new_image_fps = _vgg_to_color_mask(ann_path,
                                                      photos_path,
                                                      class_color_func,
                                                      cls_for_anns_without_name,
                                                      is_panoptic)
        if new_masks is None:
            continue
        masks.extend(new_masks)
        image_fps.extend(new_image_fps)
    if ignore_empty_masks:
        ids = [i for i, mask in enumerate(masks)
               if mask.sum() > 0]
        masks = [masks[i] for i in ids]
        image_fps = [image_fps[i] for i in ids]
    _save_masks_and_images(masks, image_fps, output_path,
                           save_originals=save_originals)


def convert_coco_to_vgg(ann_dict: dict,
                        img_dir: pathlib.Path):
    """
    Конвертация json из формата coco в формат vgg

    Parameters
    ----------
    ann_dict: dict
        Исходный json
    img_dir: pathlib.Path
        Путь к папке с фото
    """

    new_ann_dict = {}
    
    image_id_to_dict = {d['id']: d for d in ann_dict['images']}

    image_id_to_ann_list = defaultdict(list)

    for ann in ann_dict['annotations']:
        image_id_to_ann_list[int(ann['image_id'])].append(ann)
    for ann_id in (set([int(id_) for id_ in image_id_to_dict]) - set(image_id_to_ann_list)):
        image_id_to_ann_list[ann_id].append(dict())

    for image_id, ann_list in image_id_to_ann_list.items():
        image_dict = image_id_to_dict[image_id]

        filename = image_dict['file_name']

        try:
            size = os.stat(img_dir / filename).st_size
        except Exception:
            print(filename)
            continue

        new_ann = {'fileref': '',
                   'filename': filename,
                   'size': size,
                   'regions': {},
                   'base64_img_data': '',
                   'file_attributes': {}}

        new_ann_dict[f'{filename}{size}'] = new_ann
        for i, ann in enumerate(ann_list):
            try:
                cat_id = ann['category_id']
    
                region = {'shape_attributes': {'name': 'polygon'},
                          'region_attributes': {'category': cat_id}}
    
                region['shape_attributes']['all_points_x'] = ann['segmentation'][::2]
                region['shape_attributes']['all_points_y'] = ann['segmentation'][1::2]
    
                new_ann['regions'][str(i)] = region
            except KeyError:
                continue

    return new_ann_dict


def nii2images(filename: str,
               output_path: str,
               mask_flag: bool = False):
    """
    Трансформирует nii в фото

    Parameters
    ----------
    filename : str
        Имя файла
    output_path : str
        Путь сохранения фото
    mask_flag : bool, optional
        На вход подаются маски, by default False
    """

    os.makedirs(output_path, exist_ok=True)

    data = np.asanyarray(nib.load(filename).get_fdata()).T
    data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
    postfix = '.png' if mask_flag else '.jpg'
    for n, im in enumerate(data):
        cv2.imwrite(f'{output_path}/{str(n).zfill(5)}{postfix}', im)


def convertNsave(arr: np.array,
                 file_dir: str,
                 index: int = 0):
    """
    Создаёт dicom по шаблону и сохраняет

    Parameters
    ----------
    arr: np.array
        parameter will take a numpy array that represents only one slice.
    file_dir: str
        parameter will take the path to save the slices
    index: int
        parameter will represent the index of the slice, so this parameter will be used to put
        the name of each slice while using a for loop to convert all the slices
    """

    dicom_file = pydicom.dcmread('./data/dcmimage.dcm')
    arr = arr.astype('uint16')
    dicom_file.Rows = arr.shape[0]
    dicom_file.Columns = arr.shape[1]
    dicom_file.PhotometricInterpretation = "MONOCHROME2"
    dicom_file.SamplesPerPixel = 1
    dicom_file.BitsStored = 16
    dicom_file.BitsAllocated = 16
    dicom_file.HighBit = 15
    dicom_file.PixelRepresentation = 1
    dicom_file.PixelData = arr.tobytes()
    dicom_file.save_as(os.path.join(file_dir, f'{str(index).zfill(5)}.dcm'))


def nifti2dicom_1file(nifti_dir, out_dir):
    """
    Эта функция предназначена для преобразования одного файла nifti в серию dicom

    Parameters
    ----------
    nifti_dir:
        путь к одному файлу nifti
    out_dir:
        Путь сохранения файлов
    """

    nifti_file = nib.load(nifti_dir)
    nifti_array = nifti_file.get_fdata()
    number_slices = nifti_array.shape[2]
    os.makedirs(out_dir, exist_ok=True)

    for slice_ in tqdm(range(number_slices)):
        convertNsave(nifti_array[:, :, slice_], out_dir, slice_)
