"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random, math
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot
        self.img_extension = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))

    if 'rotate' in opt.preprocess:
        transform_list.append(transforms.RandomRotation(opt.rot_degree))

    if 'crop' in opt.preprocess:
        if 'center_crop' in opt.preprocess:
            transform_list.append(transforms.CenterCrop(opt.crop_size))
        elif params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    
    if 'color_jitter' in opt.preprocess:
        # transform_list.append(transforms.ColorJitter( #TODO: parametarize
        #             brightness=0.25*5,  #args.jitter_brightness,
        #             contrast=0.4*5, #args.jitter_contrast,
        #             saturation=0.2*5, #args.jitter_saturation,
        #             hue=0.05*5 #args.jitter_hue
        #         ))
        transform_list.append(transforms.ColorJitter( #TODO: parametarize
                    brightness=0.25,  #args.jitter_brightness,
                    contrast=0.4, #args.jitter_contrast,
                    saturation=0.2, #args.jitter_saturation,
                    hue=0.05 #args.jitter_hue
                ))

    transform_list.append(transforms.ToTensor())
    transform_tensor = [transforms.ToPILImage()] + transform_list
    if convert:
        transform_list += [transforms.Normalize((0.5,),
                                                (0.5,))]
    
    #cut or erase after normalization
    if 'cutout' in opt.preprocess:
        transform_list.append(Cutout(n_holes=1, length=8)) #TODO: parametarize
        transform_tensor.append(Cutout(n_holes=1, length=8)) #TODO: parametarize
    
    if 'erasing' in opt.preprocess:
        transform_list.append(RandomErasing(p=0.5, sl=0.02, sh=0.3, r1=0.3, r2=1/0.3)) #TODO: parametarize
        transform_tensor.append(RandomErasing(p=0.5, sl=0.02, sh=0.3, r1=0.3, r2=1/0.3)) #TODO: parametarize

    try:
        if opt.phase=="train" and "multi_transforms" in opt.dataset_mode:
            print("return multi transforms, phase: ", opt.phase)
            non_transform = []
            if grayscale:
                non_transform.append(transforms.Grayscale(1))
            non_transform.extend([
                transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)),
                transforms.Resize(opt.crop_size, method), #TODO: adjust size to crop size because it's same if crop is specified
                transforms.ToTensor(),
                transforms.Normalize((0.5,),
                                     (0.5,))
            ])

            return transforms.Compose(non_transform), transforms.Compose(transform_list), transforms.Compose(transform_tensor)
    except:
        pass

    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True

# https://arxiv.org/pdf/1708.04552.pdf
# modified from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask_value = img.mean()

        for n in range(self.n_holes):
            top = np.random.randint(0 - self.length // 2, h)
            left = np.random.randint(0 - self.length // 2, w)
            bottom = top + self.length
            right = left + self.length

            top = 0 if top < 0 else top
            left = 0 if left < 0 else left

            img[:, top:bottom, left:right].fill_(mask_value)

        return img
    
    def __repr__(self):
        return "Cutout(n_holes={}, length={})".format(self.n_holes, self.length)

# https://arxiv.org/pdf/1708.04896.pdf
# modified from https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    p: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    r2: max aspect ratio
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, p=0.5, sl=0.02, sh=0.3, r1=0.3, r2=1/0.3):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with Random erasing.
        """
        if np.random.uniform(0, 1) > self.p:
            return img

        area = img.size()[1] * img.size()[2]
        for _attempt in range(100):
            target_area = np.random.uniform(self.sl, self.sh) * area
            aspect_ratio = np.random.uniform(self.r1, self.r2)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = np.random.randint(0, img.size()[1] - h)
                y1 = np.random.randint(0, img.size()[2] - w)
                img[:, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(3, h, w))
                return img

        return img

    def __repr__(self):
        return "RandomErasing(p={}, sl={}, sh={}, r1={}, r2={})".format(self.p,self.sl, self.sh, self.r1, self.r2)
