import os.path, sys
from data.base_dataset import BaseDataset, get_transform
#TODO: add opt.max_dataset_size option in make_dataset
from torchvision.datasets.folder import make_dataset
from PIL import Image
import random

class ClassUnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets with labels.

    It requires two directories to host training images from domain A '/path/to/data/trainA/[class]'
    and from domain B '/path/to/data/trainB/[class]' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA/[class]' and '/path/to/data/testB/[class]' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.classes_A, self.class_to_idx_A = self._find_classes(self.dir_A) # find classes in  '/path/to/data/trainA'
        self.classes_B, self.class_to_idx_B = self._find_classes(self.dir_B) # find classes in  '/path/to/data/trainB'
        samples_A = make_dataset(self.dir_A, self.class_to_idx_A, extensions=self.img_extension, is_valid_file=None) # samples (list): List of (sample path, class_index) tuples
        samples_B = make_dataset(self.dir_B, self.class_to_idx_B, extensions=self.img_extension, is_valid_file=None) # samples (list): List of (sample path, class_index) tuples
        self.A_paths = [s[0] for s in samples_A]
        self.B_paths = [s[0] for s in samples_B]
        self.A_targets = [s[1] for s in samples_A]
        self.B_targets = [s[1] for s in samples_B]
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        A_target = self.A_targets[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        B_target = self.B_targets[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        # TODO: in some cases, might need to implemenet target transfrom

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_target': A_target, 'B_target': B_target}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
