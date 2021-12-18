from os import listdir
from os.path import join
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)

def make_dataset_train(root: str) -> list:
    """Reads a directory with data.
    """
    dataset = []

    lr_dir = 'Data/Train_LR'
    hr_dir = 'Data/Train_SR'
    
    # Get all the filenames from lr
    lr_fnames = sorted(os.listdir(os.path.join(root, lr_dir)))
 

    hr_fnames = sorted(os.listdir(os.path.join(root, hr_dir)))
 
    i = 1

    # Compare file names
    for hr_fname in hr_fnames:

      if i < 6 : 

        if hr_fname in lr_fnames:

          # create pair of full path to the corresponding images
          lr_path = os.path.join(root, lr_dir, hr_fname)
          hr_path = os.path.join(root, hr_dir, hr_fname)

          item = (lr_path, hr_path)
                    # append to the list dataset
          dataset.append(item)
          i = i + 1
              
      else:
        continue
      

    return dataset

def make_dataset_test(root: str) -> list:
    """Reads a directory with data.
    """
    dataset = []
    i = 1

    # dir names 
    lr_dir = 'Data/Test_LR'
    hr_dir = 'Data/Test_SR'
    
    # Get all the filenames from lr
    lr_fnames = sorted(os.listdir(os.path.join(root, lr_dir)))

    hr_fnames = sorted(os.listdir(os.path.join(root, hr_dir)))
    i= i+1

    # Compare file names:
    for hr_fname in sorted(os.listdir(os.path.join(root, hr_dir))):
     

      if i < 6:

        if hr_fname in lr_fnames:
          # create pair of full path to the corresponding images
          lr_path = os.path.join(root, lr_dir, hr_fname)
          hr_path = os.path.join(root, hr_dir, hr_fname)

          item = (lr_path, hr_path)
          # append to the list dataset
          dataset.append(item)
          i = i + 1
      else : 
         continue
                 
              

    return dataset


from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.folder import pil_loader

class CustomDatasetTrain(Dataset):
  def __init__(self,
               root = '/content/gdrive/My Drive',
               loader=pil_loader,
               transform=None,
               transform2=None):


      self.root = root
      self.target_transform = transform
      self.target_transform2 = transform2

      # Prepare dataset
      samples = make_dataset_train(self.root)
      
      self.loader = loader
      self.samples = samples
      # list of lr
      self.lr_samples = [s[1] for s in samples]
      # list of hr
      self.hr_samples = [s[1] for s in samples]

  def __getitem__(self, index):
      """Returns a data sample from  dataset.
      """
      # getting our paths to images
      lr_path, hr_path = self.samples[index]
        
      # import each image using loader
      lr_sample = self.loader(lr_path)
      hr_sample = self.loader(hr_path)
        
      # tranforms

      lr_sample = self.target_transform(lr_sample)
      hr_sample = self.target_transform(hr_sample)      


      return lr_sample, hr_sample

  def __len__(self):
      return len(self.samples)


from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

class CustomDatasetTest(Dataset):
  def __init__(self,
               root = '/content/gdrive/My Drive',
               loader=pil_loader,
               transform=None,
               transform2=None):


      self.root = root
      self.target_transform = transform
      self.target_transform2 = transform2

      # Prepare dataset
      samples = make_dataset_test(self.root)

      self.loader = loader
      self.samples = samples
      # list of lr
      self.lr_samples = [s[1] for s in samples]
      # list of hr
      self.hr_samples = [s[1] for s in samples]


  def __getitem__(self, index):
      """Returns a data sample from  dataset.
      """
      # getting paths to images
      lr_path, hr_path = self.samples[index]
        
      # import each image using loader
      lr_sample = self.loader(lr_path)
      hr_sample = self.loader(hr_path)
        
      # tranforms
      lr_sample = self.target_transform(lr_sample)
      hr_sample = self.target_transform(hr_sample)      

      return lr_sample, hr_sample

  def __len__(self):
      return len(self.samples)
