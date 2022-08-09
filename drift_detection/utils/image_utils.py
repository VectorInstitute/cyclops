import torch
import torchxrayvision as xrv
import torchvision
import skimage

def dicom2jpeg(dicom_path, image_size=512, normalization=1024):
    return NotImplementedError

def jpeg2tensor(img_path, image_size=224, normalization=255):

    img = skimage.io.imread(img_path)
    img = xrv.datasets.normalize(img, normalization)
    img = img[None, :, :]
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224)])
    img = transform(img)
    img = torch.from_numpy(img).unsqueeze(0)
    return img

