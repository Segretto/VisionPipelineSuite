import os
import io
from typing import Any, Callable, Optional, Tuple, Union
from enum import Enum
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from dinov3.data.datasets.extended import ExtendedVisionDataset
from dinov3.data.datasets.decoders import Decoder, ImageDataDecoder, DenseTargetDecoder

class CocoSegmentation(ExtendedVisionDataset):
    class Split(Enum):
        TRAIN = "train"
        VAL = "val"
        TEST = "test"

    def __init__(
        self,
        split: "CocoSegmentation.Split",
        root: str,
        extra: str, # Path to the COCO JSON file
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_decoder: Decoder = ImageDataDecoder,
        target_decoder: Decoder = DenseTargetDecoder,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=image_decoder,
            target_decoder=target_decoder,
        )
        
        self.coco = COCO(extra)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.root = root

    def get_image_data(self, index: int) -> bytes:
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        
        with open(os.path.join(self.root, path), 'rb') as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Any:
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Create a mask
        mask = np.zeros((img_metadata['height'], img_metadata['width']), dtype=np.uint8)
        
        for ann in anns:
            # Skip crowd annotations for now if needed, or handle them
            pixel_value = ann['category_id'] # Assuming category_ids are 1-indexed and fit in uint8
            # Note: DINOv3 might expect specific class mapping. 
            # For now, we use category_id directly. 
            # Users might need to remap if IDs are sparse or large.
            
            m = coco.annToMask(ann)
            mask[m == 1] = pixel_value
            
        # Convert mask to bytes (PNG format) for the decoder
        img = Image.fromarray(mask)
        with io.BytesIO() as output:
            img.save(output, format="PNG")
            target_data = output.getvalue()
            
        return target_data

    def __len__(self) -> int:
        return len(self.ids)
