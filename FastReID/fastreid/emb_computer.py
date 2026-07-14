import cv2
import torch
import time
import numpy as np
from fastreid.fastreid_adaptor import FastReID


class EmbeddingComputer:
    def __init__(self, config_path, weight_path, max_batch=1024):
        self.model = None
        self.config_path = config_path
        self.weight_path = weight_path
        self.crop_size = (128, 384)
        self.max_batch = max_batch

        self.reid_latencies = []
        self.call_counter = 0

    def initialize_model(self):
        self.model = FastReID(self.config_path, self.weight_path)

    def compute_embedding(self, img, bbox):
        # counter
        self.call_counter += 1

        # Initialization
        if self.model is None:
            self.initialize_model()

        torch.cuda.synchronize()
        start_time = time.time()

        # Basic embeddings
        h, w = img.shape[:2]
        bbox_clip = np.round(bbox).astype(np.int32)
        bbox_clip[:, [0, 2]] = bbox_clip[:, [0, 2]].clip(0, w)
        bbox_clip[:, [1, 3]] = bbox_clip[:, [1, 3]].clip(0, h)

        batch_crops_numpy = []
        for box in bbox_clip:
            crop = img[box[1]:box[3], box[0]:box[2]]
            
            if crop.size == 0:
                crop = np.zeros((self.crop_size[1], self.crop_size[0], 3), dtype=np.uint8)
            else:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR)
            
            # Prepare the array (H, W, C) -> (C, H, W)
            crop = crop.transpose(2, 0, 1)
            batch_crops_numpy.append(crop)

        if not batch_crops_numpy:
            return np.array([])

        # Upgrade to gpu only once for all crops
        all_crops_tensor = torch.from_numpy(np.stack(batch_crops_numpy)).to(device='cuda', non_blocking=True).float()

        # run inference in batches
        all_embeddings = []
        for idx in range(0, len(all_crops_tensor), self.max_batch):
            batch = all_crops_tensor[idx : idx + self.max_batch]
            
            with torch.no_grad():
                batch_feat = self.model(batch) 
                all_embeddings.append(batch_feat)

        # result
        embeddings = torch.cat(all_embeddings, dim=0)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)


        torch.cuda.synchronize()
        self.reid_latencies.append(time.time() - start_time)
            
        return embeddings.cpu().numpy()

    def print_summary(self):
        latency_reid = (sum(self.reid_latencies) / len(self.reid_latencies)) * 1000
        speed_reid = 1000 / latency_reid
        print(f"[*] TOTAL PIPELINE FEATURE EXTRACTION:")
        print(f"  - Latency : {latency_reid:.2f} ms/frame")
        print(f"  - Speed   : {speed_reid:.2f} FPS")
