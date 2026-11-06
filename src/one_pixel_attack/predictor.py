import numpy as np
import torch


class CNNPredictor:
    def __init__(self, model, mean, std, device="cpu", batch_size=64):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        if batch_size < 1 or len(mean) != 3 or len(std) != 3 or min(std) <= 0:
            raise ValueError("invalid batch size or RGB normalization")
        self.batch_size = batch_size

        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device).view(
            1, 3, 1, 1
        )

        self.std = torch.tensor(std, dtype=torch.float32, device=self.device).view(
            1, 3, 1, 1
        )

    @torch.inference_mode()
    def __call__(self, images):

        # Input: (N, H, W, 3)

        outputs = []

        for start in range(0, len(images), self.batch_size):
            batch = np.ascontiguousarray(images[start : start + self.batch_size])

            x = torch.from_numpy(batch).to(self.device)
            x = x.permute(0, 3, 1, 2).float() / 255.0
            x = (x - self.mean) / self.std

            logits = self.model(x)
            probabilities = logits.softmax(dim=1)
            # Output: probabilities
            outputs.append(probabilities.cpu().numpy())

        return np.concatenate(outputs, axis=0)
