import logging
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel, AutoProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.dsprites import DSprites
from src.dataset.shapes3d import Shapes3D
from src.dataset.pentominos import Pentominos, Pentominos3D
from src.dataset.utils import build_filter


torch.set_grad_enabled(False)
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


DATASETS = {
    'dsprites': (DSprites, "data/datasets/dsprites/dsprites_train.hdf5"),
    'shapes3d': (Shapes3D, "data/datasets/shapes3d/3dshapes.h5"),
    'pentominos': (Pentominos, "data/datasets/pentominos"),
    'pentominos_colored': (Pentominos, "data/datasets/pentominos_colored"),
    'pentominos_colored_black_background': (Pentominos, "data/datasets/pentominos_colored_black_background"),
    'pentominos_3d': (Pentominos3D, str(Path.home() / "Repos/vision/blender stuff/data/temp/pentominos_3d")),
}


class HFVisionEmbedder:
    """Wrapper around a HuggingFace vision model exposing `.embed(x)` for `compute_embeddings`."""

    def __init__(self, model_id: str, device: torch.device):
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.model.eval()
        self.device = device

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        inputs = self.processor(images=x, return_tensors="pt", do_rescale=False).to(self.device)
        outputs = self.model(**inputs)
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_embeddings(model, loader, device, progress_callback=None):
    latents = []
    total = len(loader)
    with torch.inference_mode():
        for i, (x, _) in enumerate(tqdm(loader, desc="Computing embeddings")):
            latents.append(model.embed(x.to(device)).cpu().numpy())
            if progress_callback is not None:
                progress_callback(i + 1, total)
    return np.concatenate(latents, axis=0).squeeze()


def load_or_compute_embeddings(
    embeddings_path: Path,
    model,
    loader,
    device,
    force_recompute: bool = False,
    split_name: str = "train"
):
    key = f"{split_name}_embeddings"

    if embeddings_path.exists() and not force_recompute:
        _logger.info(f"Loading cached embeddings from {embeddings_path}")
        data = np.load(embeddings_path)
        if key in data:
            return data[key].squeeze()
        if 'train_embeddings' in data and split_name == 'train':
            return data['train_embeddings'].squeeze()

    _logger.info(f"Computing {split_name} embeddings...")
    embeddings = compute_embeddings(model, loader, device)

    existing_data = {}
    if embeddings_path.exists():
        existing_data = dict(np.load(embeddings_path))

    existing_data[key] = embeddings
    _logger.info(f"Saving embeddings to {embeddings_path}")
    np.savez(embeddings_path, **existing_data)

    return embeddings


def _instantiate_dataset(dataset_class, path: str, data_filter=None):
    if dataset_class is Pentominos3D:
        return dataset_class(path, held_out_filter=data_filter)
    return dataset_class(path, data_filter=data_filter)


def load_dataset(
    dataset_name: str,
    filter_expr: str | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
    dataset_path: str | None = None,
):
    dataset_class, default_path = DATASETS[dataset_name]
    path = dataset_path or default_path

    if not filter_expr:
        full_dataset = _instantiate_dataset(dataset_class, path, data_filter=None)
        full_loader = DataLoader(full_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        return full_dataset, None, full_loader, None

    held_out_filter = build_filter(dataset_class, filter_expr)
    train_filter = lambda x: ~held_out_filter(x)

    train_dataset = _instantiate_dataset(dataset_class, path, data_filter=train_filter)
    test_dataset = _instantiate_dataset(dataset_class, path, data_filter=held_out_filter)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader


def load_vlm(model_id, device, compile=True):
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModel.from_pretrained(model_id, device_map=device, attn_implementation="sdpa")
    if compile:
        model = torch.compile(model)
    return model, processor


def tensor_to_pil(tensor):
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    img = (tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(img)


def get_shape_name(dataset, shape_value):
    return dataset.map_shapes(np.array([shape_value]))[0]
