from argparse import ArgumentParser

from src.dataset import Dataset
from src.encode import encoder_fn
from src.nerf_model import create_nerf_model
from src.nerf_trainer import NeRFTrainer
from src.renderer import Renderer
from src.sampler import Sampler

from pathlib import Path

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument("--dir", required=True)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    BATCH_SIZE = 1024
    L_XYZ = 10
    L_DIR = 4
    DENSE_UNITS = 256
    SKIP_LAYER = 4
    N_COARSE_SAMPLES = 64
    N_FINE_SAMPLES = 128
    NEAR = 2
    FAR = 6

    dataset = Dataset(args.dir, near=NEAR, far=FAR, n_coarse_samples=N_COARSE_SAMPLES)
    
    renderer = Renderer(batch_size=BATCH_SIZE, image_width=dataset.img_width, image_height=dataset.img_height)
    sampler = Sampler(batch_size=BATCH_SIZE, image_width=dataset.img_width, image_height=dataset.img_height)

    coarse_model = create_nerf_model(n_dim_xyz=L_XYZ, n_dim_dir=L_DIR, batch_size=BATCH_SIZE, dense_units=DENSE_UNITS, skip_layer=SKIP_LAYER)
    fine_model = create_nerf_model(n_dim_xyz=L_XYZ, n_dim_dir=L_DIR, batch_size=BATCH_SIZE, dense_units=DENSE_UNITS, skip_layer=SKIP_LAYER)

    trainer = NeRFTrainer(
        coarse_model=coarse_model,
        fine_model=fine_model,
        l_xyz=L_XYZ,
        l_dir=L_DIR,
        encoder_fn=encoder_fn,
        render_img_depth=renderer.composite_ray_samples,
        sample_pdf=sampler.sample_pdf,
        n_fine_samples=N_FINE_SAMPLES
    )
