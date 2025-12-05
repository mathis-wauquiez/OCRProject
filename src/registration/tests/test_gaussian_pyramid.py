from ..gaussian_pyramid import *

if __name__ == "__main__":
    pyr = GaussianPyramid(eta=0.5, sigma_0=1.0, N_scales=4)
    batch = torch.randn(2, 3, 256, 256)

    for i, level in enumerate(pyr(batch)):
        print(f"Level {i}: {tuple(level.shape)}")

