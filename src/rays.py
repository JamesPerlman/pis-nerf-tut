from src.camera import Camera


class GetRays:
    def __init__(self, focal_len, img_width, img_height, near, far, n_samples):
        self.focal_len = focal_len
        self.img_width = img_width
        self.img_height = img_height
        self.near = near
        self.far = far
        self.n_samples = n_samples
    
    def __call__(self, c2w):
        cam = Camera(c2w, self.focal_len, self.img_width, self.img_height)
        return cam.get_rays(self.near, self.far, self.n_samples)
