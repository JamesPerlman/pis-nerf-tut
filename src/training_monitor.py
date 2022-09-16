from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

TEST_PATH = Path("/home/jperl/Developer/tutorials/pyimagesearch-nerf-tutorial")

def get_train_monitor(test_ds, encoder_fn, l_xyz, l_dir, image_path):
    # get images and rays from testing dataset
    (t_elements, t_images) = next(iter(test_ds))
    (t_ray_origins_coarse, t_ray_dirs_coarse, t_vals_coarse) = t_elements

    # build the test coarse rays
    t_rays_coarse = t_ray_origins_coarse[..., None, :] + (t_ray_dirs_coarse[..., None, :] * t_vals_coarse[..., None])

    # positionally encode the rays and direction vectors for the coars rays
    t_rays_coarse = encoder_fn(t_rays_coarse, l_xyz)
    t_dirs_coarse_shape = tf.shape(t_rays_coarse[..., :3])
    t_ray_dirs_coarse = tf.broadcast_to(t_ray_dirs_coarse[..., None, :], shape=t_dirs_coarse_shape)
    t_dirs_coarse = encoder_fn(t_dirs_coarse, l_dir)

    class TrainMonitor(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # compute the coarse model prediction
            (t_rgb_coarse, t_sigma_coarse) = self.model.coarse_model.predict([t_rays_coarse, t_dirs_coarse])
            
            # render the image rom the model prediction
            t_render_coarse = self.model.render_img_depth(rgb=t_rgb_coarse, sigma=t_sigma_coarse, t_vals=t_vals_coarse)
            (t_img_coarse, _, t_weights_coarse) = t_render_coarse

            # compute the middle values of t_vals
            t_vals_coarse_mid = 0.5 * (t_vals_coarse[..., 1:] + t_vals_coarse[..., :-1])

            # apply hierarchical sampling and get the t_vals for the fine model
            t_vals_fine = self.model.sample_pdf(t_vals_mid=t_vals_coarse_mid, weights=t_weights_coarse, n_fine=self.model.n_fine_samples)
            t_vals_fine = tf.sort(tf.concat([t_vals_coarse, t_vals_fine], axis=-1), axis=-1)

            # build the fine rays and positionally encode them
            t_rays_fine = t_ray_origins_coarse[..., None, :] + (t_ray_dirs_coarse[..., None, :] * t_vals_fine[..., None])
            t_rays_fine = self.model.encode_fn(t_rays_fine, l_xyz)

            # build the fine directions and positionally encode them
            t_dirs_fine_shape = tf.shape(t_rays_fine[..., :3])
            t_dirs_fine = tf.broadcast_to(t_ray_dirs_coarse[..., None, :], shape=t_dirs_fine_shape)
            t_dirs_fine = self.model.encode_fn(t_dirs_fine, l_dir)

            # compute the fine model prediction
            t_rgb_fine, t_sigma_fine = self.model.fine_model.predict([t_rays_fine, t_dirs_fine])

            # render the image from the model prediction
            t_render_fine = self.model.render_img_depth(rgb=t_rgb_fine, sigma=t_sigma_fine, t_vals=t_vals_fine)
            (t_img_fine, t_depth_fine, _) = t_render_fine

            # plot the coarse image, fine image, fine depth map and target image
            (fig, ax) = plt.subplots(nrows=1, ncols=4, figsize=(10, 10))

            coarse_img = tf.keras.preprocessing.image.array_to_img(t_img_coarse[0])
            tf.keras.preprocessing.image.save_img(image_path / f"{epoch:03d}-coarse.png", coarse_img)
            
            fine_img = tf.keras.preprocessing.image.array_to_img(t_img_fine[0])
            tf.keras.preprocessing.image.save_img(image_path / f"{epoch:03d}-fine.png", coarse_img)

            depth_img = tf.keras.preprocessing.image.array_to_img(t_depth_fine[0])
            tf.keras.preprocessing.image.save_img(image_path / f"{epoch:03d}-depth-fine.png", depth_img)

            real_img = tf.keras.preprocessing.image.array_to_img(t_images[0])
            tf.keras.preprocessing.image.save_img(image_path / f"{epoch:03d}-real.png", real_img)

            plt.close()
    
    # instantiate train monitor callback
    return TrainMonitor()
