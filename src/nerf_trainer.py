import tensorflow as tf

class NeRFTrainer(tf.keras.Model):
    def __init__(self, coarse_model, fine_model, l_xyz, l_dir, encoder_fn, render_img_depth, sample_pdf, n_fine_samples):
        super().__init__()

        self.coarse_model = coarse_model
        self.fine_model = fine_model
        self.l_xyz = l_xyz
        self.l_dir = l_dir
        self.encoder_fn = encoder_fn
        self.render_img_depth = render_img_depth
        self.sample_pdf = sample_pdf
        self.n_fine_samples = n_fine_samples
    
    def compile(self, optimizer_coarse, optimizer_fine, loss_fn):
        super().compile()

        self.optimizer_coarse = optimizer_coarse
        self.optimizer_fine = optimizer_fine
        self.loss_fn = loss_fn

        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.psnr_metric = tf.keras.metrics.Mean(name="psnr")

    def train_step(self, inputs):
        # unpack inputs
        (elements, images) = inputs
        (ray_origins_coarse, ray_dirs_coarse, t_vals_coarse) = elements

        # generate coarse rays
        rays_coarse = ray_origins_coarse[..., None, :] + (ray_dirs_coarse[..., None, :] * t_vals_coarse[..., None])

        # encode the rays and dirs
        rays_coarse = self.encoder_fn(rays_coarse, self.l_xyz)
        dir_coarse_shape = tf.shape(rays_coarse[..., :3])
        dirs_coarse = tf.broadcast_to(ray_dirs_coarse[..., None, :], shape=dir_coarse_shape)
        dirs_coarse = self.encoder_fn(dirs_coarse, self.l_dir)

        # keep track of gradients
        with tf.GradientTape() as coarse_tape:
            #compute predictions from coarse model
            (rgb_coarse, sigma_coarse) = self.coarse_model([rays_coarse, dirs_coarse])

            # render the image from the predictions
            render_coarse = self.render_img_depth(rgb=rgb_coarse, sigma=sigma_coarse, t_vals=t_vals_coarse)
            (image_coarse, _, weights_coarse) = render_coarse

            # compute photometric loss
            loss_coarse = self.loss_fn(images, image_coarse)

            # compute the middle values of t vals
            t_vals_coarse_mid = 0.5 * (t_vals_coarse[..., 1:] + t_vals_coarse[..., :-1])

            # apply hierarchical sampling and get the t values for the fine model
            t_vals_fine = self.sample_pdf(t_vals_mid=t_vals_coarse_mid, weights=weights_coarse, n_fine=self.n_fine_samples)
            t_vals_fine = tf.sort(tf.concat([t_vals_coarse, t_vals_fine], axis=-1), axis=-1)

            # build the fine rays and positionally encode them
            rays_fine = ray_origins_coarse[..., None, :] + (ray_dirs_coarse[..., None, :] * t_vals_fine[..., None])
            rays_fine = self.encoder_fn(rays_fine, self.l_xyz)

            # build the fine directions and positionally encode them
            dirs_fine_shape = tf.shape(rays_fine[..., :3])
            dirs_fine = tf.broadcast_to(ray_dirs_coarse[..., None, :], shape=dirs_fine_shape)
            dirs_fine = self.encoder_fn(dirs_fine, self.l_dir)

            # keep track of gradients
            with tf.GradientTape() as fine_tape:
                # compute predictions from fine model
                rgb_fine, sigma_fine = self.fine_model([rays_fine, dirs_fine])

                # render the image from the predictions
                render_fine = self.render_img_depth(rgb=rgb_fine, sigma=sigma_fine, t_vals = t_vals_fine)
                (image_fine, _, _) = render_fine
                
                # compute photometric loss
                loss_fine = self.loss_fn(images, image_fine)
            
                # get trainable vars from coarse model and apply back propagation
                tv_coarse = self.coarse_model.trainable_variables
                grads_coarse = coarse_tape.gradient(loss_coarse, tv_coarse)
                self.optimizer_coarse.apply_gradients(zip(grads_coarse, tv_coarse))

                # get trainable vars from fine model and apply back propagation
                tv_fine = self.fine_model.trainable_variables
                grads_fine = fine_tape.gradient(loss_fine, tv_fine)
                self.optimizer_fine.apply_gradients(zip(grads_fine, tv_fine))
                psnr = tf.image.psnr(images, image_fine, max_val=1.0)

                # compute loss and psnr metrics
                self.loss_metric.update_state(loss_fine)
                self.psnr_metric.update_state(psnr)

                # return loss and psnr metrics
                return {
                    "loss": self.loss_metric.result(),
                    "psnr": self.psnr_metric.result(),
                }
    
    def test_step(self, inputs):
        # get images and rays
        (elements, images) = inputs
        (ray_origins_coarse, ray_dirs_coarse, t_vals_coarse) = elements

        # generate the coarse rays
        rays_coarse = ray_origins_coarse[..., None, :] + (ray_dirs_coarse[..., None, :] * t_vals_coarse[..., None])

        # positional encode rays and dirs
        rays_coarse = self.encoder_fn(rays_coarse, self.l_xyz)
        dirs_coarse_shape = tf.shape(rays_coarse[..., :3])
        dirs_coarse = tf.broadcast_to(ray_dirs_coarse[..., None, :], shape=dirs_coarse_shape)
        dirs_coarse = self.encoder_fn(dirs_coarse, self.l_dir)

        # compute the predictions from the coarse model
        (rgb_coarse, sigma_coarse) = self.coarse_model([rays_coarse, dirs_coarse])

        # render the image from the predictions
        render_coarse = self.render_img_depth(rgb=rgb_coarse, sigma=sigma_coarse, t_vals=t_vals_coarse)
        (_, _, weights_coarse) = render_coarse
        
        # compute middle values of t_vals
        t_vals_coarse_mid = 0.5 * (t_vals_coarse[..., 1:] + t_vals_coarse[..., :-1])
 
        
        # apply hierarchical sampling and get the t_vals for the fine model
        t_vals_fine = self.sample_pdf(t_vals_mid=t_vals_coarse_mid, weights=weights_coarse, n_fine=self.n_fine_samples)
        t_vals_fine = tf.sort(tf.concat([t_vals_coarse, t_vals_fine], axis=-1), axis=-1)

        # build the fine rays and positionally encode them
        rays_fine = ray_origins_coarse[..., None, :] + (ray_dirs_coarse[..., None, :] * t_vals_fine[..., None])
        rays_fine = self.encoder_fn(rays_fine, self.l_xyz)

        # build the fine directions and positionally encode them
        dirs_fine_shape = tf.shape(rays_fine[..., :3])
        dirs_fine = tf.broadcast_to(ray_dirs_coarse[..., None, :], shape=dirs_fine_shape)

        # compute predictions from the fine model
        rgb_fine, sigma_fine = self.fine_model([rays_fine, dirs_fine])

        # render the image from the predictions
        render_fine = self.render_img_depth(rgb=rgb_fine, sigma=sigma_fine, t_vals=t_vals_fine)
        (image_fine, _, _) = render_fine

        # compute the photometric loss and psnr
        loss_fine = self.loss_fn(images, image_fine)
        psnr = tf.image.psnr(images, image_fine, max_val=1.0)

        # compute the loss and psnr metrics
        self.loss_metric.update_state(loss_fine)
        self.psnr_metric.update_state(psnr)

        # return loss and psnr metrics
        return {
            "loss": self.loss_tracker.result(),
            "psnr": self.psnr_metric.result(),
        }
    
    @property
    def metrics(self):
        # return the loss and psnr metrics
        return [self.loss_metric, self.psnr_metric]

