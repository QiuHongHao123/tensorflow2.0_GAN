import tensorflow as tf


def D_loss_WGAN(D, gen_img, full_img, gen_out, full_out):
    def gradient_penalty(discriminator, batch_x, fake_image):
        batchsz = batch_x.shape[0]

        # [b, h, w, c]
        t = tf.random.uniform([batchsz, 1, 1, 1])
        # [b, 1, 1, 1] => [b, h, w, c]
        t = tf.broadcast_to(t, batch_x.shape)

        interplate = t * batch_x + (1 - t) * fake_image

        with tf.GradientTape() as tape:
            tape.watch([interplate])
            d_interplote_logits = discriminator(interplate, training=True)
        grads = tape.gradient(d_interplote_logits, interplate)

        # grads:[b, h, w, c] => [b, -1]
        grads = tf.reshape(grads, [grads.shape[0], -1])
        gp = tf.norm(grads, axis=1)  # [b]
        gp = tf.reduce_mean((gp - 1) ** 2)

        return gp

    w_loss = -tf.reduce_mean(full_out) + tf.reduce_mean(gen_out)

    gp = gradient_penalty(D, full_img, gen_img)
    d_loss = gp * 1. + w_loss
    return d_loss

def G_loss_WGAN(gen_out):
    g_loss = -tf.reduce_mean(gen_out)
    return g_loss
