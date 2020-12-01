import tensorflow as tf
import tensorflow.keras as keras
from tensorflow_core.python.ops.distributions.kullback_leibler import cross_entropy


def D_loss_WGAN(D, gen_img, full_img, gen_out, full_out):
    def gradient_penalty(discriminator, batch_x, fake_image):
        batchsz = batch_x.shape[0]

        # [b, h, w, c]
        t = tf.random.uniform([batchsz, 1, 1])
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

    false_loss = cross_entropy(tf.zeros_like(gen_out), gen_out)
    true_loss = cross_entropy(tf.ones_like(full_out), full_out)
    # l1_loss = tf.reduce_mean(tf.abs(full_img - fake_img))
    l2_loss = tf.reduce_mean(tf.losses.mean_squared_error(gen_img, full_img))
    gp = gradient_penalty(D, full_img, gen_img)
    d_loss = gp * 1. + false_loss + true_loss + l2_loss
    return d_loss


def G_loss_WGAN(gen_out):
    g_loss = cross_entropy(tf.ones_like(gen_out), gen_out)
    return g_loss
