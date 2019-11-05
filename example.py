import numpy as np
import ot
import matplotlib.pylab as plt

import tensorflow as tf
from ot_tf import dmat as dmat_tf, sink as sink_tf


def main():
    na = 100
    nb = 150
    reg = 0.5

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    mu_t = np.array([4, 4])
    cov_t = np.array([[1, -.8], [-.8, 1]])

    x_tf = tf.placeholder(dtype=tf.float32, shape=[na, 2])
    y_tf = tf.placeholder(dtype=tf.float32, shape=[nb, 2])
    M_tf = dmat_tf(x_tf, y_tf)
    tf_sinkhorn_loss = sink_tf(M_tf, (na, nb), reg)

    print("I can compute the gradient for a", tf.gradients(tf_sinkhorn_loss, x_tf))
    print("I can compute the gradient for b", tf.gradients(tf_sinkhorn_loss, y_tf))

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    xs = ot.datasets.make_2D_samples_gauss(na, mu_s, cov_s)
    xt = ot.datasets.make_2D_samples_gauss(nb, mu_t, cov_t)

    # Visualization
    plt.figure(1)
    plt.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
    plt.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
    plt.legend(loc=0)
    plt.title('Source and target distributions')
    plt.show()

    # TF - sinkhorn
    tf_sinkhorn_loss_val = sess.run(tf_sinkhorn_loss, feed_dict={x_tf: xs, y_tf: xt})
    print(' tf_sinkhorn_loss', tf_sinkhorn_loss_val)

    # POT - sinkhorn
    M = ot.dist(xs.copy(), xt.copy(), metric='euclidean')
    a = np.ones((na,)) / na
    b = np.ones((nb,)) / nb  # uniform distribution on samples
    pot_sinkhorn_loss = ot.sinkhorn2(a, b, M, reg)[0]
    print('pot_sinkhorn_loss', pot_sinkhorn_loss)


if __name__ == '__main__':
    main()
