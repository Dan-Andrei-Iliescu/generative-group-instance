###########################
# Author: Omer Nivron
###########################
import tensorflow as tf
from tensorflow.keras import regularizers
from model import dot_prod_attention


class Decoder(tf.keras.Model):
    def __init__(self, e, l1=256, l2=128, l3=32, rate=0.1, num_heads=1, input_vocab_size=2000):
        super(Decoder, self).__init__()
        self.e = e
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, e, name='embedding')
        self.embedding_y = tf.keras.layers.Embedding(input_vocab_size, e, name='embedding_y')

        self.mha2 = dot_prod_attention.MultiHeadAttention2D(e, num_heads)
        self.mha = dot_prod_attention.MultiHeadAttention2D(e, num_heads)
        self.A1 = tf.keras.layers.Dense(l1, name='A1')
        self.A2 = tf.keras.layers.Dense(l1, name='A2')
        self.A3 = tf.keras.layers.Dense(l2, name='A3')
        self.A4 = tf.keras.layers.Dense(l3, name='A4')
        self.A5 = tf.keras.layers.Dense(2, name='A5')


    def call(self, x, y, training, x_mask, infer=False, ix=None, iy=None, n=0, x0=None, y0=None, x1=None, y1=None):
        """

        :param x: (np.array of int) of indices associated with a range of continuous x-vals
        :param x_2: (np.array) of zeros and ones, identifying from which sequence member (out of a pair) each
        data point came from.
        :param y: (np.array float) target variables.
        :param training: (bool) must be TRUE when training and False otherwise
        :param x_mask: (np.array) of zeros where a prediction is wanted and 1 otherwise
        :return:
        2D (tf.tensor) with first dimension being the mean and second the log sigma
        """
        # x_2 = x_2[:, :, tf.newaxis]  # (batch_size, seq_len, 1)
        # x = tf.concat((self.embedding(x), x_2), axis=-1)  # (batch_size, seq_len + 1, e)
        # tf.print(x[0, 1, :])
        # y = self.embedding_y(y)
        y = y[:, :, tf.newaxis]
        print('y: ', y)
        y_attn, _, _ = self.mha(y, x, x, x_mask, infer=infer, x=ix, y=iy, n=n, x0=x0, y0=y0, x1=x1, y1=y1)  # (batch_size, seq_len, e)
        attn_output = self.dropout1(y_attn, training=training)
        current_position = x[:, 1:, :]  # (batch_size, seq_len, e)
        out1 = self.layernorm1(attn_output + current_position)
        print('out1: ', out1)
        y_attn2, _, _ = self.mha2(out1, x, x, x_mask)
        attn2_output = self.dropout2(y_attn2, training=training)
        out2 = self.layernorm2(attn2_output + out1)
        print('out2: ', out2)

        out2 = tf.nn.leaky_relu(out2)
        L = self.A1(out2)  # (batch_size, seq_len, l1)
        L = tf.nn.leaky_relu(L)
        L = self.A2(L)  # (batch_size, seq_len, l2)
        L = self.dropout3(L, training=training)
        L = self.layernorm3(L + out2)
        L = tf.nn.leaky_relu(self.A3(L))  # (batch_size, seq_len, l3)
        L = tf.nn.leaky_relu(self.A4(L))  # (batch_size, seq_len, 2)
        L = self.A5(L)  # (batch_size, seq_len, 2)
        return tf.squeeze(L)
