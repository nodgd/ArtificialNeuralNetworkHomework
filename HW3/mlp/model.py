# -*- coding: utf-8 -*-

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

class Model:
    def __init__(self,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 28 * 28])
        self.y_ = tf.placeholder(tf.int32, [None])

        self.loss, self.pred, self.acc = self.forward(is_train=True)
        self.loss_val, self.pred_val, self.acc_val = self.forward(is_train=False, reuse=True)
        
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)  # Learning rate decay

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step, var_list=self.params)  # Use Adam Optimizer

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2, max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
                                    
    def forward(self, is_train, reuse=None):
    
        with tf.variable_scope("model", reuse=reuse):
            tensor = self.x_
            tensor = linear_layer(tensor, 128, 0.01)
            tensor = batch_normalization_layer(tensor, is_train)
            tensor = relu_layer(tensor)
            tensor = dropout_layer(tensor, FLAGS.keep_prob, is_train)
            tensor = linear_layer(tensor, 10, 0.08)
            logits = tensor

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        pred = tf.argmax(logits, 1)  # Calculate the prediction result
        correct_pred = tf.equal(tf.cast(pred, tf.int32), self.y_)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch
        
        return loss, pred, acc

def batch_normalization_layer(input, is_train=True):
    return tf.layers.batch_normalization(input, training=is_train)

def dropout_layer(input, keep_prob, is_train=True):
    return tf.layers.dropout(input, rate=keep_prob, training=is_train)

def relu_layer(input):
    return tf.nn.relu(input)

def linear_layer(input, channel_out, weight_std):
    return tf.layers.dense(input, channel_out, kernel_initializer=tf.truncated_normal_initializer(stddev=weight_std))
