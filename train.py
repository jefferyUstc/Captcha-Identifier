#! /usr/bin/python
# _*_ coding: utf-8 _*_
__author__ = 'Jeffery'
__date__ = '2018/11/25 19:04'

import tensorflow as tf
from load_data import get_batch_data
import mymodel
from constant import width, height, char_num, characters, classes

if __name__ == '__main__':
    generate_batch = get_batch_data(batch_size=50, epoch=10, resize_shape=None)

    x = tf.placeholder(tf.float32, [None, height, width, 1])
    y_ = tf.placeholder(tf.float32, [None, char_num * classes])

    # my GPU cannot work recently, so use your gpu here at least and ...
    y_conv, logits = mymodel.captcha_model(x)

    predict = tf.reshape(y_conv, [-1, char_num, classes])
    real = tf.reshape(y_, [-1, char_num, classes])
    predict_logits = tf.reshape(logits, [-1, char_num, classes])
    # loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=real,
            logits=predict_logits,
            dim=2))
    # use your GPU
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
    # acc
    correct_prediction = tf.cast(tf.equal(tf.argmax(predict, 2),
                                          tf.argmax(real, 2)),
                                 tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('images', x, 3)
    merged = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("log/", sess.graph)
        ckpt = tf.train.get_checkpoint_state('./model_data')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored.")
        else:
            print('Totally new training start')
        step = 1
        total_loss = 0
        try:
            while True:
                batch_x, batch_y = next(generate_batch)
                _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_x, y_: batch_y})
                total_loss += loss
                if step % 100 == 0:
                        print('step:%d,loss:%s' % (step, total_loss/100))
                        total_loss = 0

                # validation
                # it's controversial?anyway, u can build a validation dataset
                # i'm lazy guy
                if step % 500 == 0:
                    batch_x_test, batch_y_test = next(generate_batch)
                    acc, merged_summary = sess.run([accuracy, merged], feed_dict={x: batch_x_test, y_: batch_y_test})
                    writer.add_summary(merged_summary, global_step=step)
                    print('####step:%d,accuracy:%f' % (step, acc))
                    if acc > 0.95:
                        saver.save(sess, "./model_data/capcha_model.ckpt")
                        break
                step += 1
        except Exception as e:
            print(e)
            saver.save(sess, "./model_data/capcha_model.ckpt")
