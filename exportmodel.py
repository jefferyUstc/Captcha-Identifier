#! /usr/bin/python
# _*_ coding: utf-8 _*_
__author__ = 'Jeffery'
__date__ = '2018/12/29 23:20'
import tensorflow as tf
from mymodel import captcha_model as model
import os
from constant import char_num, classes

def export_model(checkpoint_path,
                 export_model_dir,
                 model_version
                 ):
    """
    :param checkpoint_path: type string, original model path(a dir)
    :param export_model_dir: type string, save dir for exported model
    :param model_version: type int best
    :return:no return
    """
    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, 100, 120, 1], name='input_images')
        output_result, _ = model(input_images, keep_prob=1.0, trainable=False)
        output_result = tf.argmax(tf.reshape(output_result, [-1, char_num, classes]), 2)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
            model_path = os.path.join(checkpoint_path,
                                      os.path.basename(ckpt_state.model_checkpoint_path))
            saver.restore(sess, model_path)
            print('step1 => Model Restored successfully from {}'.format(model_path))
            # set-up a builder
            export_path_base = export_model_dir
            export_path = os.path.join(
                tf.compat.as_bytes(export_path_base),
                tf.compat.as_bytes(str(model_version)))
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)
            print('step2 => Export path(%s) ready to export trained model' % export_path)
            tensor_info_input = tf.saved_model.utils.build_tensor_info(input_images)
            tensor_info_output = tf.saved_model.utils.build_tensor_info(output_result)
            # prediction_signature
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'images': tensor_info_input},
                    outputs={'result': tensor_info_output},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
            # # classification_signature
            # classification_signature = (
            #     tf.saved_model.signature_def_utils.build_signature_def(
            #         inputs={
            #             tf.saved_model.signature_constants.CLASSIFY_INPUTS: '1'
            #         },
            #         outputs={
            #             tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: '2',
            #             tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES: '3'
            #         },
            #         method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME
            #     ))
            # # regression_signature
            # regression_signature = (
            #     tf.saved_model.signature_def_utils.build_signature_def(
            #         inputs={
            #             tf.saved_model.signature_constants.REGRESS_INPUTS: '1'
            #         },
            #         outputs={
            #             tf.saved_model.signature_constants.REGRESS_OUTPUTS: '2',
            #         },
            #         method_name=tf.saved_model.signature_constants.REGRESS_METHOD_NAME
            #     ))

            print('step3 => prediction_signature created successfully')
            builder.add_meta_graph_and_variables(
                # tags:SERVING,TRAINING,EVAL,GPU,TPU
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'prediction_signature': prediction_signature,
                })
            print('step4 => builder successfully add meta graph and variables\nNext is to export model...')
            builder.save(as_text=True)
            print('Done exporting!')


if __name__ == '__main__':
    export_model(
        './model_data',
        './export_model',
        1
    )
