# -*- coding: utf-8 -*-
"""
Created on Tue May 29 17:20:30 2018

@author: likkhian
"""

import numpy as np
import tensorflow as tf
import os

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features,labels,mode):
    '''model function for cnn'''
    #input layer
    input_layer=tf.reshape(features['x'],[-1,84,168,1])
    
    #convolutional layer 1, including namescopes. Examine conv2d vs Conv2d!
    #Conv2d is a class. conv2d is a function that uses the Conv2d class.
    #to get more info, look up programmers_guide/low_level_intro and layer Function shortcuts
    #Use the class to dig into layer detail.
    with tf.name_scope('lik_conv1'):
        conv1=tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5,5],
            padding='same',
            activation=tf.nn.relu,
            name='conv1')
        conv1_kernel=tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1/kernel')[0]
        conv1_kernel_transposed=tf.transpose(conv1_kernel,[3,0,1,2])
        conv1_bias = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1/bias')[0]
        convimg1=tf.reshape(conv1,[-1,84,168,32,1])
        convimg2=tf.transpose(convimg1,[0,3,1,2,4])
        
    
    #pooling layer 1
    pool1=tf.layers.max_pooling2d(inputs=conv1,pool_size=[3,6],strides=[3,6])
    
    #convolutional layer 2, and pooling layer 2
    with tf.name_scope('lik_conv2'):
        conv2=tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5,5],
            padding='same',
            activation=tf.nn.relu)
    pool2=tf.layers.max_pooling2d(inputs=conv2,pool_size=[4,4],strides=4)
    
    #dense layer
    pool2_flat=tf.reshape(pool2,[-1,7*7*64])
    dense=tf.layers.dense(inputs=pool2_flat,units=1024,activation=tf.nn.relu)
    #extract weights and bias for tensorboard histogram
    weights = tf.get_default_graph().get_tensor_by_name(os.path.split(dense.name)[0] + '/kernel:0')
    bias = tf.get_default_graph().get_tensor_by_name(os.path.split(dense.name)[0] + '/bias:0')
    dropout=tf.layers.dropout(inputs=dense,rate=0.4,
                              training=mode==tf.estimator.ModeKeys.TRAIN)
    
    #logits layer
    logits=tf.layers.dense(inputs=dropout,units=12)
    
    predictions={
        #generate predictions (for PREDICT and EVAL mode)
        'classes':tf.argmax(input=logits,axis=1),
        #Add softmax_tensor to the graph. used for predict and logging hook
        'probabilities':tf.nn.softmax(logits,name='softmax_tensor')}
        
    if(mode==tf.estimator.ModeKeys.PREDICT):
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
        
    #calculate loss
    loss=tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
    
    #save loss as a scalar
    tf.summary.scalar('lik_loss',loss)
    tf.summary.image('lik_input',input_layer,4)
    tf.summary.image('conv1_filter',conv1_kernel_transposed,32)
    tf.summary.histogram('conv1_bias',conv1_bias)
    tf.summary.histogram('lik_denasa_wts',weights)
    tf.summary.histogram('lik_dense_bias',bias)
    tf.summary.image('lik_convimg',convimg2[0,:,:,:],32)

    #add evaluation metrics. Moved it here so accuracy will be in training too
    eval_metric_ops={
        'accuracy':tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes'])}
    tf.summary.scalar('lik_acc',eval_metric_ops['accuracy'][1])
    
    #print confusion matrix images
    confused=tf.confusion_matrix(labels=labels,predictions=predictions['classes'],dtype=tf.float16)
    confused1=tf.reshape(confused,[1,10,10,1])
    tf.summary.image('confusion_mat',confused1) #cols are predictions, rows are labels
    
    #print misclassified images
    mislabeled=tf.not_equal(tf.cast(predictions['classes'],tf.int32),labels)
    wrong_input=tf.boolean_mask(predictions['classes'],mislabeled)
    actual_label=tf.boolean_mask(labels,mislabeled)
    mislabeled2=tf.Print(mislabeled,[wrong_input,actual_label],'printing mislabeled')
    mislabeled_images = tf.boolean_mask(input_layer, mislabeled2)
    tf.summary.image('mislabled',mislabeled_images,4)


    #configure training op
    if(mode==tf.estimator.ModeKeys.TRAIN):
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op=optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
    
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    #load training and eval data
    trmm=np.load('./trmm_data.npy')
    label=np.load('./trmm_label.npy')
    
    train_data=mnist.train.images
    train_labels=np.asarray(mnist.train.labels,dtype=np.int32)
    eval_data=mnist.train.images
    eval_labels=np.asarray(mnist.train.labels,dtype=np.int32)
    #create the estimator
    mnist_classifier=tf.estimator.Estimator(model_fn=cnn_model_fn,
                                            model_dir='./trmm_convnet')
    #set up logging
    tensors_to_log={'probabilities':'softmax_tensor'}
    logging_hook=tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                            every_n_iter=100)
    #training time
    train_input_fn=tf.estimator.inputs.numpy_input_fn(
        x={'x':train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,steps=20000,hooks=[logging_hook])
    eval_input_fn=tf.estimator.inputs.numpy_input_fn(
        x={'x':eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results=mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__=='__main__':
    tf.app.run()