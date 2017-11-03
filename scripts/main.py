import tensorflow as tf
import numpy as np
import model as model

import input_manager

z_dim = 100
mb_size = 64 # batch_size


model_dir = "../model_output/"
tb_dir = "../tensorboard/"
CHECK_INTERVAL  = 50 
repeat = 10 

datam = input_manager.InputManager("../data/tf_ships.pb2" , mb_size , repeat )
gan = model.GAN( 1 , 1 , 1 , 1)

sess = tf.Session()


iterator = datam.iterator()

#global step
global_step = tf.get_variable(
        name="global_step" ,
        shape = []  ,
        dtype = tf.int64 ,
        initializer = tf.zeros_initializer() ,
        trainable = False ,
        collections = [ tf.GraphKeys.GLOBAL_VARIABLES , tf.GraphKeys.GLOBAL_STEP]
)


input_test = iterator.get_next()

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

saver = tf.train.Saver( )
hooks = [
            tf.train.CheckpointSaverHook(
                checkpoint_dir = model_dir ,
                save_steps = CHECK_INTERVAL ,
                saver = saver 
            )
]

#
z = tf.placeholder(tf.float32, shape=[None, z_dim])

gan.build( input_test[0] )

with tf.train.SingularMonitoredSession( hooks = hooks , checkpoint_dir = model_dir , config = config ) as sess:
    
    # kiere luzhar 
    sess.run( iterator.initializer )
    sess.run(global_step)
    for i in range(10):
  
        tt = sess.run( input_test  )
        ## tt es banda  1 
        #print(tt[0].shape)
        
        print("kiere luzhar")
    gan.train(  10 , sess  )


print("good") 
