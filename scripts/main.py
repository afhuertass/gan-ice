import tensorflow as tf
import numpy as np
import model as model

import input_manager

z_dim = 100
mb_size = 64 # batch_size
total_steps = 3500

model_dir = "../model_output/ships/band2"
tb_dir = "../model_output/ships/band2/tensorboard"
CHECK_INTERVAL  = 1000
repeat = 1


tf.reset_default_graph()
datam = input_manager.InputManager("../data/tf_ships_norm.pb2" , mb_size , repeat*mb_size )
gan = model.GAN( 1 , 1 , 1 , 1)


iterator = datam.iterator()

input_test = iterator.get_next()


#global step
global_step = tf.get_variable(
        name="global_step" ,
        shape = []  ,
        dtype = tf.int64 ,
        initializer = tf.zeros_initializer() ,
        trainable = False ,
        collections = [ tf.GraphKeys.GLOBAL_VARIABLES , tf.GraphKeys.GLOBAL_STEP]
)


gan.build( input_test[1] , global_step  )

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




with tf.train.SingularMonitoredSession( hooks = hooks , checkpoint_dir = model_dir , config = config ) as sess:
    
    # kiere luzhar 
    sess.run( iterator.initializer )
    start_step = sess.run(global_step)
    
    print(  "start step:{}".format( start_step ))
    try:    
        gan.train( start_step ,  total_steps , sess  , tb_dir    )
    except tf.errors.OutOfRangeError:
        print("Dataset agotado")

    samples = gan.generate(  sess )


    np.save( "new_ships_band2" , samples)
    print(samples.shape)
        

print("Finished") 
