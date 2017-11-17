import numpy as np 
import tensorflow as tf 
import pandas as pd 

X_dim = 75*75
z_dim = 100
h_dim = 128 
mb_size = 64 
def xavier_init(  size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

class GAN():


    def __init__(self , units, layers , filters  , ks  ):

        # return a mo
        
        self.units = units 
        self.layers = layers 
        self.filters = filters 
        self.ks = ks 
        
        #self.init_vars()
        
        
    def xavier_init( self , size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def init_vars(self):

        # var generator
        self.G_w1 = tf.Variable( xavier_init( [z_dim , h_dim] )   )
        self.G_b1 = tf.Variable( tf.zeros( shape=[h_dim] )  )
        
        self.G_w2 = tf.Variable( xavier_init( [h_dim , X_dim] )   )
        self.G_b2 = tf.Variable( tf.zeros( shape=[X_dim] )  )
        
        #var discriminator 

        self.D_w1 = tf.Variable( xavier_init( [X_dim, h_dim ] )  )
        self.D_b1 =  tf.Variable( tf.zeros( shape=[h_dim] )  )

        self.D_w2 = tf.Variable( xavier_init( [h_dim , 1 ] ))
        self.D_b2 = tf.Variable( tf.zeros( shape=[1] ) )
        # placeholder 
        self.z = tf.placeholder(tf.float32, shape=[None, z_dim])

    def generator(self , z):
        
        # recive un vextor aleatorio y genera 
        # nuevo 
        # z  = [ batch_size , 100 ]
        with tf.variable_scope("generator") as scope:
            

            

            g_h1 = tf.nn.relu(  tf.matmul( z , self.G_w1 ) + self.G_b1 )
            g_log_prob = tf.matmul( g_h1 , self.G_w2 ) + self.G_b2

            g_h1 = tf.nn.relu( g_log_prob)
            
            # g_log_prob [ mb_size ,75*75]

            

            
        g_prob = tf.nn.sigmoid( g_log_prob )
        return g_prob

        # g_prob [ 75 * 75 ]
        

    def discriminator( self , x ):
        
        # x [mb_size,75*75] 
        D_h1 = tf.nn.relu( tf.matmul( x , self.D_w1 ) + self.D_b1  )
        # shape [ h_dim]
        out = tf.matmul( D_h1 , self.D_w2 ) + self.D_b2
        return out 
        
        
    def varlist_G(self):

        th = [ 
               self.G_w1 , self.G_b1 , self.G_w2 , self.G_b2
        ]
        return th 
    def varlist_D(self):
        th = [ 
            self.D_w1  , self.D_b1 , self.D_w2 , self.D_b2 
        ]
        
        return th 

    def build(self , X  , global_step ):

        self.init_vars()
        var_list_G = self.varlist_G()
        var_list_D = self.varlist_D()


        G_sample = self.generator(self.z)
        D_real = self.discriminator(X )
        D_fake = self.discriminator( G_sample)

        self.D_loss = tf.reduce_mean( D_real)- tf.reduce_mean(D_fake)
        self.G_loss = -tf.reduce_mean( D_fake )

        self.D_solver = tf.train.RMSPropOptimizer(
            learning_rate = 1e-4  ).minimize( -self.D_loss , var_list = var_list_D  )

        self.G_solver = tf.train.RMSPropOptimizer(
            learning_rate = 1e-4
        ).minimize( self.G_loss , var_list = var_list_G , global_step = global_step )

        
        self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in var_list_D]

        tf.summary.scalar('G_loss' , self.G_loss)
        tf.summary.scalar('D_loss' , self.D_loss ) 
        
        self.build_merge()
        self.G_sample = G_sample
        
        print( self.merged_op ) 
        
    def train(self , start_step ,epochs , sess , tb_dir   ):
        
        self.build_writer( tb_dir , sess )
        
        for it in range( start_step , epochs):
            D_loss_curr = None
            for _ in range(5):

                _ , D_loss_curr , _ = sess.run( [ self.D_solver , self.D_loss , self.clip_D ] ,feed_dict = { self.z : self.sample_z(  mb_size , z_dim) }   )
                
            _ , G_loss_curr = sess.run( [self.G_solver , self.G_loss ]  , feed_dict = { self.z : self.sample_z(  mb_size , z_dim) } ) 
           
            
            if it % 100 == 0 :
            
                summary = sess.run( self.merged_op ,feed_dict = { self.z : self.sample_z(  mb_size , z_dim) }  )
                self.writer.add_summary( summary , it )
                print("Step: {}".format(it) )
                print("Generator Loss:{}".format( G_loss_curr ))
                print("Discriminator Loss:{}".format( D_loss_curr ))
                
        return


    def generate(self, sess ):

        
        z = self.sample_z( mb_size*20 , z_dim )
        samples  = sess.run( self.G_sample , feed_dict = {self.z : z })

        sam_l = []
        for i in range( samples.shape[0] ):

            sam_l.append(  [ samples[i][:] ]  )


        sam_l = np.array( sam_l )
        #df = pd.DataFrame(   sam_l    )
       
        return samples
        
        
    def sample_z(self , m, n):

        return np.random.uniform(-1., 1., size=[m, n])


    def build_merge(self):

        self.merged_op = tf.summary.merge_all()
        
        return 

    def build_writer(self , tb_dir , sess ):

        self.writer = tf.summary.FileWriter(tb_dir , sess.graph)
        return 
