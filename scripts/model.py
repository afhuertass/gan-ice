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
        self.dropout = 0.33
        #self.init_vars()
        
        self.indxdi = 0
        self.indxge = 0 
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
        
        with tf.variable_scope('generator'  ) as scope :
            self.indxge = self.indxge + 1 
            inputs = tf.reshape( z , [-1, z_dim])
            units = 5*5*64
            gen_fc = tf.layers.dense(
                inputs = inputs ,
                units = units ,
                name = "kapa1"
            )
            
            gen_fc = self.batchnormalize( gen_fc ) 
            gen_fc = tf.nn.relu( gen_fc ) 
            gen_fc = tf.reshape( gen_fc , [-1 , 5 , 5 , 64  ])
            
            deconv1 = tf.layers.conv2d_transpose(
                inputs = gen_fc ,
                filters = 32 ,
                kernel_size = [3,3] , 
                strides = [ 3, 3] , 
                activation = None ,
                padding = 'same' , 
                name = "kapa2"
            )

        
            deconv1 = tf.nn.relu( deconv1 ) 
            deconv1 = tf.layers.dropout(deconv1 , self.dropout )

            print("deconv shape ") # 15 , 15 , 32 
            print( deconv1.shape) 

            deconv2 = tf.layers.conv2d_transpose(
                inputs = deconv1 ,
                strides = [ 5 , 5] ,
                filters = 2 ,
                kernel_size = [5,5] , 
                padding = 'same' ,

                name = "kapa3"
            )
            deconv2 = self.batchnormalize( deconv2 )
            deconv2 = tf.nn.relu( deconv2 )
            
            deconv2 = tf.layers.dropout( deconv2 , self.dropout )
         
            gen_output = tf.tanh( deconv2 , 'gen_out' )
            print("shape output generator")
            print(gen_output.shape )
            
            return gen_output

    

    def discriminator( self , inputs , scope_name   ):
        
        # x [75*75]
        inputs = tf.reshape( inputs , [-1 , 75,75 , 2  ])
        with tf.variable_scope( scope_name ) as scope :
        

            # inputs [-1 , 75 , 75 , 2 ]
            # conv1 [-1 , 15 , 15 , 8 ]
            conv1 = tf.layers.conv2d(
                inputs = inputs ,
                filters = 8 ,
                kernel_size = [ 5 , 5 ],
                strides = [ 5, 5] ,
                padding = 'same' ,
                name = "capa1"
            )
           
            conv1 = self.batchnormalize( conv1 )
            conv1 = tf.layers.dropout( conv1 , self.dropout)
            conv1 = self.lrelu( conv1 )
            print(conv1.shape)
            # conv2 [-1 , 5 , 5 , 32 ]
            conv2 = tf.layers.conv2d(
                inputs = conv1 ,
                filters = 32 ,
                strides = [3,3 ],
                kernel_size = [3,3] , 
                padding = 'same' ,
                name = "capa2"
            )
            
            
            conv2 = self.batchnormalize( conv2 )
            conv2 = tf.layers.dropout( conv2 , self.dropout )
            conv2 = self.lrelu( conv2 ) 

            print(conv2.shape)
            fc_input = tf.reshape( conv2  , (-1 , 5*5*32) )
            logits = tf.layers.dense(
                inputs = fc_input ,
                units = 1 ,
                activation = None , 
                name = "capa3"
            )
            print("shape output discriminator")
            out = self.lrelu(logits)
            print(out.shape)
            #return out , logits 
            return out , logits 
        

    def build_inputs_dis( self, band1 , band2 ):

         band1 = tf.reshape( band1  , [-1 ,  75 , 75 ] )
         band2 =  tf.reshape( band2  , [-1 ,  75 , 75 ] )
            
         inputs = tf.stack( [band1 , band2] , axis = -1 )
         inputs = tf.reshape( inputs , [-1 , 75,75 , 2  ])

         return inputs 

    def lrelu(self ,  X , leak = 0.2 ):

        f1 = 0.5*(1 + leak )
        f2 = 0.5*(1-leak) 
        return f1*X + f2*tf.abs(X)



    def build(self , X  , global_step ):

        self.init_vars()
        inputs = self.build_inputs_dis(X[0] , X[1])
        
        #var_list_G = self.varlist_G()
        #var_list_D = self.varlist_D()
        
        G_sample = self.generator(self.z)
        D_real ,D_logits = self.discriminator( inputs , "real_scope"  )
        D_fake , D_logits_fake = self.discriminator( G_sample  , "fake_scope")


        # real images are (1) , fake ones are (0) 
        D_loss_real = self.cross_entropy_loss( D_logits , tf.ones_like(D_real) , name ="dloss1" )
        #  fake ones are (0) 
        D_loss_fake = self.cross_entropy_loss(D_logits_fake , tf.zeros_like(D_fake) , name = "dloss2" )

        self.D_loss = D_loss_real + D_loss_fake
        # the generator tries to produce outputs with label 1 
        self.G_loss = self.cross_entropy_loss( D_fake , tf.ones_like(D_fake) , name = "gloss1" )
        
        #self.D_loss = tf.reduce_mean( D_real)- tf.reduce_mean(D_fake)
        #self.G_loss = -tf.reduce_mean( D_fake )


        
        var_list_dis_real  = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , scope = "real_scope" )
        var_list_dis_fake =  tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , scope = "fake_scope" )

        var_list_gen = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        
        self.D_solver = tf.train.AdamOptimizer(
            learning_rate = 2e-4  ).minimize( self.D_loss , var_list = [var_list_dis_real , var_list_dis_fake  ]  )

        self.G_solver = tf.train.AdamOptimizer(
            learning_rate = 2e-4
        ).minimize( self.G_loss , var_list = var_list_gen , global_step = global_step )

        
        #self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in var_list_D]
        #self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in var_list_dis_real   ]
        #self.clip_D2 = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in var_list_dis_fake   ]
        
        tf.summary.scalar('G_loss' , self.G_loss)
        tf.summary.scalar('D_loss' , self.D_loss ) 
        
        self.build_merge()
        self.G_sample = G_sample
      
        
    def train(self , start_step ,epochs , sess , tb_dir   ):
        
        self.build_writer( tb_dir , sess )
        
        for it in range( start_step , epochs):
            D_loss_curr = None
            for _ in range(5):

                _ , D_loss_curr  = sess.run( [ self.D_solver , self.D_loss   ] ,feed_dict = { self.z : self.sample_z(  mb_size , z_dim) }   )
                
            _ , G_loss_curr = sess.run( [self.G_solver , self.G_loss ]  , feed_dict = { self.z : self.sample_z(  mb_size , z_dim) } ) 
           
            #print("Step: {}".format(it) )
            #print("Generator Loss:{}".format( G_loss_curr ))
            #print("Discriminator Loss:{}".format( D_loss_curr ))
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


    def cross_entropy_loss(self , logits , labels , name="xentropy" ):

        xentropy = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))

        return xentropy

    def batchnormalize(self, X, eps=1e-8, g=None, b=None):
        if X.get_shape().ndims == 4:
            mean = tf.reduce_mean(X, [0,1,2])
            std = tf.reduce_mean( tf.square(X-mean), [0,1,2] )
            X = (X-mean) / tf.sqrt(std+eps)
            
            if g is not None and b is not None:
                g = tf.reshape(g, [1,1,1,-1])
                b = tf.reshape(b, [1,1,1,-1])
                X = X*g + b

        elif X.get_shape().ndims == 2:
            mean = tf.reduce_mean(X, 0)
            std = tf.reduce_mean(tf.square(X-mean), 0)
            X = (X-mean) / tf.sqrt(std+eps)

            if g is not None and b is not None:
                g = tf.reshape(g, [1,-1])
                b = tf.reshape(b, [1,-1])
                X = X*g + b

        else:
            raise NotImplementedError
            
        return X
