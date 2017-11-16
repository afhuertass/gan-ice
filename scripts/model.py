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

    def generator_mlp(self ,z ):

        with tf.variable_scope("generator") as scope:
            # inputs [batch , 100 ]
            inputs = tf.reshape(z ,  [mb_size , z_dim ])
            units =  75*75 
            lin1 = self.linear( inputs , units , "g_lin1" )
            lin1 = tf.nn.relu(lin1)
            print(lin1.shape)
            #lin1 [b , 75*75 ]
            units2 = 75*75*2
            
            lin2 = self.linear(lin1 , units2 , "g_lin2" )
            lin2 = tf.nn.relu( lin2)
            print(lin2.shape)
            
            units3 = 75*75*2

            lin3 = self.linear( lin2 , units3 , "g_lin3")
            lin3 = tf.nn.tanh( lin2 )
            print(lin3.shape)
            lin3 = tf.reshape( lin3 , [mb_size , 75,75,2] )
            return tf.nn.tanh( lin3 ) 
            
        
    def discriminator_mlp(self , x , reuse ) :

        with tf.variable_scope("discriminator") as scope:
            
            if reuse:
                scope.reuse_variables() 

            # x [mb , 75, 75 *2 ]
            inputs = tf.reshape(x ,[mb_size , 75*75*2 ] )
            units1 = 75*75 
            lin1 = self.linear( inputs , units1 , "d_lin1")
            lin1 = self.lrelu( lin1)

            units2 = 15*15 
            lin2 = self.linear( lin1,  units2 , "d_lin2")
            lin2 = self.lrelu( lin2)

            units3 = 5*5
            lin3 = self.linear( lin2 , units3 , "d_lin3")
            lin3 = self.lrelu( lin3)

            lin4 = self.linear( lin3 , 1 , "d_lin4")

            return lin4 , tf.nn.sigmoid( lin4 ) 

        
    def generator(self , z):
        
        # recive un vextor aleatorio y genera 
        # nuevo 
        # z  = [ batch_size , 100 ]
        
        with tf.variable_scope('generator'  ) as scope :
            
            inputs = tf.reshape( z , [ mb_size, z_dim])
            
            units = 5*5*64
            lin1 = self.linear( inputs , units , "g_lin1" )
            lin1 = self.batch_norm2( lin1 , "g_normlin0" )
            gen_fc = tf.nn.relu( lin1 , name = "g_relu1" )
            
            gen_fc = tf.reshape( gen_fc , [ mb_size , 5 , 5 , 64  ])
            
            
            deconv1 = tf.layers.conv2d_transpose(
                inputs = gen_fc ,
                filters = 32 ,
                kernel_size = [3,3] , 
                strides = [ 3, 3] , 
                padding = 'same' , 
                name = "g_deconv1" ,
               
            )
            
            deconv1 = self.batch_norm2( deconv1 , "g_norm2" ) 
            deconv1 = tf.nn.relu( deconv1 ) 
            print("shape g_deconv1")
            print( deconv1.shape)
            
            deconv2 = tf.layers.conv2d_transpose(
                inputs = deconv1 ,
                strides = [ 5 , 5] ,
                filters = 32 ,
                kernel_size = [5,5] , 
                padding = 'same' ,
                
                name = "g_deconv2" , 
            )
            deconv2 = self.batch_norm2( deconv2 , "g_norm3" )
            deconv2 = tf.nn.relu( deconv2 )
            
            print("shape g_deconv2")
            print( deconv2.shape)
            
            deconv3 = tf.layers.conv2d_transpose(
                inputs = deconv2 ,
                filters = 2 ,
                strides = [1,1] ,
                kernel_size = [1,1] ,
                name = "g_deconv3" ,
                activation = tf.nn.tanh
            )
            # no wgan 
            #gen_output = tf.tanh( deconv3 , name = 'g_tanh'  )
            
            print("shape output generator")
            
            print(deconv3.shape )
            
            return deconv3 

    

    def discriminator( self , inputs , reuse   ):

        with tf.variable_scope("discriminator") as scope:

            if reuse:
                scope.reuse_variables() 
        
            # x [75*75]
            stddev = 0.02
            inputs = tf.reshape( inputs , [mb_size , 75,75 , 2  ])
            # inputs [-1 , 75 , 75 , 2 ]
            # conv1 [-1 , 15 , 15 , 8 ]
            conv1 = tf.layers.conv2d(
                inputs = inputs ,
                filters = 8 ,
                kernel_size = [ 5 , 5 ],
                strides = [ 5, 5] ,
                padding = 'same' ,
                name = "d_conv1" ,
                kernel_initializer =  tf.random_normal_initializer( stddev = stddev ) , 
                bias_initializer =  tf.random_normal_initializer( stddev = stddev )
            )
           
            conv1 = self.batch_norm2( conv1 , "d_norm1" )
            conv1 = self.lrelu( conv1 , name = "d_relu1")
            
            print("shape d_conv1 ")
            print(conv1.shape)
            # conv2 [-1 , 5 , 5 , 32 ]
            conv2 = tf.layers.conv2d(
                inputs = conv1 ,
                filters = 32 ,
                strides = [3,3 ],
                kernel_size = [3,3] , 
                padding = 'same' ,
                name = "d_conv2" ,
                kernel_initializer =  tf.random_normal_initializer( stddev = stddev ) , 
                bias_initializer =  tf.random_normal_initializer( stddev = stddev )
            )
            
            conv2 = self.batch_norm2( conv2 , "d_norm2" )
            conv2 = self.lrelu(conv2 , name="d_relu2")
            
            conv3 = tf.layers.conv2d(
                inputs = conv2 ,
                filters = 64 ,
                strides = [5,5] ,
                kernel_size = [5,5] ,
                padding = "same" ,
                name = "d_conv3" ,
                kernel_initializer =  tf.random_normal_initializer( stddev = stddev ) , 
                bias_initializer =  tf.random_normal_initializer( stddev = stddev )
            )
           
            conv3 = self.batch_norm2( conv3 , "d_norm3" )
            conv3 = self.lrelu(conv3, name = "d_relu23")
            
            print("shape d_conv2 ")
            print(conv3.shape)
            
            conv3 = tf.reshape( conv3  , [mb_size , -1] )
            output = self.linear( conv3 , 1 , "d_linear_out" )
            
           
            print("shape output discriminator ")
            print(output.shape)
            # output [ logits , sigmoid ]
            
            return   output , tf.nn.sigmoid( output ) 
        

    def build_inputs_dis( self, band1 , band2 ):

         band1 = tf.reshape( band1  , [-1 ,  75 , 75 ] )
         band2 =  tf.reshape( band2  , [-1 ,  75 , 75 ] )
            
         inputs = tf.stack( [band1 , band2] , axis = 3 )
         print("oie que rika sheip")
         print( inputs.shape)
         inputs = tf.reshape( inputs , [mb_size , 75,75 , 2  ])

         return inputs 

    def lrelu(self ,  X , leak = 0.2 , name = "lrelu"):
        with tf.variable_scope(name):
            f1 = 0.5*(1 + leak )
            f2 = 0.5*(1-leak) 
            return f1*X + f2*tf.abs(X)

    
    def linear( self , input_ , output_size , scope=None , stddev = 0.02, with_w = False):
        #linear layer
        shape = input_.get_shape().as_list()

        with tf.variable_scope(scope or "Linear"):
            m = tf.get_variable( "{}_matrix".format(scope) , [shape[1] , output_size] , tf.float32 , tf.random_normal_initializer( stddev = stddev ))
            b = tf.get_variable("{}_bias".format(scope) , [output_size ])
            if with_w:

                return tf.matmul( input_, m ) + b , m , b
            else :

                return tf.matmul( input_ , m ) + b

    def build(self , X  , global_step ):

        self.init_vars()
        inputs = self.build_inputs_dis(X[0] , X[1])
        
        #var_list_G = self.varlist_G()
        #var_list_D = self.varlist_D()
        
        G_sample = self.generator_mlp(self.z)
        
        D_real , _  = self.discriminator_mlp( inputs , False )
        
        D_fake , _  = self.discriminator_mlp( G_sample  , True)


        # real images are (1) , fake ones are (0)
        
        D_loss_real = self.cross_entropy_loss( D_real , tf.ones_like(D_real)  )
        #  fake ones are (0) 
        D_loss_fake = self.cross_entropy_loss(D_fake , tf.zeros_like(D_fake) )

        self.D_loss = tf.reduce_mean( D_fake)  - tf.reduce_mean(D_real )
        self.G_loss = -tf.reduce_mean(D_fake )       
        GAN_loss = tf.reduce_mean( self.G_loss + self.D_loss )
        
   
        t_vars = tf.trainable_variables()

        #var_list_d = [ var for var in t_vars if 'd_' is in var.name ]
        var_list_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator" )
        var_list_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator" )
        
        var_list_d = [ var for var in t_vars if 'd_'  in var.name ]
        var_list_g = [ var for var in t_vars if 'g_'  in var.name ]

        print(var_list_g ) 
        self.D_solver = tf.train.RMSPropOptimizer(
            learning_rate = 1e-5  ).minimize( self.D_loss , var_list = var_list_d  )

        self.G_solver = tf.train.RMSPropOptimizer(
            learning_rate = 1e-5
        ).minimize( self.G_loss , var_list = var_list_g , global_step = global_step )

        self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in var_list_d]

        tf.summary.scalar('G_loss' , self.G_loss )
        tf.summary.scalar('D_loss' , self.D_loss )
        tf.summary.scalar('GAN_loss',  GAN_loss )
        
        self.build_merge()
        self.G_sample = G_sample
      
        
    def train(self , start_step ,epochs , sess , tb_dir   ):
        
        self.build_writer( tb_dir , sess )
        
        for it in range( start_step , epochs):
            D_loss_curr = None
            n_d = 10 if it < 25 or (it+1) % 500 == 0 else 5
            print("training discriminator {} before generator".format(n_d))
            for _ in range(n_d):

           
                
                ## updating D network 
                _ , D_loss_curr, _   = sess.run( [ self.D_solver , self.D_loss , self.clip_D  ] ,feed_dict = { self.z : self.sample_z(  mb_size , z_dim) }   )

                
            # updating G network - two times 
            _ , G_loss_curr = sess.run( [self.G_solver , self.G_loss ]  , feed_dict = { self.z : self.sample_z(  mb_size , z_dim) } ) 
            
            print("Step: {}".format(it) )
            if it % 10 == 0 :
                
                summary = sess.run( self.merged_op ,feed_dict = { self.z : self.sample_z(  mb_size , z_dim) }  )
                self.writer.add_summary( summary , it )
                print("Step: {}".format(it) )
                print("Generator Loss:{}".format( G_loss_curr ))
                print("Discriminator Loss:{}".format( D_loss_curr ))
                
        return


    def generate(self, sess ):

        
        z = self.sample_z( mb_size , z_dim )
        samples  = sess.run( self.G_sample , feed_dict = {self.z : z })

        return samples
        
        
    def sample_z(self , m, n):
        
        return np.random.normal( 0 , 0.1 , size=[m, n])

        #return np.random.uniform(-1., 1., size=[m, n])



    def build_merge(self):

        self.merged_op = tf.summary.merge_all()
        
        return 

    def build_writer(self , tb_dir , sess ):

        self.writer = tf.summary.FileWriter(tb_dir , sess.graph)
        return 


    def cross_entropy_loss(self , logits , labels , name="xentropy" ):
        o = tf.clip_by_value( logits , 1e-7, 1. - 1e-7)
        xentropy = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits = o, labels = labels))

        return xentropy


    def batch_norm2( self , x , name ,  train = True):
        momentum = 0.5
        epsilon = 1e-5
        return tf.layers.batch_normalization(
            x ,
            momentum = momentum ,
            epsilon = epsilon ,
            scale = True ,
            training = train ,
            name = name
        )
    
        
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
