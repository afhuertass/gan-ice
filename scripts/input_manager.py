
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
import multiprocessing

from tensorflow.contrib.data import Dataset , Iterator
from tensorflow.contrib.data import TFRecordDataset 

from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor 


size = 75*75
def parse_examples( examples ):

    feature_map = {

        'b1' : tf.FixedLenFeature( shape = [ size ] , dtype = tf.float32 )  ,  
        'b2' :  tf.FixedLenFeature( shape = [ size ] , dtype = tf.float32 ) , 
    }


    features = tf.parse_example( examples , feature_map )

    return features['b1'] , features['b2']


class InputManager():

    def __init__( self , datafile , batch_size ,repeat ) :

        self.data_path = datafile

        self.data = None
        self.dataset = TFRecordDataset( datafile , "GZIP" )
        self.dataset = self.dataset.map(  self._parse_sample )
        self.dataset = self.dataset.repeat( repeat )
        self.data = self.dataset.batch( batch_size )



    def _parse_sample( self , example  ) :

        feature_map = {

            'b1' : tf.FixedLenFeature( shape = [ size ] , dtype = tf.float32 )  ,  
            'b2' :  tf.FixedLenFeature( shape = [ size ] , dtype = tf.float32 ) , 
        }
        
        parsed = tf.parse_single_example( example , feature_map )
        
        return parsed['b1'] , parsed['b2'] 


        
    def iterator( self ):

        return self.data.make_initializable_iterator()
    
    
