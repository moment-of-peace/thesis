import tensorflow as tf

class TFModel(object):
    def __init__(self, params):
        self.batch_size = 
        self.sent_size = 
        self.token_embed_dim = 
        self.word_size = 
        self.char_embed_dim = 
        self.class_num = 
        self.token_indices = tf.placeholder(tf.int32,shape=(params['batch_size'],params['sent_size']),name='token_indices')
        self.char_indices = tf.placeholder(tf.int32,shape=(params['batch_size'],params['sent_size'],params['word_size']),name='char_indices')
        self.true_labels = tf.placeholder(tf.int32,shape=(params['batch_size'],params['sent_size'],params['class_num']),name='true_labels')
        
        token_vec = tf.nn.embedding_lookup(params['token_weights'],
    
def main():
    model
    