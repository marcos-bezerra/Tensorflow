#%tensorflow_version 2.x

#Hello World Tensorflow

# Configurando Nível de Log
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# Cria o objeto Tensorflow chamado Hello
# constantes são objetos cujo valor não pode ser alterado.
hello = tf.constant('Hello World!')

print(hello)
tf.print(hello)
print('\n')
