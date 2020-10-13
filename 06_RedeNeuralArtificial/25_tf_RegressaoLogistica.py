# Regressão Logística

"""
A regressão logística é uma técnica estatística que tem como objetivo produzir, 
a partir de um conjunto de observações, um modelo que permita a predição de valores 
tomados por uma variável categórica, frequentemente binária, a partir de uma série 
de variáveis explicativas contínuas e/ou binárias A regressão logística é amplamente 
usada em ciências médicas e sociais, e tem outras denominações, como modelo logístico, 
modelo logit, e classificador de máxima entropia. Wikipedia
"""

# Com Regressão logística, buscamos uma função que nos diga qual é a probabilidade de um elemento pertencer a uma classe. 

# A aprendizagem supervisionada é configurada como um processo iterativo de otimização dos pesos.
# Estes são então modificados com base no desempenho do modelo.

# De fato, o objetivo é minimizar a função de perda, que indica o grau em que o comportamento do modelo se desvia do desejado. 
# O desempenho do modelo é então verificado em um conjunto de teste, consistindo em imagens diferentes das de treinamento.

# Os passos básicos do treinamento que vamos implementar são os seguintes: 

# 1- Os pesos são inicializados com valores aleatórios seguindo uma distribuição normal.
# 2- Para cada elemento do conjunto de treino é calculado o erro, ou seja, a diferença entre a saída prevista e a saída real. Este erro é usado para ajustar os pesos. 
# 3- O processo é repetido em todos os exemplos do conjunto de treinamento até que o erro em todo o conjunto de treinamento não seja inferior a um certo limite, 
# ou até que o número máximo de iterações seja atingido.

# Nosso objetivo é a classificação de imagens de peças de vestuário.
# Classes do dataset fashion_mnist - ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


########################################## Importando bibliotecas
import os # configurando o nível de log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist # importando dataset fashion_mnist 

import warnings; warnings.simplefilter('ignore')


########################################## Importando dataset
# x: images
# Y: labels
(x_treino, y_treino), (x_teste, y_teste) = fashion_mnist.load_data() # metodo load_data divide em treino e teste


########################################## Normalizando as imagens
# 0 = branco
# 255 = preto
# a imagem é uma matriz com vários pixels, que varia de 0 a 255 ( tom cinza )
# dividindo por 255 apenas estamos diminuindo a escala da matriz
x_treino = x_treino/255. # em python o ponto irá determinar que é um tipo float
x_teste = x_teste/255. # sem o ponto será considerado tipo inteiro

# O reshape transforma o shape de x que é uma matriz bidimensional de 28 x 28 para um vetor unidimensional 784 posições
# transforma a matriz bidimensional em unidimensional
# a função tf.reshape retorna um tensor de uma dimensão
# o reshape não é aplicado em Y, este representa as classes / labels
x_treino = tf.reshape(x_treino, shape = (-1, 784)) # o valor -1 indica ao tf que está sendo removido uma das dimensões
x_teste  = tf.reshape(x_teste, shape = (-1, 784)) # transformando de bidimensional para unidimensional


########################################## Construindo o Modelo
# definindo os pesos, variáveis W (weight) e b (bias - viés)
# inicializando os Coeficientes de Forma Randômica com Distribuição Normal
# distribuição normal é um conjunto de dados aleatórios com média igual a 0 e desvio padrão igual a 1.
pesos = tf.Variable(tf.random.normal(shape = (784, 10), dtype = tf.float64)) # matriz bidimensional
vieses  = tf.Variable(tf.random.normal(shape = (10,), dtype = tf.float64)) # matriz unidimensional


########################################## Função para o cálculo da regressão logísitica
# g(y) = β(x) + βo  
def logistic_regression(x): # vetor unidimensional
    lr = tf.add(tf.matmul(x, pesos), vieses)
    # função tf.matmul irá multiplica matrizes x e pesos
    # função tf.add irá somar o resultado da função tf.matmul com o vieses
    return lr

# Minimizando o erro usando cross entropy (Função de Custo).
# A fim de treinar nosso modelo, devemos definir como identificar a precisão. 
# Nosso objetivo é tentar obter valores de parâmetros W e b que minimizem o valor da métrica que indica quão ruim é o modelo.
# Diferentes métricas calculam o grau de erro entre a saída desejada e as saídas de dados de treinamento. 
# Uma medida comum de erro é o "Erro Quadrático Médio ou a Distância Euclidiana Quadrada".
# No entanto, existem algumas descobertas de pesquisa que sugerem usar outras métricas para uma rede neural.
# Neste exemplo, usamos a chamada função de erro de entropia cruzada.

def cross_entropy(y_true, y_pred): # função de custo
    y_true = tf.one_hot(y_true, 10) # conversão do valor verdadeiro 
    loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y_pred) # função de custo = entropia cruzada
    return tf.reduce_mean(loss) # calculo do erro médio

# Otimizando a Cost Function
# Em seguida, devemos minimizá-lo usando o algoritmo de otimização de descida de gradiente:
# este calculo define a direção de descida do gradiente, se deve aumentar ou diminuir os valores dos pesos
# o gradiente é o cálculo da derivada
def grad(x, y): # função de otimização da função de custo
    with tf.GradientTape() as tape: # loop
        y_pred = logistic_regression(x) # função do cálculo da regressão logística, valor previsto pelo modelo
        loss_val = cross_entropy(y, y_pred) # cálculo do erro médio
    return tape.gradient(loss_val, [pesos, vieses]) # cálculo do gradiente com o erro médio, peso e viés


########################################## Hiperparâmetros
n_batches = 10000 # quantidade de lotes / batch
learning_rate = 0.01 # taxa de aprendizado que é a magnitude do aumento (0,01 ou 0,001)
batch_size = 128 # lotes de dados a serem colocados na memória para cálculo do SGD


# Cria o otimizador usando SGD (Stochastic Gradient Descent)
optimizer = tf.optimizers.SGD(learning_rate) # calcula a magnitude do aumento no gradiente


########################################## Função para o cálculo da Acurácia
def accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, dtype = tf.int32) # tf.cast converte tipo de dado
    preds = tf.cast(tf.math.argmax(y_pred, axis = 1), dtype = tf.int32) # a função tf.argmax trará a maior probabilidade
    preds = tf.equal(y_true, preds) # verifica a igualdade entre dois valores
    return tf.reduce_mean(tf.cast(preds, dtype = tf.float32)) # cálculo médio da acurácia

# Preparando batches de dados de treino
dataset_treino = tf.data.Dataset.from_tensor_slices((x_treino, y_treino)) # slices irá fatiar x_treino e y_treino
dataset_treino = dataset_treino.repeat().shuffle(x_treino.shape[0]).batch(batch_size) # shuffle embaralha os dados


########################################## Iniciando o treinamento
print ("\nIniciando o Treinamento!")

# Ciclo de treinamento
for batch_numb, (batch_xs_treino, batch_ys_treino) in enumerate(dataset_treino.take(n_batches), 1):

    # Calcula os gradientes
    gradientes = grad(batch_xs_treino, batch_ys_treino)

    # Otimiza os pesos com o valor do gradiente
    optimizer.apply_gradients(zip(gradientes, [pesos, vieses]))

    ### a etapa abaixo não é obrigatoria, os cálculos são refeitos para que se possa imprimir cada iteração

    # Faz uma previsão
    y_pred = logistic_regression(batch_xs_treino)

    # Calcula o erro
    loss = cross_entropy(batch_ys_treino, y_pred)

    # Calcula a acurácia
    acc = accuracy(batch_ys_treino, y_pred)

    # Print
print("Número do Batch: %i, Erro do Modelo: %f, Acurácia em Treino: %f" % (batch_numb, loss, acc))

print ("\nTreinamento concluído!")

########################################## Testando o Modelo
# Preparando os dados de teste
dataset_teste = tf.data.Dataset.from_tensor_slices((x_teste, y_teste))
dataset_teste = dataset_teste.repeat().shuffle(x_teste.shape[0]).batch(batch_size)

print ("\nIniciando a Avaliação com Dados de Teste. Por favor aguarde!")

# Loop pelos dados de teste, previsões e cálculo da acurácia
for batch_numb, (batch_xs_teste, batch_ys_teste) in enumerate(dataset_teste.take(n_batches), 1):
    y_pred = logistic_regression(batch_xs_teste) # previsão
    acc = accuracy(batch_ys_teste, y_pred) #acurácia
    acuracia = tf.reduce_mean(tf.cast(acc, tf.float64)) #acurácia média total

print("\nAcurácia em Teste: %f" % acuracia)

print("\nFazendo Previsão para 10 imagens")

# Obtendo os dados de algumas imagens
dataset_teste = tf.data.Dataset.from_tensor_slices((x_teste, y_teste))
dataset_teste = dataset_teste.repeat().shuffle(x_teste.shape[0]).batch(10) # batch(3), qtde de imagens para teste

# Fazendo previsões
for batch_numb, (batch_xs, batch_ys) in enumerate(dataset_teste.take(1), 1):
    #print("\nImagem:", batch_xs)
    print("Classe Real: \t", batch_ys)
    y_pred = tf.math.argmax(logistic_regression(batch_xs), axis = 1)
    # y_pred = logistic_regression(batch_xs) # sem a função argmax
    print("Classe Prevista:", y_pred)

print("\nExemplo de Peso e Viés Aprendidos:")
print(pesos[2,9])
print(vieses[2])
print("\n")


