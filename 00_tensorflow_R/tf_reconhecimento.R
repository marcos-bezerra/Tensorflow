# informa o diretório de trabalho
setwd("/home/marcos/Documentos/02_DSA/00_Tensorflow/00_tensorflow_R/recognition_R_tf/")
getwd()

# carrega as bibliotecas
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install(version = "3.9") # atualizar todos os pacotes para uma versão específica
BiocManager::install() # atualiza pacotes instalados
BiocManager::version() # versão do Bioconductor
BiocManager::valid() # valida se todos os pacotes estão na mesma versão
BiocManager::available("EBImage") # verifica se o pacote existe
BiocManager :: install ("EBImage") # instala um pacote
browseVignettes("EBImage") # visualizar documentação do pacote

install_tensorflow(version = "1.13.1")

library(reticulate)
library(tensorflow)
library(EBImage)
library(keras) # conexão do R com o tensorflow
library(caret) # auxilia na separação dos dados train e test
library(pbapply) 

# cria um path para os diretórios das imagens
img_A_dir <- "imagens/caes"
img_B_dir <- "imagens/gatos"
test_dir <- "imagens/images_test"

# apresenta a leitura de uma imagem
img_test <- readImage("imagens/images_test/cat1.jpeg")
print(img_test)
str(img_test)
display(img_test)

# determina o tamanho das imagens 30 x 30
width <- 30
height <- 30

# transforma a dimensão das imagens para quadrado
resize_img <- function(dir_path, width, height){
  img_size <- width * height * 3 # área da imagem : qtde de pixels da imagem
  img_name <-  list.files(dir_path)
  print(paste("Iniciando Processo", length(img_name), "imagens"))
  
  lista_parametros <- pblapply(img_name, function(imgname){
    img <- readImage(file.path(dir_path, imgname))
    img_resized <- resize(img, w = width, h = height) # redimensiona o tamanho
    img_matriz <- as.matrix(img_resized@.Data)
    img_vector <- as.vector(t(img_matriz))
    
    return(img_vector)
  })
  
  feature_matriz <- do.call(rbind, lista_parametros)
  feature_matriz <- as.data.frame(feature_matriz)
  
  names(feature_matriz) <- paste0("pixel", c(1:img_size))
  return(feature_matriz) # retorna a matriz completa
}

# chama a função para redimensionar o tamanho da imagem
img_A_data <- resize_img(dir_path = img_A_dir, width = width, height = height)
img_B_data <- resize_img(dir_path = img_B_dir, width = width, height = height)

# adicionar as etiquetas para os tipos
img_A_data$label <- 0
img_B_data$label <- 1

# agrupa dos dois dataset
allData <- rbind(img_A_data, img_B_data)

# cria uma amostra com dos dados para treino e para teste
indices <- createDataPartition(allData$label, p=0.70, list = FALSE)
train <- allData[indices, ] # " p " dados para train
test <- allData[-indices,] # " 1 - p " dados para test

# passar os valores "indices" para categoricos
trainLabels <- to_categorical(train$label)
testLabels <- to_categorical(test$label)
trainLabels
testLabels

# transforma em matriz 
x_train <- data.matrix(train[,-ncol(train)])
y_train <- data.matrix(train[,ncol(train)])

x_test <- data.matrix(test[,-ncol(test)])
y_test <- data.matrix(test[,ncol(test)])

# criando um modelo sequencial Keras
model <- keras_model_sequential()
model %>%
  layer_dense(units = 256, activation = 'relu', input_shape = c(2700)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = 2, activation = 'softmax')
# 256*256*3 = 196608
# 30*30*3

# descreve a estrutura do modelo
summary(model)

# compilar o modelo
model %>%
  compile(loss = "binary_crossentropy", #categorical_crossentropy
          optimizer = "adam", #optimizer_adam()
          metrics = "categorical_accuracy" #accuracy
          )

# passar o modelo para os dados de treinamento
history <- model%>%
  fit(x_train,
      trainLabels,
      epochs = 10,
      batch_size = 32,
      validation_split = 0.2
      )

# plotar o histórico de aprendizagem
plot(history)

# avaliar o modelo na base teste
model %>% evaluate(x_test, testLabels, verbose = 1)

# prevendo o modelo
pred <- model %>% predict_classes(x_test)
table(Predicted = pred, Reais = y_test)

# testando o modelo com um nova imagem
test <- resize_img(test_dir, width = width, height = height)
pred_test <- model %>% predict_classes(as.matrix(test))
pred_test

#images_test <- 1 cat
#img_A_data$label <- 0 dog
#img_B_data$label <- 1 cat