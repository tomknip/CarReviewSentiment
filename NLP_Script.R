library(tidyverse)
library(tm)
library(wordcloud2)
library(wordcloud)
library(tidytext)
library(stringr)
library(dplyr)
library("rpart")
library(caTools)
library(randomForest)
library(reshape2)

# Directory of the datasets
directory = "OpinRankDatasetWithJudgments/cars/data/2007/"

# Functions for File Handling
processFile <- function(path){
  input <- readLines(path)
  reviews <- c()
  for(i in c(2:length(input))){
    if (startsWith(input[i], "<TEXT>")){
      text = stringr::str_remove(input[i], "<TEXT>")
      text = stringr::str_remove(text, "</TEXT>")
      reviews <- c(reviews, text)
    }
  }
  
  return(reviews)
}

prepCorpusBrand <- function (directory, brand_name, func=NULL){
  brand_files <- list.files(directory, pattern= paste("2007_", brand_name, sep=""))
  print(paste("There have been found", length(brand_files), "different files for this brand", sep=" "))
  brand_reviews <- c()
  for (i in c(1:length(brand_files))){
    brand_reviews <- c(brand_reviews, processFile(paste(directory, brand_files[i], sep = "")))
  } 
  if(is.null(func)){
     return(brand_reviews)
  }
  else{
    corpus <- func(VectorSource(brand_reviews))
    return(corpus)
  }
}


# Part 1: Word Cloud
# Quick look at one brand
bmw_corpus <- prepCorpusBrand(directory, 'bmw', Corpus)
length(bmw_corpus)
inspect(bmw_corpus[4])

# Check term document matrix
tdm <- TermDocumentMatrix(bmw_corpus)
freq <- rowSums(as.matrix(tdm))
# Zipfs law also holds here
plot(sort(freq, decreasing=TRUE))

# Bag of words
bagOfWordsBrand <- function(brand, directory){
  set.seed(123)
  corpus_new <- prepCorpusBrand(directory, brand, VCorpus)
  carStopwords = c(stopwords(), "car", "cars", brand, "drive", "driving")
  tdm <- TermDocumentMatrix(corpus_new, 
                            control=list( 
                                         removeWhitespace=TRUE,
                              removePunctuation = TRUE,
                                         stemming=FALSE,
                              stopwords=carStopwords))
  
  freq<- sort(rowSums(as.matrix(tdm)), decreasing = T)
  wordcloud(words=names(freq), freq=freq, random.order=FALSE, rot.per = 0.25,
                          min.freq = 25, main = "Title")
}

# Plot Wordclouds
for (name in c('BMW', "Mazda", "Cadillac")) {
  jpeg(paste(name, "_wordcloud.jpg", sep= ""))
  bagOfWordsBrand(tolower(name), directory)
  dev.off()
}

# Part 2: Sentiment Analysis
# Approach 1: Dictionary Based Sentiment Analysis
dictBasedSentAnalysis <- function(brand){
  corp <- prepCorpusBrand(directory, brand)
  df <- data.frame(matrix(unlist(corp), nrow=length(corp), byrow=T))
  colnames(df)<- c('Text')
  df$'DocNo' <- as.integer(rownames(df))
  head(df)
  tokens <- unnest_tokens(df, "word", "Text")
  words_with_sentiment <- inner_join(tokens, get_sentiments('bing'))
  count(words_with_sentiment, sentiment)
  outcome <- count(group_by(words_with_sentiment, DocNo), sentiment)
  outcome <- spread(outcome, sentiment,n, fill = 0)
  outcome$class <- ifelse(outcome$positive - outcome$negative < 0, -1, 1)
  return(outcome)
}

# Plot Results
for (name in c('BMW', "Mazda", "Cadillac")) {
  print(name)
  outcome <- dictBasedSentAnalysis(tolower(name))
  jpeg(paste(name, "_DicSentAnalysis.jpg", sep= ""))
  barplot(tally(group_by(outcome, class))$n / nrow(outcome), names.arg = c('Negative', 'Positive')
          , main = name)  
  dev.off()
}


# Approach 2: Learning-Based Sentiment Analysis
# Create the word matrix
# For this we need to have both data from movies as well as the cars
# Read movies dataset
movies <- read.csv("IMDB Dataset/IMDB Dataset.csv")
prop <- 0.04 * nrow(movies)
movies <- movies[c(1:prop), ]

# Read car dataset
bmw_corpus <- prepCorpusBrand(directory, 'bmw')
mazda_corpus <- prepCorpusBrand(directory, 'mazda')
cadillac_corpus <- prepCorpusBrand(directory, 'cadillac')
car_text <- c(bmw_corpus, mazda_corpus, cadillac_corpus)
brand_column <- c(rep("BMW", length(bmw_corpus)), rep("Mazda", length(mazda_corpus)),
                  rep("Cadillac", length(cadillac_corpus)))
car_df <- data.frame(car_text, brand_column, sentiment = c('NULL'))
colnames(car_df) <- c('review', 'Brand', 'sentiment')
input_df <- rbind(movies, car_df[,c('review', 'sentiment')])

# Create corpus for pre-processing and perform pre-processing
preprocess_data <- function(input_df){
  ml_input <- Corpus(VectorSource(input_df$review))
  ml_input <- tm_map(ml_input, PlainTextDocument)
  ml_input <- tm_map(ml_input, tolower)
  ml_input <- tm_map(ml_input, removePunctuation)
  ml_input <- tm_map(ml_input, removeWords, c("movies", stopwords(), "car", "cars"
                                              , "bmw", 'mazda', 'cadillac', "drive", 
                                              "driving"))
  ml_input <- tm_map(ml_input, stemDocument)
  return(ml_input)
}

ml_input <- preprocess_data(input_df)
ml_dtm <- DocumentTermMatrix(ml_input)
sparse_dtm <- removeSparseTerms(ml_dtm, 0.995)
sparseReviews <- as.data.frame(as.matrix(sparse_dtm))
colnames(sparseReviews) <- make.names(colnames(sparseReviews))
sparseReviews$Positive <- ifelse(input_df$sentiment == "positive", TRUE, FALSE)

summary(sparseReviews$Positive)

split <- c(rep('train', 0.8*nrow(movies)), rep('test', 0.2*nrow(movies)), 
           rep('car', nrow(car_df)))


# Make a split to train, validate on movie data and use on car data
train <- subset(sparseReviews, split == "train")
test <- subset(sparseReviews, split == "test")
car_data <- subset(sparseReviews, split == "car")

# Make accuracy function
accuracy <- function(res_table){
  return(sum(diag(res_table))/sum(res_table))
}

# Test different models
# Model 1: Tree
CART_model <- rpart(Positive ~ . , data=train, method="class")
predictions <- predict(CART_model, newdata = test, type='class')
conf_matrix <- table(test$Positive, predictions)
conf_matrix
accuracy(conf_matrix)

# Model 2: Random Forest
set.seed(123)
train$Positive <- as.factor(train$Positive)
RF_model <- randomForest(Positive ~ . , data=train)
predictions <- predict(RF_model, newdata = test, type='class')
conf_matrix <- table(test$Positive, predictions)
conf_matrix
accuracy(conf_matrix)

# Model 3: Logistic Regression
lr_model <- glm(Positive~ . , data= train, family = 'binomial')
predictions <- predict(lr_model, newdata = test, type='response')
conf_matrix <- table(test$Positive, predictions > 0.5)
conf_matrix
accuracy(conf_matrix)

# Use Random Forest model on car reviews
predictions_car <- predict(RF_model, newdata = car_data, type="class")
car_df$class <- predictions_car
car_df$sentiment <- ifelse(car_df$class == "FALSE", 'negative', 'positive')

# Plot Results
for (name in c('BMW', "Mazda", "Cadillac")) {
  # Barplot of percentage of positive / negative reviews per brand
  df <- car_df[car_df$Brand == name, ]
  print(sum(df$sentiment == 'positive'))
  jpeg(paste(name, "_ML_sentiment.jpg", sep= ""))
  barplot(tally(group_by(df, sentiment))$n / nrow(df), names.arg = c('Negative', 'Positive')
  , main = name)  
  dev.off()
  
  # Word Cloud divided by sentiment
  prep_df <- df
  prep_df$review <- tolower(prep_df$review)
  prep_df$review <- removePunctuation(prep_df$review)
  prep_df$review <- removeWords(prep_df$review, c("movies", stopwords(), "car", 
                                                  "cars", "bmw", 'mazda', 
                                                  'cadillac', "drive", "driving"))
  
  jpeg(paste(name, "_wordcloud_sentiment.jpg", sep= ""))
  prep_df %>% unnest_tokens(word, review) %>% 
    count(word, sentiment, sort=TRUE) %>% 
    acast(word ~ sentiment, value.var='n', fill=0) %>%
    comparison.cloud(colors=c('orange', 'blue'), max.words=100)
  dev.off()
}

# Approach 1 on test movies data
idx <- seq(0.8*nrow(movies)+1, nrow(movies))
df <- movies[idx, ]
df$DocNo <- idx
tokens <- unnest_tokens(df, "word", "review")
words_with_sentiment <- inner_join(tokens, get_sentiments('bing'))
count(words_with_sentiment, sentiment)
outcome <- group_by(words_with_sentiment, DocNo) %>% count(sentiment)
outcome <- spread(outcome, sentiment, n, fill = 0)
`%notin%` <- Negate(`%in%`)
missing <- idx[idx %notin% outcome$DocNo]
outcome[nrow(outcome) + 1, ] <- list(missing[1], 0, 0)
outcome <- outcome[order(outcome$DocNo), ]
df$predicted <- ifelse(outcome$positive - outcome$negative < 0, 'negative', 'positive')
conf_matrix <- table(df$sentiment, df$predicted)
conf_matrix
accuracy(conf_matrix)



