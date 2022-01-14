##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
# Start of code by Xabriel J Collazo Mojica
##########################################################

# First, we divide the dataset into training and test sets
set.seed(123)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# we'd like the users and movies in the test set to be mutually exclusive
# from the movies and users of the train set
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# To test our progress, we define a function to compute the
# root-mean-square error:
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# as a base model, let's calculate the average rating mu
mu <- mean(edx$rating)

# we can now compute a simple model that just guesses the average rating.
# In the literature, this is called a 'baseline rating'.
model_baseline <- mu
rmse_baseline <- RMSE(test_set$rating, model_baseline)
print(paste("RMSE of baseline model: ", rmse_baseline))

# let's add a movie effect; the rationale being that some movies are simply
# rated higher than other movies.
movie_effect <- edx %>%
  group_by(movieId) %>%
  summarize(b_m = mean(rating - mu))

# let's rerun the RMSE to see where we at:
movie_effect_on_test_set <- test_set %>% inner_join(movie_effect, by = "movieId") %>% pull(b_m)
model_movie_effect <- mu + movie_effect_on_test_set
rmse_movie_effect <- RMSE(test_set$rating, model_movie_effect)
print(paste("RMSE of baseline + movie effect: ", rmse_movie_effect))

# now, lets add a user effect; 