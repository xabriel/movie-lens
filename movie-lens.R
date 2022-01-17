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
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# Make sure userId and movieId in train_set set are also in test_set set
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# To test our progress, we define a function to compute the
# root-mean-square error:
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# as a base model, let's calculate the average rating mu
mu <- mean(train_set$rating)

# we can now compute a simple model that just guesses the average rating.
# In the literature, this is called a 'baseline rating'.
model_baseline <- mu
rmse_baseline <- RMSE(test_set$rating, model_baseline)
print(paste("RMSE of baseline model: ", rmse_baseline))

# let's add a movie effect and a temporal movie effect; the rationale being that some movies are simply rated higher than other movies, and also that movie go thru cycles.

# create n temporal bins that span the full dataset

bin_range <- append(c(1,2,3,4), seq(5, 100, 5))

binned_movie_effect_rmses <- sapply(bin_range, function(num_bins) {

  max_timestamp <- max(train_set$timestamp) + 86400 # one day margin to be safe
  min_timestamp <- min(train_set$timestamp) - 86400
  bin_size <- (max_timestamp - min_timestamp) / num_bins
  
  # calculate the temporal movie effect
  binned_movie_effect <- train_set %>%
    mutate(bin = ceiling( (timestamp - min_timestamp) / bin_size )) %>%
    group_by(movieId, bin) %>%
    summarize(b_m_t = mean(rating - mu))
  
  # the test data likely has movieId-bin permutations that we did not see in
  # the train data. So let's also compute a regular movie effect
  regular_movie_effect <- train_set %>%
    group_by(movieId) %>%
    summarize(b_m = mean(rating - mu))
  
  # apply the movie effect to test set, preferring the temporal one if available
  movie_effect_on_test_set <-
    test_set %>%
    mutate(bin = ceiling( (timestamp - min_timestamp) / bin_size )) %>%
    left_join(binned_movie_effect, by = c("movieId", "bin")) %>%
    left_join(regular_movie_effect, by = "movieId")
  
  model_movie_effect <-
    mu + coalesce(movie_effect_on_test_set$b_m_t, movie_effect_on_test_set$b_m)
  #rmse_movie_effect <- RMSE(test_set$rating, model_movie_effect)
  RMSE(test_set$rating, model_movie_effect)
})

# now repeat once more with best bin_num
# TODO: we should really make a function out of this calculation and reuse it

best_bin_num <- bin_range[which.min(binned_movie_effect_rmses)]

num_bins <- best_bin_num
max_timestamp <- max(train_set$timestamp) + 86400 # one day margin to be safe
min_timestamp <- min(train_set$timestamp) - 86400
bin_size <- (max_timestamp - min_timestamp) / num_bins

# calculate the temporal movie effect
binned_movie_effect <- train_set %>%
  mutate(bin = ceiling( (timestamp - min_timestamp) / bin_size )) %>%
  group_by(movieId, bin) %>%
  summarize(b_m_t = mean(rating - mu))

# the test data likely has movieId-bin permutations that we did not see in
# the train data. So let's also compute a regular movie effect
regular_movie_effect <- train_set %>%
  group_by(movieId) %>%
  summarize(b_m = mean(rating - mu))

# apply the movie effect to test set, preferring the temporal one if available
movie_effect_on_test_set <-
  test_set %>%
  mutate(bin = ceiling( (timestamp - min_timestamp) / bin_size )) %>%
  left_join(binned_movie_effect, by = c("movieId", "bin")) %>%
  left_join(regular_movie_effect, by = "movieId")

model_movie_effect <-
  mu + coalesce(movie_effect_on_test_set$b_m_t, movie_effect_on_test_set$b_m)
rmse_movie_effect <- RMSE(test_set$rating, model_movie_effect)
print(paste("RMSE of baseline + temporal movie effect: ", rmse_movie_effect))

# let's add a user effect; some users are harsh critics, some always give 5's.
user_effect <- train_set %>%
  mutate(bin = ceiling( (timestamp - min_timestamp) / bin_size )) %>%
  left_join(binned_movie_effect, by = c("movieId", "bin")) %>%
  left_join(regular_movie_effect, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - coalesce(b_m_t, b_m)))
# rmse
user_effect_on_test_set <- test_set %>% inner_join(user_effect, by = "userId") %>% pull(b_u)
model_movie_and_user_effect <-
  mu + coalesce(movie_effect_on_test_set$b_m_t, movie_effect_on_test_set$b_m) +
  user_effect_on_test_set
rmse_movie_and_user_effect <- RMSE(test_set$rating, model_movie_and_user_effect)
print(paste("RMSE of baseline + temporal movie effect and user effect: ", rmse_movie_and_user_effect))

# after fixing to use just train set, we now get .8625 :(

# 0.83 !!!!!

# let's now regularize the effects since some users do few ratings
# and some movies have few ratings

lambdas <- seq(0, 10, 0.25)

regularized_rmses <- sapply(lambdas, function(lambda){
  mu <- mean(train_set$rating)

  num_bins <- best_bin_num
  max_timestamp <- max(train_set$timestamp) + 86400 # one day margin to be safe
  min_timestamp <- min(train_set$timestamp) - 86400
  bin_size <- (max_timestamp - min_timestamp) / num_bins
  
  # calculate the temporal movie effect
  binned_movie_effect <- train_set %>%
    mutate(bin = ceiling( (timestamp - min_timestamp) / bin_size )) %>%
    group_by(movieId, bin) %>%
    summarize(b_m_t = sum(rating - mu)/ (n()+lambda))
  
  # the test data likely has movieId-bin permutations that we did not see in
  # the train data. So let's also compute a regular movie effect
  regular_movie_effect <- train_set %>%
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu)/ (n()+lambda))
  
  # apply the movie effect to test set, preferring the temporal one if available
  movie_effect_on_test_set <-
    test_set %>%
    mutate(bin = ceiling( (timestamp - min_timestamp) / bin_size )) %>%
    left_join(binned_movie_effect, by = c("movieId", "bin")) %>%
    left_join(regular_movie_effect, by = "movieId")
  
  model_movie_effect <-
    mu + coalesce(movie_effect_on_test_set$b_m_t, movie_effect_on_test_set$b_m)
  rmse_movie_effect <- RMSE(test_set$rating, model_movie_effect)
  print(paste("RMSE of baseline + temporal movie effect + regularization: ", rmse_movie_effect))
  
  # let's add a user effect; some users are harsh critics, some always give 5's.
  user_effect <- train_set %>%
    mutate(bin = ceiling( (timestamp - min_timestamp) / bin_size )) %>%
    left_join(binned_movie_effect, by = c("movieId", "bin")) %>%
    left_join(regular_movie_effect, by = "movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu - coalesce(b_m_t, b_m)) / (n() + lambda))
  # rmse
  user_effect_on_test_set <- test_set %>% inner_join(user_effect, by = "userId") %>% pull(b_u)
  model_movie_and_user_effect <-
    mu + coalesce(movie_effect_on_test_set$b_m_t, movie_effect_on_test_set$b_m) +
    user_effect_on_test_set
  rmse_movie_and_user_effect <- RMSE(test_set$rating, model_movie_and_user_effect)
  print(paste("RMSE of baseline + temporal movie effect and user effect + regularization: ", rmse_movie_and_user_effect))

  rmse_movie_and_user_effect
})

qplot(lambdas, regularized_rmses)

best_lambda <- lambdas[which.min(regularized_rmses)]


# finally, test with validation:

mu <- mean(train_set$rating)

num_bins <- best_bin_num
max_timestamp <- max(train_set$timestamp) + 86400 # one day margin to be safe
min_timestamp <- min(train_set$timestamp) - 86400
bin_size <- (max_timestamp - min_timestamp) / num_bins

# calculate the temporal movie effect
binned_movie_effect <- train_set %>%
  mutate(bin = ceiling( (timestamp - min_timestamp) / bin_size )) %>%
  group_by(movieId, bin) %>%
  summarize(b_m_t = sum(rating - mu)/ (n()+best_lambda))

# the test data likely has movieId-bin permutations that we did not see in
# the train data. So let's also compute a regular movie effect
regular_movie_effect <- train_set %>%
  group_by(movieId) %>%
  summarize(b_m = sum(rating - mu)/ (n()+best_lambda))

# apply the movie effect to test set, preferring the temporal one if available
movie_effect_on_test_set <-
  validation %>%
  mutate(bin = ceiling( (timestamp - min_timestamp) / bin_size )) %>%
  left_join(binned_movie_effect, by = c("movieId", "bin")) %>%
  left_join(regular_movie_effect, by = "movieId")

model_movie_effect <-
  mu + coalesce(movie_effect_on_test_set$b_m_t, movie_effect_on_test_set$b_m, 0)
rmse_movie_effect <- RMSE(validation$rating, model_movie_effect)
print(paste("RMSE of baseline + temporal movie effect + regularization: ", rmse_movie_effect))

# let's add a user effect; some users are harsh critics, some always give 5's.
user_effect <- train_set %>%
  mutate(bin = ceiling( (timestamp - min_timestamp) / bin_size )) %>%
  left_join(binned_movie_effect, by = c("movieId", "bin")) %>%
  left_join(regular_movie_effect, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - mu - coalesce(b_m_t, b_m, 0)) / (n() + best_lambda))
# rmse
user_effect_on_test_set <- validation %>% inner_join(user_effect, by = "userId") %>% pull(b_u)
model_movie_and_user_effect <-
  mu + coalesce(movie_effect_on_test_set$b_m_t, movie_effect_on_test_set$b_m, 0) +
  user_effect_on_test_set
rmse_movie_and_user_effect <- RMSE(validation$rating, model_movie_and_user_effect)
print(paste("RMSE of baseline + temporal movie effect and user effect + regularization: ", rmse_movie_and_user_effect))

# "RMSE of baseline + temporal movie effect and user effect + regularization:  0.861813658382263"
# 