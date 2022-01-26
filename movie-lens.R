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

# First, divide the data set into training and test sets
set.seed(1, sample.kind = "Rounding")
test_index <-
  createDataPartition(
    y = edx$rating,
    times = 1,
    p = 0.2,
    list = FALSE
  )
train_set <- edx[-test_index, ]
test_set <- edx[test_index, ]

# Make sure userId and movieId in train_set set are also in test_set
test_set <- test_set %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Let's define our objective function: root mean square error
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings) ^ 2))
}

# First model iteration is just the average:
# Y = mu + error
mu <- mean(train_set$rating)

rmse_baseline <- RMSE(test_set$rating, mu)
print(paste("RMSE of average: ", rmse_baseline))

# Second model iteration uses least squares approximation to discover an item effect
# Y = mu + bi + error
regular_movie_effect <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# apply model to test set
movie_effect_on_test_set <-
  test_set %>%
  left_join(regular_movie_effect, by = "movieId") %>%
  mutate(model = mu + b_i)

rmse_movie_effect <- RMSE(test_set$rating, movie_effect_on_test_set$model)
print(paste("RMSE of average + movie effect: ", rmse_movie_effect))

# There also seems to be a *temporal* item effect. That is, the mean rating of an item
# changes over time. We can use least squares to discover it. Let's incorporate on model:
# Y = mu + bi + bi(t) + error

# divide time into n bins, find what n is best based on RMSE
bin_range <- append(c(1, 2, 3, 4), seq(5, 25, 5))
max_timestamp <- max(train_set$timestamp) + 86400 # one day margin to be safe
min_timestamp <- min(train_set$timestamp) - 86400

binned_movie_effect_rmses <- sapply(bin_range, function(num_bins) {
  bin_size <- (max_timestamp - min_timestamp) / num_bins
  
  regular_movie_effect <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = mean(rating - mu))
  
  binned_movie_effect <- train_set %>%
    left_join(regular_movie_effect, by = "movieId") %>%
    mutate(bin = ceiling((timestamp - min_timestamp) / bin_size)) %>%
    group_by(movieId, bin) %>%
    summarize(b_i_t = mean(rating - mu - b_i))
  
  # apply model to test set
  movie_effect_on_test_set <-
    test_set %>%
    mutate(bin = ceiling((timestamp - min_timestamp) / bin_size)) %>%
    left_join(regular_movie_effect, by = "movieId") %>%
    left_join(binned_movie_effect, by = c("movieId", "bin")) %>%
    # we coalesce b_i_t, as test set may have movie ratings in bins that we did not see
    # in the train set. Thus, in those cases, we default to 0.
    mutate(model = mu + b_i + coalesce(b_i_t, 0))
  
  RMSE(test_set$rating, movie_effect_on_test_set$model)
})

best_bin_num <- bin_range[which.min(binned_movie_effect_rmses)]

print(paste(
  "RMSE of average + temporal movie effect: ",
  min(binned_movie_effect_rmses)
))

# Let's add a user effect. Using least squares we can discover it.
# Y = mu + bi + bi(t) + bu + error

# use best bin size from previous code
bin_size <- (max_timestamp - min_timestamp) / best_bin_num

regular_movie_effect <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

binned_movie_effect <- train_set %>%
  left_join(regular_movie_effect, by = "movieId") %>%
  mutate(bin = ceiling((timestamp - min_timestamp) / bin_size)) %>%
  group_by(movieId, bin) %>%
  summarize(b_i_t = mean(rating - mu - b_i))

user_effect <- train_set %>%
  mutate(bin = ceiling((timestamp - min_timestamp) / bin_size)) %>%
  left_join(regular_movie_effect, by = "movieId") %>%
  left_join(binned_movie_effect, by = c("movieId", "bin")) %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i - coalesce(b_i_t, 0)))

# apply model to test set
user_effect_on_test_set <-
  test_set %>%
  mutate(bin = ceiling((timestamp - min_timestamp) / bin_size)) %>%
  left_join(regular_movie_effect, by = "movieId") %>%
  left_join(binned_movie_effect, by = c("movieId", "bin")) %>%
  left_join(user_effect, by = "userId") %>%
  mutate(model = mu + b_i + coalesce(b_i_t, 0) + b_u)

rmse_movie_and_user_effect <-
  RMSE(test_set$rating, user_effect_on_test_set$model)
print(
  paste(
    "RMSE of baseline + temporal movie effect and user effect: ",
    rmse_movie_and_user_effect
  )
)

# Now apply regularization to penalize predictions with few ratings,
# or users with few ratings. Lets find what the best lambda is:
lambdas <- seq(6, 8, 0.25)

regularized_rmses <- sapply(lambdas, function(lambda) {
  bin_size <- (max_timestamp - min_timestamp) / best_bin_num
  
  regular_movie_effect <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + lambda))
  
  binned_movie_effect <- train_set %>%
    left_join(regular_movie_effect, by = "movieId") %>%
    mutate(bin = ceiling((timestamp - min_timestamp) / bin_size)) %>%
    group_by(movieId, bin) %>%
    summarize(b_i_t = sum(rating - mu - b_i) / (n() + lambda))
  
  user_effect <- train_set %>%
    mutate(bin = ceiling((timestamp - min_timestamp) / bin_size)) %>%
    left_join(regular_movie_effect, by = "movieId") %>%
    left_join(binned_movie_effect, by = c("movieId", "bin")) %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu - b_i - coalesce(b_i_t, 0)) / (n() + lambda))
  
  regularization_on_test_set <-
    test_set %>%
    mutate(bin = ceiling((timestamp - min_timestamp) / bin_size)) %>%
    left_join(regular_movie_effect, by = "movieId") %>%
    left_join(binned_movie_effect, by = c("movieId", "bin")) %>%
    left_join(user_effect, by = "userId") %>%
    mutate(model = mu + b_i + coalesce(b_i_t, 0) + b_u)
  
  RMSE(test_set$rating, regularization_on_test_set$model)
})

best_lambda <- lambdas[which.min(regularized_rmses)]
print(paste(
  "RMSE of baseline + temporal movie effect and user effect + regularization: ",
  min(regularized_rmses)
))

# we are happy with current RMSE, so we now do a final evaluation with the validation set:
bin_size <- (max_timestamp - min_timestamp) / best_bin_num

regular_movie_effect <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + best_lambda))

binned_movie_effect <- train_set %>%
  left_join(regular_movie_effect, by = "movieId") %>%
  mutate(bin = ceiling((timestamp - min_timestamp) / bin_size)) %>%
  group_by(movieId, bin) %>%
  summarize(b_i_t = sum(rating - mu - b_i) / (n() + best_lambda))

user_effect <- train_set %>%
  mutate(bin = ceiling((timestamp - min_timestamp) / bin_size)) %>%
  left_join(regular_movie_effect, by = "movieId") %>%
  left_join(binned_movie_effect, by = c("movieId", "bin")) %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - mu - b_i - coalesce(b_i_t, 0)) / (n() + best_lambda))

# apply model to validation set
results_on_validation <-
  validation %>%
  mutate(bin = ceiling((timestamp - min_timestamp) / bin_size)) %>%
  left_join(regular_movie_effect, by = "movieId") %>%
  left_join(binned_movie_effect, by = c("movieId", "bin")) %>%
  left_join(user_effect, by = "userId") %>%
  # coalesce b_i since validation set includes some unknown movies.
  mutate(model = mu + coalesce(b_i, 0) + coalesce(b_i_t, 0) + b_u)

rmse_validation = RMSE(validation$rating, results_on_validation$model)
print(paste("RMSE of final model on validation set: ", rmse_validation))
