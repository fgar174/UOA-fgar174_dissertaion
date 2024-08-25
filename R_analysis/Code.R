###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################

library(dplyr)
# Load datasets
yard_moves_by_year.df_base <-
  read.csv("yard_moves_for_year.csv", header = TRUE)
yard_location_mapping.df <-
  read.csv("yard_location_mapping.csv", header = TRUE)
stack_row_block.df <- read.csv("stack_row_block.csv", header = TRUE)

# Turns columns in chr into dates
yard_moves_by_year.df_base$time_out <- as.POSIXct(yard_moves_by_year.df_base$time_out, format = "%Y-%m-%d %H:%M:%S")
yard_moves_by_year.df_base$time_in <-
  as.POSIXct(yard_moves_by_year.df_base$time_in, format = "%Y-%m-%d %H:%M:%S")


# Split time_in data column into day, month, year, and weekday
yard_moves_by_year.df_base$time_in_day <-
  as.integer(format(yard_moves_by_year.df_base$time_in, "%d"))
yard_moves_by_year.df_base$time_in_month <-
  as.integer(format(yard_moves_by_year.df_base$time_in, "%m"))
yard_moves_by_year.df_base$time_in_year <-
  as.integer(format(yard_moves_by_year.df_base$time_in, "%Y"))
yard_moves_by_year.df_base$time_in_weekday <-
  as.integer(strftime(yard_moves_by_year.df_base$time_in, "%u"))
yard_moves_by_year.df_base$time_in_hour <- as.numeric(format(yard_moves_by_year.df_base$time_in, "%H"))

yard_moves_by_year.df_base$time_in_week_of_month <- (as.integer(format(yard_moves_by_year.df_base$time_in, "%d")) - 1) %/% 7 + 1
yard_moves_by_year.df_base$time_in_week_of_year <- as.integer(strftime(yard_moves_by_year.df_base$time_in, "%V"))
yard_moves_by_year.df_base$time_in_business_day = ifelse(yard_moves_by_year.df_base$time_in_weekday >= 1 & yard_moves_by_year.df_base$time_in_weekday <= 5, 1, 0)



# Split time_out data column into day, month, year, and weekday
yard_moves_by_year.df_base$time_out_day <-
  as.integer(format(yard_moves_by_year.df_base$time_out, "%d"))
yard_moves_by_year.df_base$time_out_month <-
  as.integer(format(yard_moves_by_year.df_base$time_out, "%m"))
yard_moves_by_year.df_base$time_out_year <-
  as.integer(format(yard_moves_by_year.df_base$time_out, "%Y"))
yard_moves_by_year.df_base$time_out_weekday <-
  as.integer(strftime(yard_moves_by_year.df_base$time_out, "%u"))


# TO FILTER SPECIFIC CONTAINER
# yard_moves_by_year.df <-
#   yard_moves_by_year.df_base[yard_moves_by_year.df_base$ufv_gkey == 9490315, ]

yard_moves_by_year.df <- yard_moves_by_year.df_base

yard_moves_by_year.df$year_ <-
  as.integer(substr(yard_moves_by_year.df$period, 1, 4))
yard_moves_by_year.df$month_ <-
  as.integer(substr(yard_moves_by_year.df$period, 6, 7))
yard_moves_by_year.df$day_ <-
  as.integer(substr(yard_moves_by_year.df$period, 9, 10))

yard_moves_by_year.df[yard_moves_by_year.df$mve_gkey == "NA", ]


# Remove useless columns

# Cleaning some data based on MANVI's comments on  24 APR 2024
## ufv_gkey: Feel free to disregard this gkey for your analysis, its an internal key we use to join across tables.
yard_moves_by_year.df <-
  subset(
    yard_moves_by_year.df,
    select = -c(
      che_fetch_login_name,
      che_carry_login_name,
      che_put_login_name
    )
  )

## rehandle_count if NA will be 0
yard_moves_by_year.df$rehandle_count[is.na(yard_moves_by_year.df$rehandle_count)] <-
  0

## twin_fetch and twin_carry if NA will be 0
yard_moves_by_year.df$twin_carry[is.na(yard_moves_by_year.df$twin_carry)] <-
  0
yard_moves_by_year.df$twin_fetch[is.na(yard_moves_by_year.df$twin_fetch)] <-
  0

# Cleaning some data based on MANVI's comments on  26 APR 2024
yard_moves_by_year.df$dist_start[is.na(yard_moves_by_year.df$dist_start)] <-
  0
yard_moves_by_year.df$dist_carry[is.na(yard_moves_by_year.df$dist_carry)] <-
  0
yard_moves_by_year.df$twin_put[is.na(yard_moves_by_year.df$twin_put)] <-
  0


cols_with_na <-
  sapply(yard_moves_by_year.df, function(x)
    any(is.na(x)))
columns_with_na <- names(cols_with_na[cols_with_na])



yard_moves_by_year.df$to_pos_name_N <-
  paste(strsplit(as.character(yard_moves_by_year.df$to_pos_name), split = " "), sep = '-')

yard_moves_by_year.df$to_pos_name_N <-
  gsub("\\(", "", yard_moves_by_year.df$to_pos_name_N)
yard_moves_by_year.df$to_pos_name_N <-
  gsub("\\)", "", yard_moves_by_year.df$to_pos_name_N)
yard_moves_by_year.df$to_pos_name_N <-
  gsub(" ", "-", yard_moves_by_year.df$to_pos_name_N)

yard_moves_by_year.df$fm_pos_name_N <-
  as.character(gsub("\\(", "", yard_moves_by_year.df$fm_pos_name))
yard_moves_by_year.df$fm_pos_name_N <-
  as.character(gsub("\\)", "", yard_moves_by_year.df$fm_pos_name_N))
yard_moves_by_year.df$fm_pos_name_N <-
  as.character(gsub(" ", "-", yard_moves_by_year.df$fm_pos_name_N))


yard_moves_by_year.df$fm_pos_stack <-
  sapply(strsplit(as.character(yard_moves_by_year.df$fm_pos_name_N), split = "-"), function(x)
    if (length(x) >= 3)
      x[3]
    else
      NA)

yard_moves_by_year.df$to_pos_stack <-
  sapply(strsplit(as.character(yard_moves_by_year.df$to_pos_name_N), split = "-"), function(x)
    if (length(x) >= 3)
      x[3]
    else
      NA)


yard_moves_by_year.df$to_pos_stack <-
  gsub("\\.", "", yard_moves_by_year.df$to_pos_stack)
yard_moves_by_year.df$fm_pos_stack <-
  gsub("\\.", "", yard_moves_by_year.df$fm_pos_stack)

yard_moves_by_year.df$to_pos_stack_level <-
  substr(
    yard_moves_by_year.df$to_pos_stack,
    nchar(yard_moves_by_year.df$to_pos_stack),
    nchar(yard_moves_by_year.df$to_pos_stack)
  )
yard_moves_by_year.df$to_pos_stack <-
  substr(
    yard_moves_by_year.df$to_pos_stack,
    1,
    nchar(yard_moves_by_year.df$to_pos_stack) - 1
  )

yard_moves_by_year.df$fm_pos_stack_level <-
  substr(
    yard_moves_by_year.df$fm_pos_stack,
    nchar(yard_moves_by_year.df$fm_pos_stack),
    nchar(yard_moves_by_year.df$fm_pos_stack)
  )
yard_moves_by_year.df$fm_pos_stack <-
  substr(
    yard_moves_by_year.df$fm_pos_stack,
    1,
    nchar(yard_moves_by_year.df$fm_pos_stack) - 1
  )

# JOIN stack row block information for to (where it was moved to)

yard_moves_by_year.df <-
  left_join(yard_moves_by_year.df,
            stack_row_block.df,
            by = c("to_pos_stack" = "stack"))

yard_moves_by_year.df <- rename(yard_moves_by_year.df,
                                to_pos_block_row = row,
                                to_pos_block = block)

# JOIN stack row block information for from (where it was moved from)

yard_moves_by_year.df <-
  left_join(yard_moves_by_year.df,
            stack_row_block.df,
            by = c("fm_pos_stack" = "stack"))

yard_moves_by_year.df <- rename(yard_moves_by_year.df,
                                fm_pos_block_row = row,
                                fm_pos_block = block)

yard_moves_by_year.df <-
  left_join(yard_moves_by_year.df,
            yard_location_mapping.df,
            by = c("to_pos_block" = "Block"))

yard_moves_by_year.df <- rename(
  yard_moves_by_year.df,
  to_pos_yard = Yard,
  to_pos_yard_area = YardArea,
  to_pos_yard_location = YardLocation,
  to_pos_yard_teu = teu,
  to_pos_yard_max_buffer = max_stack_buffer,
  to_pos_yard_productivity_area = YardProductivityArea,
)

yard_moves_by_year.df <-
  left_join(yard_moves_by_year.df,
            yard_location_mapping.df,
            by = c("fm_pos_block" = "Block"))

yard_moves_by_year.df <- rename(
  yard_moves_by_year.df,
  fm_pos_yard = Yard,
  fm_pos_yard_area = YardArea,
  fm_pos_yard_location = YardLocation,
  fm_pos_yard_teu = teu,
  fm_pos_yard_max_buffer = max_stack_buffer,
  fm_pos_yard_productivity_area = YardProductivityArea,
)

yard_moves_by_year.df$fm_pos_stack_level <-
  ifelse(
    is.na(yard_moves_by_year.df$fm_pos_block),
    NA,
    as.numeric(yard_moves_by_year.df$fm_pos_stack_level)
  )

yard_moves_by_year.df$to_pos_stack_level <-
  ifelse(
    is.na(yard_moves_by_year.df$to_pos_block),
    NA,
    as.numeric(yard_moves_by_year.df$to_pos_stack_level)
  )

yard_moves_by_year.df$DAYS_IN_PORT <-
  as.numeric(round(
    difftime(
      yard_moves_by_year.df$time_out,
      yard_moves_by_year.df$time_in,
      units = "days"
    )
  ))

yard_moves_by_year.df$DAYS_IN_PORT <-
  as.numeric(yard_moves_by_year.df$DAYS_IN_PORT)

# Create categories for DAYS_IN_PORT
yard_moves_by_year.df$DAYS_IN_PORT_CATEGORY <- cut(
  yard_moves_by_year.df$DAYS_IN_PORT,
  breaks = c(1, 3, 6),
  labels = c("IN 3", "IN 6"),
  right = FALSE  # Ensure the intervals are [1, 7), [7, 10), and [10, 15)
)


######### GRAPHS ##############################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################


yard_moves_by_year_by_ufv_gkey <-
  subset(
    yard_moves_by_year.df,
    select = c(
      ufv_gkey,
      time_in,
      time_out,
      freight_kind,
      time_in_weekday,
      time_out_weekday,
      time_out_day,
      category
    )
  )[!is.na(yard_moves_by_year.df$fm_pos_block), ]

# yard_moves_by_year_by_ufv_gkey %>%
#   group_by(ufv_gkey) %>%
#   summarize(num_categories = n_distinct(category)) %>%
#   filter(num_categories > 1)

# Calculate the days that the container was kept in the port
yard_moves_by_year_by_ufv_gkey$DAYS_IN_PORT <-
  as.numeric(round(
    difftime(
      yard_moves_by_year_by_ufv_gkey$time_out,
      yard_moves_by_year_by_ufv_gkey$time_in,
      units = "days"
    )
  ))

yard_moves_by_year_by_ufv_gkey$DAYS_IN_PORT <-
  as.numeric(yard_moves_by_year_by_ufv_gkey$DAYS_IN_PORT)


days_by_ufv_gkey <-
  aggregate(DAYS_IN_PORT ~ ufv_gkey + freight_kind + category, data = yard_moves_by_year_by_ufv_gkey, min)

boxplot(
  days_by_ufv_gkey$DAYS_IN_PORT,
  main = "Distribution of days that a container is retrieved from the port",
  ylab = "Days",
  xlab = "Days Distribution"
)

boxplot(
  days_by_ufv_gkey[days_by_ufv_gkey$DAYS_IN_PORT > 15, ]$DAYS_IN_PORT,
  main = "Containers retrieved in more than 15 Days",
  ylab = "Days",
  xlab = "Days Distribution"
)

boxplot(
  days_by_ufv_gkey[days_by_ufv_gkey$DAYS_IN_PORT <= 15, ]$DAYS_IN_PORT,
  main = "Containers retrieved in the first 14 Days",
  ylab = "Days",
  xlab = "Days Distribution",
  notch = TRUE
)
axis(2, at = 1:15, labels = 1:15)

# We can check that the most of the containers are retrieved from 1 to 5 days
# after entering into the port

par(mar = c(10, 4, 4, 2) + 0.1)
boxplot(
  DAYS_IN_PORT ~ freight_kind + category,
  data = days_by_ufv_gkey[days_by_ufv_gkey$DAYS_IN_PORT < 25, ],
  main = "Box Plot of Days by Freight Kind and Category",
  xlab = "Freight Kind and Category",
  ylab = "Days",
  las = 2,
  # Rotate x-axis labels for better readability
  col = "lightblue",
  # Color for the box plots
  notch = TRUE,
  # Add notches to the box plots to compare medians
  outline = TRUE,
  # Remove outliers from the box plots
  ylim = c(1, 25),
  yaxt = "n"
)
axis(2, at = 1:25, labels = 1:25)

table(days_by_ufv_gkey$ufv_gkey, days_by_ufv_gkey$category)

# Containers that are kind EMPTY and category IMPORT are removed from the yard in
# 1 or 3 days

# Plot the distribution of time_out_weekday
barplot(
  table(yard_moves_by_year_by_ufv_gkey$time_out_weekday),
  main = "Distribution of time_out_weekday",
  xlab = "Day of the Week",
  ylab = "Frequency",
  col = "lightblue",
  las = 2
)

# Plot the distribution of time_in_weekday
barplot(
  table(yard_moves_by_year_by_ufv_gkey$time_in_weekday),
  main = "Distribution of time_in_weekday",
  xlab = "Day of the Week",
  ylab = "Frequency",
  col = "lightblue",
  las = 2
)


library(ggplot2)

# Filter the dataset to include only rows where DAYS_IN_PORT is less than 15
filtered_data <-
  days_by_ufv_gkey[days_by_ufv_gkey$DAYS_IN_PORT < 25, ]

# Create a summary dataframe with the mean DAYS_IN_PORT for each combination of freight_kind and category
summary_data <-
  aggregate(DAYS_IN_PORT ~ freight_kind + category,
            data = filtered_data,
            FUN = mean)

# Create the heatmap using ggplot2
ggplot(summary_data,
       aes(x = freight_kind, y = category, fill = DAYS_IN_PORT)) +
  geom_tile() +
  scale_fill_gradient(low = "lightblue",
                      high = "darkblue",
                      name = "Avg Days in Port") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +  # Rotate x-axis labels for better readability
  labs(title = "Heatmap of Average Days in Port by Freight Kind and Category", x = "Freight Kind", y = "Category")

###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################

str(yard_moves_by_year.df)

yard_moves_by_year_unique <- yard_moves_by_year.df[!is.na(yard_moves_by_year.df$fm_pos_yard), ] %>%
  distinct(ufv_gkey, 
           arrive_pos_loctype, 
           requires_power, 
           # fm_pos_yard, 
           time_in, 
           time_in_year, 
           time_in_weekday, 
           time_in_business_day, 
           time_in_week_of_year, 
           time_in_week_of_month, 
           time_in_month, 
           time_in_hour, 
           freight_kind, 
           category, 
           nominal_length, 
           DAYS_IN_PORT_CATEGORY, 
           DAYS_IN_PORT)

ufv_gkey_counts <- table(yard_moves_by_year_unique$ufv_gkey)

ufv_gkey_counts[ufv_gkey_counts > 1]

yard_moves_by_year_unique[yard_moves_by_year_unique$ufv_gkey==8955429,]

subset_df <- subset(
  yard_moves_by_year_unique,
  select = c(
    arrive_pos_loctype,
    requires_power,
    # fm_pos_yard,
    time_in,
    time_in_year,
    time_in_weekday,
    time_in_business_day,
    time_in_week_of_year,
    time_in_week_of_month,
    time_in_month,
    time_in_hour,
    freight_kind,
    category,
    nominal_length,
    DAYS_IN_PORT_CATEGORY,
    DAYS_IN_PORT
  )
)

subset_df <- subset_df[subset_df$time_in_year > 2022, ]
subset_df <- subset_df[!is.na(subset_df$time_in_year), ]

subset_df$FOR_TEST <- FALSE

subset_df[subset_df$time_in > as.POSIXct("2023-12-31 23:59:59"), ]$FOR_TEST <- TRUE

write.csv(subset_df,
          "../python_modeling/dataset/subset_base.csv",
          row.names = TRUE)


min(subset_df$time_in)
max(subset_df$time_in)



# NA 2016 2015

###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################

yard_moves_by_year_by_ufv_gkey_less_15 <-
  days_by_ufv_gkey[days_by_ufv_gkey$DAYS_IN_PORT < 15, ]


boxplot(
  DAYS_IN_PORT ~ freight_kind + category,
  data = yard_moves_by_year_by_ufv_gkey_less_15,
  main = "Box Plot of Days by Freight Kind and Category",
  xlab = "Freight Kind and Category",
  ylab = "Days",
  las = 2,
  # Rotate x-axis labels for better readability
  col = "lightblue",
  # Color for the box plots
  notch = TRUE,
  # Add notches to the box plots to compare medians
  outline = TRUE  # Remove outliers from the box plots
)

# Add mean points to the box plot
means <-
  tapply(
    yard_moves_by_year_by_ufv_gkey$days,
    list(
      yard_moves_by_year_by_ufv_gkey$freight_kind,
      yard_moves_by_year_by_ufv_gkey$category
    ),
    mean
  )
points(
  rep(1:length(means), each = length(means[[1]])),
  unlist(means),
  col = "red",
  pch = 18,
  cex = 1.5
)








summary(days_by_ufv_gkey[days_by_ufv_gkey$DAYS_IN_PORT < 15, ]$DAYS_IN_PORT)

nrow(days_by_ufv_gkey[days_by_ufv_gkey$DAYS_IN_PORT < 1, ])
nrow(days_by_ufv_gkey[days_by_ufv_gkey$DAYS_IN_PORT > 0, ])

yard_moves_by_year.df_base[yard_moves_by_year.df_base$ufv_gkey == 8942321, ]






###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################


subset(
  yard_moves_by_year.df,
  select = c(
    category,
    inv_unit_id,
    ufv_gkey,
    move_kind,
    fm_pos_block,
    fm_pos_stack,
    fm_pos_stack_level,
    fm_pos_yard,
    to_pos_block,
    to_pos_stack,
    to_pos_stack_level,
    to_pos_yard,
    fm_pos_yard_location,
    to_pos_yard_location,
    freight_kind,
    arrive_pos_loctype,
    last_pos_loctype
  )
)[yard_moves_by_year.df$category == 'THRGH', ]




###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################


tempo_base <- unique(subset(
  yard_moves_by_year.df,
  select = c(
    ufv_gkey,
    time_in_weekday,
    time_in_month,
    freight_kind,
    category,
    nominal_length,
    DAYS_IN_PORT
  )
))

tempo_base$category <- as.factor(tempo_base$category)
tempo_base$freight_kind <- as.factor(tempo_base$freight_kind)
tempo_base$nominal_length <- as.factor(tempo_base$nominal_length)

set.seed(123)  # Set seed for reproducibility
sample <-
  sample.int(
    n = nrow(tempo_base),
    size = floor(.75 * nrow(tempo_base)),
    replace = F
  )
train <- tempo_base[sample, ]
test <- tempo_base[-sample, ]

model <-
  lm(
    DAYS_IN_PORT ~ time_in_weekday + time_in_month + freight_kind + category + nominal_length,
    data = train
  )

# Summarize the model to view coefficients and statistics
summary(model)

# Predict on the test set
predictions <- round(predict(model, newdata = test))

data.frame(
  days_in_port = test$DAYS_IN_PORT,
  predictions = predictions,
  diff_days = (predictions - test$DAYS_IN_PORT)
)

######################################

library(randomForest)

# Convert factors to numeric if necessary (example transformation)
tempo_base$time_in_weekday <-
  as.numeric(as.factor(tempo_base$time_in_weekday))
tempo_base$time_in_month <-
  as.numeric(as.factor(tempo_base$time_in_month))
tempo_base$freight_kind <-
  as.numeric(as.factor(tempo_base$freight_kind))
tempo_base$category <- as.numeric(as.factor(tempo_base$category))
tempo_base$nominal_length <-
  as.numeric(as.factor(tempo_base$nominal_length))

# Split data into training and test sets
set.seed(123)  # For reproducibility
train_indices <-
  sample(1:nrow(tempo_base), size = 0.7 * nrow(tempo_base))
train_data <- tempo_base[train_indices, ]
test_data <- tempo_base[-train_indices, ]
test_data <- test_data[!is.na(test_data$DAYS_IN_PORT)]


# Train a Random Forest model
library(randomForest)
model <-
  randomForest(
    DAYS_IN_PORT ~ time_in_weekday + time_in_month + freight_kind + category + nominal_length,
    data = train_data,
    ntree = 100,
    na.action = na.omit
  )


predictions <- floor(predict(model, test_data))




data.frame(
  days_in_port = test_data$DAYS_IN_PORT,
  predictions = predictions,
  diff_days = (predictions - test_data$DAYS_IN_PORT)
)




######################################





str(tempo_base)


boxplot(
  DAYS_IN_PORT ~ freight_kind + time_in_weekday + category,
  data = tempo_base,
  main = "Box Plot of Days by Freight Kind and Category",
  xlab = "Freight Kind and Category",
  ylab = "Days",
  las = 2,
  # Rotate x-axis labels for better readability
  col = "lightblue",
  # Color for the box plots
  notch = TRUE,
  # Add notches to the box plots to compare medians
  outline = FALSE  # Remove outliers from the box plots
)




###############################################################################################################



subset(
  yard_moves_by_year.df,
  select = c(
    inv_unit_id,
    ufv_gkey,
    move_kind,
    fm_pos_block,
    fm_pos_stack,
    fm_pos_stack_level,
    fm_pos_yard,
    to_pos_block,
    to_pos_stack,
    to_pos_stack_level,
    to_pos_yard,
    fm_pos_yard_location,
    to_pos_yard_location,
    freight_kind,
    arrive_pos_loctype,
    last_pos_loctype,
    time_in_weekday,
    time_in_month,
    freight_kind,
    category,
    nominal_length,
    DAYS_IN_PORT
  )
)

subset_df <- subset(
  yard_moves_by_year.df,
  select = c(
    time_in_weekday,
    time_in_month,
    freight_kind,
    category,
    nominal_length,
    DAYS_IN_PORT
  )
)



write.csv(subset_df,
          "python_modeling/dataset/subset_base.csv",
          row.names = TRUE)

str(subset_df)


#
#
#
#
#
#
#
# head(yard_moves_by_year.df)
# head(stack_row_block.df)
# head(yard_location_mapping.df)
#
# unique(yard_moves_by_year.df$to_pos_name)
#
#
#
# for (stack_name in stack_row_block.df$stack) {
#   # Find rows in yard_moves_by_year.df where to_pos_name contains stack_name
#   matching_rows <-
#     grepl(stack_name, yard_moves_by_year.df$to_pos_name)
#
#   # Update the stack column in yard_moves_by_year.df for matching rows
#   yard_moves_by_year.df$stack[matching_rows] <- stack_name
# }
#
#
# table(yard_moves_by_year.df$move_kind)
#
# # SHFT and YARD
#
#
#
#
#
#
#
# min(nchar(stack_row_block.df$stack))
# max(nchar(stack_row_block.df$stack))
#
# merge(yard_moves_by_year.df, yard_location_mapping.df, by = "id")
#
#
#
# cols_with_na <-
#   sapply(yard_moves_by_year.df, function(x)
#     any(is.na(x)))
# columns_with_na <- names(cols_with_na[cols_with_na])
#
# sorted_counts <-
#   sort(table(yard_moves_by_year.df[yard_moves_by_year.df$freight_kind == 'FCL',]$ufv_gkey), decreasing = TRUE)
# head(sorted_counts, 10)
#
# # movements_by_container <-
# # yard_moves_by_year.df[yard_moves_by_year.df$ufv_gkey == 9490315,]
#
# movements_by_container <- yard_moves_by_year.df
#
# movements_by_container <-
#   movements_by_container[order(movements_by_container$ufv_gkey, decreasing = TRUE),]
#
# subset(
#   movements_by_container,
#   select = c(
#     inv_unit_id,
#     ufv_gkey,
#     move_kind,
#     freight_kind,
#     nominal_length,
#     arrive_pos_loctype,
#     last_pos_loctype,
#     category,
#     dist_start,
#     dist_carry,
#     period,
#     t_dispatch,
#     t_carry_dispatch,
#     t_discharge,
#     t_fetch,
#     t_put,
#     time_in,
#     time_out
#   )
# )
#
# movements_by_container$CALCULATED_DELTA_DIST <-
#   movements_by_container$dist_carry - movements_by_container$dist_start
#
# movements_by_container$time_out <-
#   as.Date(movements_by_container$time_out, format = "%Y-%m-%d %H:%M:%S")
# movements_by_container$time_in <-
#   as.Date(movements_by_container$time_in, format = "%Y-%m-%d %H:%M:%S")
#
#
#
#
# movements_by_container <-
#   movements_by_container[!is.na(movements_by_container$time_out), ]
# movements_by_container <-
#   movements_by_container[!is.na(movements_by_container$time_in), ]
#
# movements_by_container_by_ufv_gkey <- subset(movements_by_container,
#                                              select = c(ufv_gkey,
#                                                         time_in,
#                                                         time_out))
#
# movements_by_container_by_ufv_gkey$DAYS_IN_PORT <-
#   as.numeric(round(
#     difftime(
#       movements_by_container_by_ufv_gkey$time_out,
#       movements_by_container_by_ufv_gkey$time_in,
#       units = "days"
#     )
#   ))
#
# movements_by_container_by_ufv_gkey$DAYS_IN_PORT <-
#   as.numeric(movements_by_container_by_ufv_gkey$DAYS_IN_PORT)
#
#
#
# # movements_by_container$MIN_MOVE <-
# #   round(
# #     difftime(
# #       movements_by_container$t_put,
# #       movements_by_container$t_dispatch,
# #       units = "mins"
# #     )
# #   )
#
# subset(
#   movements_by_container,
#   select = c(
#     time_in,
#     move_kind,
#     mve_gkey,
#     t_dispatch,
#     category,
#     dist_start,
#     dist_carry,
#     CALCULATED_DELTA_DIST,
#     freight_kind,
#     goods_and_ctr_wt_kg,
#     arrive_pos_loctype,
#     last_pos_loctype,
#     time_out,
#     DAYS_IN_PORT
#   )
# )
#
# # library(openxlsx)
# # write.xlsx(movements_by_container, "DataAnalysis.xlsx")
#
#
# all(
#   aggregate(DAYS_IN_PORT ~ ufv_gkey,
#             data = movements_by_container_by_ufv_gkey,
#             max) == aggregate(DAYS_IN_PORT ~ ufv_gkey,
#                               data = movements_by_container_by_ufv_gkey,
#                               min)
# )
#
# days_by_ufv_gkey <- aggregate(DAYS_IN_PORT ~ ufv_gkey,
#                               data = movements_by_container_by_ufv_gkey,
#                               min)
#
# summary(days_by_ufv_gkey)
#
#
# nrow(days_by_ufv_gkey)
# nrow(days_by_ufv_gkey[days_by_ufv_gkey$DAYS_IN_PORT < 15, ])
#
# summary(days_by_ufv_gkey[days_by_ufv_gkey$DAYS_IN_PORT < 15, ])
#
#
# boxplot(
#   days_by_ufv_gkey[days_by_ufv_gkey$DAYS_IN_PORT < 15, ]$DAYS_IN_PORT,
#   main = "Containers retrieved in the first 14 Days",
#   ylab = "Days",
#   xlab = "Days Distribution"
# )
#
#
# summary(days_by_ufv_gkey[days_by_ufv_gkey$DAYS_IN_PORT > 14, ])
#
#
#
#
#
# boxplot(
#   days_by_ufv_gkey[days_by_ufv_gkey$DAYS_IN_PORT > 14, ]$DAYS_IN_PORT,
#   main = "Containers retrieved in the first 14 Days",
#   ylab = "Days",
#   xlab = "Days Distribution"
# )
#
# summary(days_by_ufv_gkey[days_by_ufv_gkey$DAYS_IN_PORT > 23, ])
#
#
# nrow(days_by_ufv_gkey)
#
# nrow(days_by_ufv_gkey[days_by_ufv_gkey$DAYS_IN_PORT > 5, ])






##### https://catalogue.data.govt.nz/dataset/road-events2/resource/31dc213e-7cde-48c8-a29a-b339a6506a75
# The Traffic Road Event Information System (TREIS) API lets you access real time data on 
# events / incidents that affect traffic conditions across the network of national highways in New...




road_events <- read.csv("Road_Events.csv", header = TRUE)

road_events$endDate <- as.Date(road_events$endDate, format = "%Y/%m/%d %H:%M:%S")
road_events$startDate <- as.Date(road_events$startDate, format = "%Y/%m/%d %H:%M:%S")

str(road_events)

unique(road_events$eventType)
unique(road_events$eventIsland)
unique(road_events$restrictions)
unique(road_events$impact)
unique(road_events$eventDescription)
