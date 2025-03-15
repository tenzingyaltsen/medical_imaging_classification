# Load packages.
library(keras)
library(ggplot2)

# Looking at images, we can see they are named from "BloodImage_00000" to "BloodImage_00410".
# Examine labels.
labels <- read.csv("labels.csv")
str(labels)
unique(labels$Category)
# We can see there are many categories, with many overlapping.
# Record indices of labels that are only "NEUTROPHIL" or "EOSINOPHIL".
neu_indices <- which(labels$Category == "NEUTROPHIL")
eos_indices <- which(labels$Category == "EOSINOPHIL")
# Adjust index by subtracting 1, as the image names start from 0 not 1.
neu_indices <- neu_indices - 1
eos_indices <- eos_indices - 1
# Check how much data is lost by focusing on only neutrophils and eosinophils.
length(labels$Category) - (length(neu_indices) + length(eos_indices))

# Class-wise splitting, assigning training (75%) and validation (25%) indices.
set.seed(123)
# First for neutrophils.
neu_train_indices <- sample(neu_indices, size = 0.75 * length(neu_indices))
neu_val_indices <- setdiff(neu_indices, neu_train_indices)
# Then for eosinophils.
eos_train_indices <- sample(eos_indices, size = 0.75 * length(eos_indices))
eos_val_indices <- setdiff(eos_indices, eos_train_indices)

# Now let's create the subdirectory tree using the provided code.
# Define the original data source folder.
original_dataset_dir <- "~/JPEGImages"
base_dir <- "~/Downloads/cats_and_dogs_small"
dir.create(base_dir)
train_dir <- file.path(base_dir, "train")
dir.create(train_dir)
validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)
test_dir <- file.path(base_dir, "test")
dir.create(test_dir)

train_cats_dir <- file.path(train_dir, "cats")
dir.create(train_cats_dir)
train_dogs_dir <- file.path(train_dir, "dogs")
dir.create(train_dogs_dir)
validation_cats_dir <- file.path(validation_dir, "cats")
dir.create(validation_cats_dir)
validation_dogs_dir <- file.path(validation_dir, "dogs")
dir.create(validation_dogs_dir)
test_cats_dir <- file.path(test_dir, "cats")
dir.create(test_cats_dir)
test_dogs_dir <- file.path(test_dir, "dogs")
dir.create(test_dogs_dir)

fnames <- paste0("cat.", 1:1000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(train_cats_dir))
fnames <- paste0("cat.", 1001:1500, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(validation_cats_dir))
fnames <- paste0("cat.", 1501:2000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(test_cats_dir))

fnames <- paste0("dog.", 1:1000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(train_dogs_dir))
fnames <- paste0("dog.", 1001:1500, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(validation_dogs_dir))
fnames <- paste0("dog.", 1501:2000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(test_dogs_dir))