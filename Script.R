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
# These indices need to be converted into a specific format that matches image names.
neu_train_indices <- sprintf("%05d", neu_train_indices)
neu_val_indices <- sprintf("%05d", neu_val_indices)
eos_train_indices <- sprintf("%05d", eos_train_indices)
eos_val_indices <- sprintf("%05d", eos_val_indices)

# Now let's create the subdirectory tree using the provided code.
# Define the original data source folder.
original_dataset_dir <- file.path(getwd(), "JPEGImages")
# Define and create the new data source folder.
base_dir <- file.path(getwd(),"neutrophils_and_eosinophils")
dir.create(base_dir)
# Define and create the training folder (within new data source folder).
train_dir <- file.path(base_dir, "train")
dir.create(train_dir)
# Define and create the validation folder (within new data source folder).
validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)
# No need for test folder as per instructions.

# Create folder for neutrophils within training folder.
train_neu_dir <- file.path(train_dir, "neutrophil")
dir.create(train_neu_dir)
# Create folder for eosinophils within training folder.
train_eos_dir <- file.path(train_dir, "eosinophil")
dir.create(train_eos_dir)
# Create folder for neutrophils within validation folder.
validation_neu_dir <- file.path(validation_dir, "neutrophil")
dir.create(validation_neu_dir)
# Create folder for eosinophils within validation folder.
validation_eos_dir <- file.path(validation_dir, "eosinophil")
dir.create(validation_eos_dir)

# Copy over the training images to their respective folders.
fnames <- paste0("BloodImage_", neu_train_indices, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(train_neu_dir))
# Note that "BloodImage_00280.jpg" does not exist (manually checked). Oh well!
fnames <- paste0("BloodImage_", eos_train_indices, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(train_eos_dir))
# Copy over the validation images to their respective folders.
fnames <- paste0("BloodImage_", neu_val_indices, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(validation_neu_dir))
fnames <- paste0("BloodImage_", eos_val_indices, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(validation_eos_dir))

