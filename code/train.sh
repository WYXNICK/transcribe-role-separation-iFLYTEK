#!/bin/bash

# Set environment variables
# The main script uses relative paths, so we run it from the 'code' directory.

# Path to the training data directory
# This corresponds to the --eval-dir argument in main.py
TRAIN_DATA_DIR="../xfdata/eval"

# Path for the main script to save processed data and models
# This corresponds to the directories configured inside main.py, like ../user_data

# Step 1: Preprocess the data
# The 'train' mode depends on preprocessed data. Let's run preprocessing first.
echo "Running data preprocessing..."
python main.py --mode preprocess --eval-dir $TRAIN_DATA_DIR

# Step 2: Run the training
# This will execute the training process defined in main.py
echo "Running model training..."
python main.py --mode train --eval-dir $TRAIN_DATA_DIR

echo "Training script finished."