#!/bin/bash

# Set environment variables
# The main script uses relative paths, so we run it from the 'code' directory.

# Path to the test data directory
# This corresponds to the --test-dir argument in main.py
TEST_DATA_DIR="../xfdata/test_data"

# Path for the prediction results
# This corresponds to the --output-dir argument in main.py
PREDICTION_RESULT_DIR="../prediction_result"

# Run inference using the main script
# This will execute the inference pipeline defined in main.py
echo "Running inference..."
CUDA_VISIBLE_DEVICES=1 python main.py --mode inference --test-dir $TEST_DATA_DIR --output-dir $PREDICTION_RESULT_DIR

echo "Inference script finished. Results are in $PREDICTION_RESULT_DIR"