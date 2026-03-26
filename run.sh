#!/bin/bash

set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Downloading dataset..."
python data/get_data.py

echo "Training main model..."
python -m src.train

echo "Evaluating main model..."
python -m src.eval

echo "Running RL threshold optimization..."
python -m src.rl_agent

echo "Pipeline completed successfully."


#============================================================
#For running with ablation results:
#!/bin/bash

# set -e

# echo "Installing dependencies..."
# pip install -r requirements.txt

# echo "Downloading dataset..."
# python data/get_data.py

# echo "Training baseline model..."
# python -m src.train

# echo "Evaluating baseline model..."
# python -m src.eval

# echo "Running ablation: no augmentation..."
# python -m src.ablations.train_no_augmentation
# python -m src.ablations.eval_no_augmentation

# echo "Running ablation: no class weights..."
# python -m src.ablations.train_no_class_weights
# python -m src.ablations.eval_no_class_weights

# echo "Running RL threshold optimization..."
# python -m src.rl_agent

# echo "Pipeline completed successfully."
#============================================================