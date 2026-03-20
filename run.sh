pip install -r requirements.txt
python data/get_data.py


#!/bin/bash

echo "Training model..."
python src/train.py

echo "Evaluating model..."
python src/eval.py

echo "Running RL agent..."
python src/rl_agent.py

echo "Done."