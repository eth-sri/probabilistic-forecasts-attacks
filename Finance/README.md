# Experiments on S&P500 historical stock prices

## Running the experiments

Create virtual environment and install requirements

        virtualenv -p python3.6 venv && source venv/bin/activate
        pip install -r requirements.txt

## Download dataset
1. (Optional) Update constituents file from [github.com/datasets/s-and-p-500-companies-financials](https://github.com/datasets/s-and-p-500-companies-financials),
and place it at ``datasets/raw_data/constituents.csv``.
2. Download raw data for constituents with the [yfinance API](https://github.com/ranaroussi/yfinance)

        python dataset/generation/data_downloader.py

(This might create exceptions for tickers that changed names, if the previous list is not updated. This won't impact the next steps).

3. Create training/validation/test sets for each period

       python dataset/generation/dataset.py
This will create a subfolder for each period in the ``dataset`` folder.

## Train models (Optional)

The following command will train all models, and save it in ``training/training_logs/Trained_Models``. Models used in the paper are in ``training/training_logs/Models``

    python training/training_loop.py --file training/exp_files/comparison.json


## Evaluation of the trained models

The following commands will evaluate the models, and save the results in ``eval/eval_logs``.

1. Long/Short trading strategies


         python eval/eval_loop.py --training_folder Models --steps 1 5 10 --output_folder Eval --samples 1 10 100 1000 10000 --days 85 --k 10 30 100 --run --summarize


2. Ranked probability skill

         python eval/eval_loop.py --training_folder Models --steps 1 5 10 --output_folder Eval_rps --samples 1 10 100 1000 10000 --n_bins 10 --days 85 --k 10 30 100 --rps --run --summarize


## Attack the trained models

The following commands will attack the models, and save the results in ``attack/attack_logs``. The argument ``--days`` controls the second batching dimension to speed up the attack. On a Nvidia RTX 2080 Ti GPU, the optimal value is 52, this should be decreased on a CPU or on a GPU with less memory.

1. Classification task, no Bayesian observation

         python attack/attack_loop.py --fixed_epsilon --target binary --training_folder Attack_Models --steps 10 --samples 10000 --n_iterations 1000 --batch_size 50 --max_pert 0.001 0.00158 0.00251 0.00398 0.00631 0.01 0.0158 0.0251 0.0398 0.0631 0.1 0.158 0.251 0.398 0.631 1 --learning_rate 0.001 --c 0.01 0.1 1 10 100 1000 10000 100000 --days 52 --output_folder normal_attack --run --summarize

2. Classification task, Bayesian observation

        python attack/attack_loop.py --fixed_epsilon --target binary --conditional --training_folder Attack_Models  --step_prediction 5 --step_condition 10 --value_condition 1.0008 --samples 10000 --n_iterations 1000 --batch_size 50 --max_pert 0.001 0.00158 0.00251 0.00398 0.00631 0.01 0.0158 0.0251 0.0398 0.0631 0.1 0.158 0.251 0.398 0.631 1 --learning_rate 0.001 --c 0.01 0.1 1 10 100 1000 10000 100000 1000000 --days 52 --output_folder conditional_attack --run --summarize

3. Trading task

        python attack/attack_loop.py --fixed_epsilon --target regression --k 10 30 100 --training_folder Attack_Models --steps 10 --samples 10000 --n_iterations 1000 --batch_size 50 --max_pert 0.001 0.00158 0.00251 0.00398 0.00631 0.01 0.0158 0.0251 0.0398 0.0631 0.1 0.158 0.251 0.398 0.631 1 --learning_rate 0.001 --c 0.01 0.1 1 10 100 1000 10000 100000 --days 50 --output_folder normal_attack_regression --run --summarize

## Experimental results 

![figure](https://github.com/eth-sri/probabilistic-forecasts-attacks/edit/master/Finance/figure.jpg?raw=true)

## Notes

Our architecture is denominated as **MDN** in the code.
