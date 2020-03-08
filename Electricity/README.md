# Experiments on Electricity Dataset

This implementation of *DeepAR* is due to **Yunkai Zhang** and **Qiao Jiang** ([repository](https://github.com/zhykoties/TimeSeries)).
We added the files ``attack.py``, ``attack_plot.py``, and ``attack_utils.py`` to run our attack.

## Requirements
Create virtual environment and install requirements

        virtualenv -p python3.6 venv && source venv/bin/activate

        pip install -r requirements.txt

## Generate Dataset

Run

        python preprocess_elect.py

## Running the attack

    python attack.py --target -2 --tolerance 0.2 0.4 0.6 0.8 1 --c  0.1 0.2 0.3 0.5 0.7 1 2 3 5 7 10 20 30 50 70 100 200 300 --lr 0.01 --n_iterations 1000 --batch_c 6 --output_folder attack_results

This will save the logs in the **attack_logs/attack_results/** folder. The logs contain the adversarial samples. The parameter ``--batch_c`` controls the second batching dimension. The optimal value on a Nvidia RTX 2080 Ti GPU is 6, this can be adapted on other architectures.

## Loading and plotting generated adversarial samples

The file **attack_plot.py** contains an example code for obtaining and plotting adversarial samples.




