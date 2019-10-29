# Logarithmic Reinforcement Learning

This repository hosts sample code for the NeurIPS 2019 paper: [van Seijen, Fatemi, Tavakoli (2019)][log_rl]. 

We provide code for the linear experiments of the paper as well as the deep RL Atari 2600 examples (LogDQN).

## [LICENSE](https://github.com/microsoft/logrl/blob/master/LICENSE)

## [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

The code for LogDQN has been developed by [Arash Tavakoli](https://atavakol.github.io/) and the code for the linear experiments has been developed by [Harm van Seijen](mailto:Harm.vanSeijen@microsoft.com). 

## Citing

If you use this research in your work, please cite the accompanying [paper][log_rl]:

```
@inproceedings{vanseijen2019logrl,
  title={Using a Logarithmic Mapping to Enable Lower Discount Factors in Reinforcement Learning},
  author={van Seijen, Harm and
          Fatemi, Mehdi and 
          Tavakoli, Arash},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```

---
## Linear Experiments

First navigate to [linear_experiments](https://github.com/microsoft/logrl/linear_experiments/) folder.

To create result-files: 
```
python main
```
To visualize result-files: 
```
python show_results
```

With the default settings (i.e., keeping [main.py](https://github.com/microsoft/logrl/linear_experiments/main.py) unchanged), a scan over different gamma values is performed for a tile-width of 2 for a version of Q-learning without a logarithmic mapping.

All experimental settings can be found at the top of the [main.py](https://github.com/microsoft/logrl/linear_experiments/main.py) file.
To run the logarithmic-mapping version of Q-learning, set:
```
agent_settings['log_mapping'] = True
```

Results of the full scans are provided. To visualize these results for regular Q-learning or logarithmic Q-learning, set `filename` in [show_results.py](https://github.com/microsoft/logrl/linear_experiments/show_results.py) to `full_scan_reg` or `full_scan_log`, respectively.

---
## Logarithmic Deep Q-Network (LogDQN) 

This part presents an implementation of LogDQN from [van Seijen, Fatemi, Tavakoli (2019)][log_rl].

### Instructions

Our implementation of LogDQN builds on Dopamine ([Castro et al., 2018][dopamine_paper]), a Tensorflow-based research framework for fast prototyping of reinforcement learning algorithms. 

Follow the instructions below to install the LogDQN package along with a compatible version of Dopamine and their dependencies inside a conda environment.

First install [Anaconda](https://docs.anaconda.com/anaconda/install/), and then proceed below.

```
conda create --name log-env python=3.6 
conda activate log-env
```

#### Ubuntu

```
sudo apt-get update && sudo apt-get install cmake zlib1g-dev
pip install absl-py atari-py gin-config gym opencv-python tensorflow==1.15rc3
pip install git+git://github.com/google/dopamine.git@a59d5d6c68b1a6e790d5808c550ae0f51d3e85ce
```

Finally, navigate to [log_dqn_experiments](https://github.com/microsoft/logrl/log_dqn_experiments) and install the LogDQN package from source.

```
cd log_dqn_experiments
pip install .
```

### Training an agent

To run a LogDQN agent,

```
python -um log_dqn.train_atari \
    --agent_name=log_dqn \
    --base_dir=/tmp/log_dqn \
    --gin_files='log_dqn/log_dqn.gin' \
    --gin_bindings="Runner.game_name = \"Asterix\"" \
    --gin_bindings="LogDQNAgent.tf_device=\"/gpu:0\""
```

You can set `LogDQNAgent.tf_device` to `/cpu:*` for a non-GPU version.



[log_rl]: https://arxiv.org/abs/1906.00572
[dopamine_paper]: https://arxiv.org/abs/1812.06110
[dqn]: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
