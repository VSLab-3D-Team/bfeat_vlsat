
Make sure you set the proper path in "config.py" and project name for wandb in "train_mv_bfeat_ri.py"

### Data Preparation

```bash
python3 preprocess_ply.py
```

### Training Script

```bash
python3 train_mv_bfeat_ri.py --exp_name <experiment name for wandb>
```

### Experiments Script

Experiments for simple MLP classifier with Object Feature Encoder

1. Make sure you set the configuration path in "classifier.py" and "eval_feat_discriminative.py" in experiment directory

2. Run 'eval_feat_discriminative.py' with command below.

```bash
python3 -m experiment.eval_feat_discriminative --exp_dir <Your Path>
```

3. Run "eval_discriminative.py" with command below.

```bash
python3 -m experiment.eval_discriminative --exp_dir <Your Path>
```

4. Run classifier after you configure the "Your Path" part and proper experiment path

```bash
python3 -m experiment.classifier
```

You need to set the 'exp_name' and 'v_exp_name' variable following your path with '.pkl' file extracted in step 2 and 3. 
