# Neural Importance Sampling 
## Configuration example
config/config.conf:
```
train
{
  function = Gaussian
  epochs = 200
  batch_size = 100
  learning_rate = 0.0001
  num_hidden_dims = 128
  num_coupling_layers = 4
  num_hidden_layers = 5
  num_blob_bins = 0
  num_piecewise_bins = 5
  loss = MSE
  coupling_name="piecewiseLinear",
}
logging
{
  plot_dir_name = "./Plots/SaveOutput"
  save_plots = True
  save_plt_interval = 10
  plot_dimension = 2
  tensorboard
  {
    use_tensorboard = False
    # wandb_project = "NIS"
  }
}
```
## How to run code 

Try : 

```
python3 Integration2D.py --help
```

To see all available arguments

Some testing functions are in functions.py



