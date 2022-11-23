# Neural Importance Sampling 

## How to run code 
Server:
```
integration_experiment.py -c some_config.conf
```
Example config file:
```
train
{
  function = Gaussian
  epochs = 200
  batch_size = 100
  learning_rate = 0.0001
  num_hidden_dims = 128
  num_coupling_layers = 3
  num_hidden_layers = 5
  num_blob_bins = 0
  num_piecewise_bins = 5
  loss = MSE
  num_context_features = 6
  coupling_name="piecewiseLinear",
  hybrid_sampling = False
}
logging
{
  plot_dir_name = "./Plots/SaveOutput"
  save_plots = False
  save_plt_interval = 10
  plot_dimension = 2
  tensorboard
  {
    use_tensorboard = False
    # wandb_project = "NIS"
  }
}
```
Client:
```
client.py
```



