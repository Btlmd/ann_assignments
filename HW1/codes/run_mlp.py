from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear, Gelu
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss, HingeLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
from argparse import ArgumentParser
import numpy as np
import random
import wandb
import sys


def init(_config):

    # Random Seed
    np.random.seed(_config.seed)
    random.seed(_config.seed)

    # Activation Recorder
    def _activation_report(_self):
        
        if not hasattr(_self, "activation_record"):
            _self.activation_record = np.zeros(_self.x.shape[1], dtype=np.int64)
            _self.activation_counter = 0

        activation = (_self.x > 0).astype(np.int64)

        # Neuron Record
        _self.activation_record += activation.sum(axis=0)
        _self.activation_counter += activation.shape[0]

        # print("%s: %.4f" % (f"activation/{_self.name}", activation.mean()))
        if not _config.dr:
            wandb.log({
                f"activation/{_self.name.split('_')[-1]}": activation.mean()
            })

    # Norm Recorder
    def _norm_report(_self):
        
        norm = np.linalg.norm(_self.grad_W, 'fro')
        mean = np.abs(_self.grad_W).mean()
        _max = np.abs(_self.grad_W).max()

        # print("%s: %.4f" % (f"activation/{_self.name}", activation.mean()))
        if not _config.dr:
            wandb.log({
                f"g_norm/{_self.name.split('_')[-1]}": norm,
                f"g_max/{_self.name.split('_')[-1]}": _max,
                f"g_mean/{_self.name.split('_')[-1]}": mean,
            })

    # Load Architecture
    _model = Network()

    if _config.nonlinearity == "Sigmoid":
        g = np.sqrt(2)
        NL = Sigmoid
    elif _config.nonlinearity == "Relu":
        g = 1
        NL = Relu
    elif _config.nonlinearity == "Gelu":
        g = 1
        NL = Gelu

    for i in range(len(_config.layout) - 1):
        _model.add(
            Linear(
                f"FC_{i}", 
                _config.layout[i], 
                _config.layout[i + 1], 
                _config.init,
                _norm_report if _config.norm_report else None
            )
        )
        if i < len(_config.layout) - 2:
            _model.add(NL(f"{NL.__name__}_{i}", _activation_report if _config.activation_report else None))
    print(_model)
        
    
    # Define Loss
    if _config.loss.lower() == 'mse':
        _loss = EuclideanLoss(name='loss')
    elif _config.loss.lower() == 'ce':
        _loss = SoftmaxCrossEntropyLoss(name='loss')
    elif _config.loss.lower() == 'hinge':
        _loss = HingeLoss(name='loss', margin=config.margin)
    else:
        raise ValueError("Unexpected loss type " + _config['loss'])

    if not _config.dr:
        # Connect WandB
        wandb.init(
            project=f"ann-1_{_config.seed}",
            config={
                **vars(_config),
                "command": sys.argv
            },
            name=_config.name
        )

        # Define Metrics
        wandb.define_metric("train/loss", summary="min")
        wandb.define_metric("train/acc", summary="max")
        wandb.define_metric("train/time", summary="mean")
        wandb.define_metric("test/loss", summary="min")
        wandb.define_metric("test/acc", summary="max")

    return _model, _loss

    
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--loss", required=True, type=str, choices=["mse", "ce", "hinge"])
    parser.add_argument("--layout", required=True, type=int, nargs='+')
    parser.add_argument("--nonlinearity", required=True, type=str, choices=["Gelu", "Relu", "Sigmoid"])
    parser.add_argument("--learning_rate", default=5e-2, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--max_epoch", default=100, type=int)
    parser.add_argument("--disp_freq", default=100, type=int)
    parser.add_argument("--test_epoch", default=1, type=int)
    parser.add_argument("--init", default=0.01, type=float)
    parser.add_argument("--seed", default=2022, type=int)
    parser.add_argument("--log", default='iteration', type=str, choices=["iteration", "epoch"])
    parser.add_argument("--margin", default=0.1, type=float)
    parser.add_argument("--activation_report", default=False, action="store_true")
    parser.add_argument("--activation_th", default=0.2, action="store_true")
    parser.add_argument("--dr", default=False, action="store_true")
    parser.add_argument("--norm_report", default=False, action="store_true")
    parser.add_argument("--data_dir", default='data', type=str)

    config = parser.parse_args()

    train_data, test_data, train_label, test_label = load_mnist_2d(config.data_dir)

    model, loss = init(config)
    
    iteration = 0
    config = vars(config)
    for epoch in range(config['max_epoch']):
        LOG_INFO('Training @ %d epoch...' % (epoch))
        iteration = train_net(model, loss, config, train_data, train_label, iteration, config['batch_size'], config['disp_freq'], epoch)

        if epoch % config['test_epoch'] == 0:
            LOG_INFO('Testing @ %d epoch...' % (epoch))
            test_net(model, loss, test_data, test_label, config['batch_size'], iteration, epoch + 1, config)
    
        # Activation Record for Each Epoch
        if config['activation_report']:
            for layer in model.layer_list:
                if hasattr(layer, "activation_record"):
                    alive_pct = ((layer.activation_record.astype(np.float64) / layer.activation_counter) > config['activation_th']).mean()
                    if not config['dr']:
                        wandb.log({
                            f"activate_gt_{config['activation_th']}/{layer.name.split('_')[-1]}": alive_pct
                        }, step=iteration)
                    print("Alive", layer, round(alive_pct, 4))                    
