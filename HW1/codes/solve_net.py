from utils import LOG_INFO, onehot_encoding, calculate_acc
import numpy as np
import time
import wandb


def data_iterator(x, y, batch_size, shuffle=True):
    indx = list(range(len(x)))
    if shuffle:
        np.random.shuffle(indx)

    for start_idx in range(0, len(x), batch_size):
        end_idx = min(start_idx + batch_size, len(x))
        yield x[indx[start_idx: end_idx]], y[indx[start_idx: end_idx]]


def train_net(model, loss, config, inputs, labels, iter_counter, batch_size, disp_freq, epoch):

    loss_list = []
    acc_list = []
    time_list = []

    for batch_idx, (input, label) in enumerate(data_iterator(inputs, labels, batch_size)):
        tick_0 = time.time()
        target = onehot_encoding(label, 10)
        iter_counter += 1

        # forward net
        output = model.forward(input)
        # calculate loss
        loss_value = loss.forward(output, target)
        # generate gradient w.r.t loss
        grad = loss.backward(output, target)
        # backward gradient

        model.backward(grad)
        # update layers' weights
        model.update(config)

        acc_value = calculate_acc(output, label)
        loss_list.append(loss_value)
        acc_list.append(acc_value)
        time_list.append(time.time() - tick_0)

        if iter_counter % disp_freq == 0:
            mean_loss, mean_acc, mean_time = np.mean(loss_list), np.mean(acc_list), np.mean(time_list)
            if config['log'] == 'iteration' and not config['dr']:
                wandb.log({
                    "train/loss": mean_loss,
                    "train/acc": mean_acc,
                    "train/time": mean_time
                }, step=iter_counter)
            msg = '  Training iter %d, batch loss %.4f, batch acc %.4f, time_cost %.4f' % (iter_counter, mean_loss, mean_acc, mean_time)
            loss_list = []
            acc_list = []
            LOG_INFO(msg)
    
    if config['log'] == 'epoch' and not config["dr"]:
        wandb.log({
            "train/loss": mean_loss,
            "train/acc": mean_acc,
            "train/time": mean_time
        }, step=epoch)
    return iter_counter


def test_net(model, loss, inputs, labels, batch_size, iteration, epoch, config):
    loss_list = []
    acc_list = []

    for input, label in data_iterator(inputs, labels, batch_size, shuffle=False):
        target = onehot_encoding(label, 10)
        output = model.forward(input)
        loss_value = loss.forward(output, target)
        acc_value = calculate_acc(output, label)
        loss_list.append(loss_value)
        acc_list.append(acc_value)
    if not config["dr"]:
        wandb.log({
            "test/loss": np.mean(loss_list),
            "test/acc": np.mean(acc_list)
        }, step=iteration if config['log'] == 'iteration' else epoch)

    msg = '    Testing, total mean loss %.5f, total acc %.5f' % (np.mean(loss_list), np.mean(acc_list))
    LOG_INFO(msg)
