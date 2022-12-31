from ast import arg
import tqdm
import torch
import torch.nn as nn
import json
import numpy as np
import time
import random
import argparse
import torch
from torch import optim
import torch.nn.functional as F
from tokenizer import get_tokenizer
import os
from model_tfmr import TfmrLMHeadModel, TransposeLinear
from configuration import ModelConfig
from copy import deepcopy

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, default="run",
    help='Experiment name. Default: run')
parser.add_argument('--model_config', type=str, default="./config.json",
    help='Path to the configuration file. Default: ./config.json')    
parser.add_argument('--tokenizer_dir', type=str, default="./tokenizer",
    help='Tokenizer file directory. Default: ./tokenizer')    
parser.add_argument('--num_epochs', type=int, default=20,
    help='Number of training epoch. Default: 20')
parser.add_argument('--cpu_count', type=int, default=64,
    help='Number of CPU cores for evaluation. Default: 20')    
parser.add_argument('--batch_size', type=int, default=32,
    help='The number of batch_size. Default: 32')
parser.add_argument('--learning_rate', type=float, default=1e-4,
    help='Learning rate during optimization. Default: 1e-4')
parser.add_argument('--test', type=str, default=None,
    help='Evaluate the model with the specified name. Default: None')
parser.add_argument('--data_dir', type=str, default='./data',
    help='Data directory. Default: ../data')
parser.add_argument('--train_dir', type=str, default='./checkpoints',
    help='Training directory for saving model. Default: ./train')
parser.add_argument('--pretrain_dir', type=str, default='None',
    help='Pre-Training directory for loading pretrained model. Default: None')
parser.add_argument('--pretrain_ckpt', type=str, default='pretrained_ckpt12.tar',
    help='Pre-Trained checkpoint.')
parser.add_argument('--maxlen', type=int, default=35,
    help='Maximum length for training/inference. Default: 35')    
parser.add_argument('--decode_strategy', type=str, choices=["random", "top-p", "top-k"], default="random",
    help='The strategy for decoding. Can be "random", "top-p" or "top-k". Default: random')
parser.add_argument('--temperature', type=float, default=1,
    help='The temperature for decoding. Default: 1')
parser.add_argument('--top_p', type=float, default=0.9,
    help='The p for top-p sampling. Default: 1.0')    
parser.add_argument('--top_k', type=int, default=40,
    help='The k for top-k sampling. Default: 40')
parser.add_argument('--tb_path', type=str, default='../tb',
                    help='Tensorboard Path')
parser.add_argument('--shuffle', default=False, action='store_true',
                    help='Shuffle the training dataset')
parser.add_argument('--shuffle_seed', default=2022, type=int)
parser.add_argument('--seed', default=2022, type=int)
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--head', type=int, default=None)
parser.add_argument('--layers', type=int, nargs='+', default=[1, 2, 3])
parser.add_argument('--load', default=False, action='store_true')
parser.add_argument('--output_dir', type=str, default='output')
parser.add_argument('--full_epoch', default=False, action='store_true')
parser.add_argument('--observe_bleu', default=False, action='store_true')
parser.add_argument('--es_criteria', default='ppl', choices=['ppl', 'bleu'])
parser.add_argument('--tolerance', type=int, default=0)
parser.add_argument('--data_cross_bleu', default=False, action='store_true')
parser.add_argument('--data_repeat', default= 1, type=int)

args = parser.parse_args()

if args.es_criteria == 'bleu':
    args.observe_bleu = True

# Global Random Seed
if args.seed < 0:
    args.seed = int(time.time())
print("Random Seed", args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from torch.utils.tensorboard import SummaryWriter

import nltk
assert nltk.__version__ == "3.5", f"Invalid nltk version {nltk.__version__}"

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
def _sentence_bleu(ele):
    return sentence_bleu(ele[0], ele[1], weights=ele[2], smoothing_function=SmoothingFunction().method1)

def fast_evaluate(model, data, batch_size, PAD_ID, device):
    model.eval()
    st, ed, all_loss = 0, 0, []
    bar = tqdm.tqdm(desc="Evalutaing PPL")
    while ed < len(data):
        st, ed = ed, (ed + batch_size) if (ed + batch_size < len(data)) else len(data)
        with torch.no_grad():
            input_ids = torch.tensor(data[st:ed]).to(device)
            ce_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

            # TODO START
            outputs = model(input_ids)
            lm_logits = outputs["logits"]
            loss = model.batch_loss(lm_logits, input_ids, ce_loss_fct, PAD_ID)
            # TODO END
            all_loss += loss.cpu().numpy().tolist()
        bar.update(1)
    loss = np.mean(all_loss)
    ppl = np.exp(loss)
    model.train()
    return loss, ppl

def evaluate(gen_ids, truth_ids, cpu_count=20):
    from multiprocessing import Pool
    print("Evaluating BLUE score with %d CPUs" % cpu_count)

    # assert len(gen_ids) == len(truth_ids)
    # sample_hyps_num = len(gen_ids)
    res = {}
    for ngrams in [4]:
        print("computing BLEU-%d"%ngrams)
        bleu_irl_fw, bleu_irl_bw = [], []
        weights = np.ones(ngrams) / ngrams

        tasks = ((truth_ids, gen_ids[i], weights) for i in range(len(gen_ids)))
        pool = Pool(cpu_count)
        values = pool.imap_unordered(_sentence_bleu, tasks, chunksize=20)
        values = tqdm.tqdm(values, total=len(gen_ids))
        for ans in values:
            bleu_irl_fw.append(ans)
        pool.close()
        pool.join()

        tasks = ((gen_ids, truth_ids[i], weights) for i in range(len(truth_ids)))
        pool = Pool(cpu_count)
        values = pool.imap_unordered(_sentence_bleu, tasks, chunksize=20)
        values = tqdm.tqdm(values, total=len(truth_ids))
        for ans in values:
            bleu_irl_bw.append(ans)
        pool.close()
        pool.join()

        fw_bleu = (1.0 * sum(bleu_irl_fw) / len(bleu_irl_fw))
        bw_bleu = (1.0 * sum(bleu_irl_bw) / len(bleu_irl_bw))
        if fw_bleu + bw_bleu > 0:
            fw_bw_bleu = 2.0 * bw_bleu * fw_bleu / (fw_bleu + bw_bleu)
        else:
            fw_bw_bleu = 0

        res.update({"fw-bleu-%d"%ngrams : fw_bleu, \
            "bw-bleu-%d"%ngrams : bw_bleu, \
            "fw-bw-bleu-%d"%ngrams : fw_bw_bleu \
        })
    return res

def load_data(path, tokenizer, PAD_ID, field_list=["train", "dev", "test"], maxlen=40):
    data, data_remove_pad = {}, {}
    for name in field_list:
        data[name], data_remove_pad[name] = [], []
        with open("%s/%s.txt"%(path, name)) as fin:
            for line in fin:
                tokens = tokenizer.encode(line.strip())
                if len(tokens) < maxlen:
                    data[name].append([PAD_ID] + tokens + [PAD_ID]*(maxlen - len(tokens)))
                else:
                    data[name].append([PAD_ID] + tokens[:maxlen])
                data_remove_pad[name].append(tokens)
    return data, data_remove_pad

def get_init_weights_func(config):
    def init_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, TransposeLinear)):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    return init_weights

def create_model(_args):
    with open(_args.model_config) as fin:
        model_config = json.load(fin)
    if _args.head is not None:
        model_config['n_head'] = _args.head
        assert _args.pretrain_dir == 'None'
    if _args.layers is not None:
        model_config['n_layer'] = len(_args.layers)
    print("Model Configs", model_config)
    config = ModelConfig(**model_config)
    _model = TfmrLMHeadModel(config)
    init_weights_func = get_init_weights_func(config=config)
    _model.apply(init_weights_func)
    return _model

def load_ckpt(_args):
    _path = os.path.join(_args.pretrain_dir, _args.pretrain_ckpt)
    if os.path.exists(_path):
        print("Loading model from %s" % _path)
        _model = torch.load(_path)
    else:
        raise RuntimeError("No such checkpoint: %s" % _path)
    return _model

def evaluate_bleu(split='test'):
    result = model.inference(device=args.device, PAD_ID=PAD_ID,
                             batch_size=args.batch_size, maxlen=args.maxlen, decode_strategy=args.decode_strategy,
                             temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)

    eval_result = evaluate(gen_ids=result, truth_ids=data_remove_pad[split], cpu_count=args.cpu_count)
    return result, eval_result

class WindowLogger:
    def __init__(self, _writer: SummaryWriter, _window: int = 20, _log_name = 'loss/train'):
        self.window = _window
        self.writer = _writer
        self.running_loss = []
        self.log_name = _log_name

    def append(self, val, _step):
        self.running_loss += [val]
        if len(self.running_loss) == self.window:
            self.writer.add_scalar(self.log_name, np.mean(self.running_loss), _step)
            self.running_loss = []

if __name__ == '__main__':
    print(args)
    device = args.device
    if not os.path.exists(args.train_dir):
        os.mkdir(args.train_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = get_tokenizer(args.tokenizer_dir)
    PAD_ID = tokenizer.encoder['<|endoftext|>']
    print("Tokenizer PAD ID:", PAD_ID)

    print("Loading Data ...")
    data, data_remove_pad = load_data(path=args.data_dir, tokenizer=tokenizer, PAD_ID=PAD_ID, field_list=["train", "dev", "test"], maxlen=args.maxlen)

    if args.data_cross_bleu:
        dev_on_test = evaluate(data_remove_pad['dev'], data_remove_pad['test'], args.cpu_count)
        train_on_test = evaluate(data_remove_pad['train'], data_remove_pad['test'], args.cpu_count)
        train_on_dev = evaluate(data_remove_pad['train'], data_remove_pad['dev'], args.cpu_count)
        print("dev_on_test", dev_on_test)
        print("train_on_test", train_on_test)
        print("train_on_dev", train_on_dev)
        exit(0)

    if args.test is None:
        writer = SummaryWriter(os.path.join(args.tb_path, args.name))
        if args.load:
            print("Load pretrained model")
            model = load_ckpt(args)
        elif args.pretrain_dir == 'None':
            print("Create model with fresh parameters.")
            model = create_model(args)
        else:
            assert args.layers is not None
            print("Load pretrained model")
            pretrained_model = load_ckpt(args)
            selected_modules = [pretrained_model.transformer.h[x - 1] for x in args.layers]
            pretrained_model.transformer.h = torch.nn.ModuleList(selected_modules)

            print("Create model with specified layers", args.layers)
            model = create_model(args)
            model.load_state_dict(pretrained_model.state_dict(), strict=True)

        model.to(device)
        print(model)

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
        best_val_ppl = float("inf")
        best_epoch = -1
        best_bleu = 0
        bleu = 0
        step = 0
        wl = WindowLogger(writer)
        criteria_failure_counter = 0

        lst = []
        for i in range(args.data_repeat):
            lst += deepcopy(data["train"])
        data["train"] = lst

        for epoch in range(1, args.num_epochs + 1):
            print(optimizer)
            start_time = time.time()
            st, ed, batch_num = 0, 0, 0
            losses = []
            if args.shuffle:
                rng = random.Random(args.shuffle_seed + epoch)
                rng.shuffle(data["train"])
            while ed < len(data["train"]):
                batch_num += 1
                st_time = time.time()
                st, ed = ed, ((ed + args.batch_size) if (ed + args.batch_size < len(data["train"])) else len(data["train"]))
                batched_data = torch.tensor(data["train"][st:ed]).to(device)

                optimizer.zero_grad()
                loss = model(input_ids=batched_data, labels=batched_data, PAD_ID=PAD_ID)["loss"]
                loss.backward()
                optimizer.step()
                step += 1
                losses.append(loss.tolist())
                wl.append(loss.tolist(), step)
                writer.add_scalar('loss/step_loss', loss.tolist(), step)

                if (batch_num) % 10 == 0:
                    print("Epoch %d Batch %d, train loss %f" % (epoch, batch_num, np.mean(losses[-100:])))

            train_loss = np.mean(losses)

            val_loss, val_ppl = fast_evaluate(model=model, data=data["dev"], batch_size=args.batch_size, PAD_ID=PAD_ID, device=device)
            writer.add_scalar('loss/validation', val_loss.item(), step)
            writer.add_scalar('perplexity/validation', val_ppl.item(), step)

            def better_criteria():
                return val_ppl < best_val_ppl if args.es_criteria == 'ppl' else bleu > best_bleu

            def continue_criteria():
                global criteria_failure_counter
                if better_criteria():
                    criteria_failure_counter = 0
                    return True
                else:
                    criteria_failure_counter += 1
                    if criteria_failure_counter >= args.tolerance:
                        return False
                    return True

            def info():
                epoch_time = time.time() - start_time
                print("Epoch " + str(epoch) + " of " + str(args.num_epochs) + " took " + str(epoch_time) + "s")
                print("  training loss:                 " + str(train_loss))
                print("  validation loss:               " + str(val_loss))
                print("  validation perplexity:         " + str(val_ppl))
                print("  best epoch:                    " + str(best_epoch))
                print("  best validation perplexity:    " + str(best_val_ppl))
                if args.observe_bleu:
                    print("  dev_set, forward BLEU-4 %.3f, backward BLEU-4 %.3f, harmonic BLEU-4 %.3f" % (
                        eval_result["fw-bleu-4"], eval_result["bw-bleu-4"], eval_result["fw-bw-bleu-4"]))
                print("  criteria failure counter %d" % criteria_failure_counter)

            if args.observe_bleu:
                _, eval_result = evaluate_bleu('dev')
                writer.add_scalar('bleu-4/validation_forward', eval_result['fw-bleu-4'], step)
                writer.add_scalar('bleu-4/validation_backward', eval_result['bw-bleu-4'], step)
                writer.add_scalar('bleu-4/validation_harmonic', eval_result['fw-bw-bleu-4'], step)
                bleu = eval_result['fw-bw-bleu-4']

            if continue_criteria() or args.full_epoch:
                if better_criteria():
                    with open(os.path.join(args.train_dir, 'checkpoint_%s.pth.tar' % args.name), 'wb') as fout:
                        print("Saving checkpoint as a better result")
                        torch.save(model, fout)

                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    best_epoch = epoch
                if bleu > best_bleu:
                    best_bleu = bleu

                info()
            else:
                info()
                time.sleep(5)
                break

    else:
        model_path = os.path.join(args.train_dir, 'checkpoint_%s.pth.tar' % args.test)
        if os.path.exists(model_path):
            print("Loading model from %s" % model_path)
            model = torch.load(model_path)
        else:
            raise RuntimeError("No such checkpoint")
        model.to(device)
        print(model)
        test_loss, test_ppl = fast_evaluate(model=model, data=data["test"], batch_size=args.batch_size, PAD_ID=PAD_ID, device=device)
        print("        test_set, perplexity %.2f" % (test_ppl))

        result, eval_result = evaluate_bleu()
        print("        test_set, forward BLEU-4 %.3f, backward BLEU-4 %.3f, harmonic BLEU-4 %.3f" % (
            eval_result["fw-bleu-4"], eval_result["bw-bleu-4"], eval_result["fw-bw-bleu-4"]))
        print("        test_set, write inference results to output_%s.txt" % args.decode_strategy)
        with open(os.path.join(args.output_dir, '%s_%.2f_%s.txt' % (args.test, args.temperature, args.decode_strategy)), 'w') as fout:
            fout.write(
"""[Checkpoint Name] %s
[Temp, Strategy ] %.2f %s
[Perplexity     ] %.2f
[Forward BLEU-4 ] %.3f
[Backward BLEU-4] %.3f
[Harmonic BLEU-4] %.3f

""" % (args.test, args.temperature, args.decode_strategy, test_ppl, eval_result["fw-bleu-4"], eval_result["bw-bleu-4"], eval_result["fw-bw-bleu-4"]))
            for k, output in enumerate(result):
                out = tokenizer.decode(output)
                # print(k, out)
                fout.write(out + "\n")
        with open(os.path.join(args.output_dir, 'inference.jsonl'), 'a') as f:
            json.dump({
                "config": vars(args),
                "ppl": test_ppl,
                "f_bleu4": eval_result["fw-bleu-4"],
                "b_bleu4": eval_result["bw-bleu-4"],
                "h_bleu4": eval_result["fw-bw-bleu-4"]
            }, f)
            f.write("\n")
