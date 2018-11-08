import argparse
import loader
from torchtext import data
import models
#from torch.nn import init
import config
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from bleu_score import bleu
from test_tube import Experiment, HyperOptArgumentParser, SlurmCluster
import os
import pathlib

def run(args):
    device = torch.device("cuda" if (not args.cpu) and torch.cuda.is_available() else "cpu")
    print("Using device", device)
    
    train_data, val_data, test_data, src, trg = loader.load_data(args)
    
    src_padding_idx = src.vocab.stoi['<pad>']
    trg_padding_idx = trg.vocab.stoi['<pad>']
    
    for i in range(5):
        print(i, src.vocab.itos[i])
        print(i, trg.vocab.itos[i])
    
    assert src_padding_idx == config.PAD_TOKEN
    assert trg_padding_idx == config.PAD_TOKEN
    
    #src_unk_idx = src.vocab.stoi['<unk>']
    #trg_unk_idx = trg.vocab.stoi['<unk>']
    
    src_vocab_size = len(src.vocab)
    trg_vocab_size = len(trg.vocab)
    
    encoder = models.RnnEncoder(args, src_padding_idx, src_vocab_size).to(device)
    decoder = models.RnnDecoder(args, trg_padding_idx, trg_vocab_size).to(device)
    
    # initialize weights using gaussian with 0 mean and 0.01 std, just like the paper said
    # TODO: Better initialization. Xavier?
    for net in [encoder, decoder]:
        for name, param in net.named_parameters(): 
            #print(name, type(param), param)
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
            #nn.init.normal_(param, std=0.01)

    # TODO: other optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.l2_penalty)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.l2_penalty)
    
    # TODO: use different loss?
    loss_function = nn.NLLLoss()
    
    # TODO: save/load weights
    # TODO: early stopping
    loss_history = defaultdict(list)
    bleu_history = defaultdict(list)
    
    # Initiate test-tube experiment object
    exp = Experiment(
        name='rnn_encoder_decoder',
        save_dir=args.logs_path,
        autosave=True,
    )
    exp.argparse(args)

    model_path = os.path.join(args.model_weights_path, exp.name)
    model_path = os.path.join(model_path, str(exp.version))
    pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    for i in range(args.epoch):
        if args.test:
            test(args, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, device, i, test_data)
        else:
            train_loss, val_loss, val_bleu = train_and_val(args, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, device, i, train_data, val_data, trg)
            loss_history["train"].append(train_loss)
            loss_history["val"].append(val_loss)
            bleu_history["val"].append(val_bleu)
            
            # update best models
            if val_bleu == np.max(bleu_history["val"]):
                # save model weights of the best models
                torch.save(encoder.state_dict(), os.path.join(model_path, 'encoder_weights.pt'))
                torch.save(decoder.state_dict(), os.path.join(model_path, 'decoder_weights.pt'))
            if args.save_all_epoch:
                model_path_current_epoch = os.path.join(model_path, str(i))
                pathlib.Path(model_path_current_epoch).mkdir(parents=True, exist_ok=True)
                torch.save(encoder.state_dict(), os.path.join(model_path_current_epoch, 'encoder_weights.pt'))
                torch.save(decoder.state_dict(), os.path.join(model_path_current_epoch, 'decoder_weights.pt'))
                

            # add logs
            exp.log({'train epoch loss': train_loss})
            exp.log({'val epoch loss': val_loss})
            exp.log({'val epoch bleu': val_bleu})
            
            if early_stop(loss_history["val"], args.early_stopping):
                print("Early stopped.")
                break
    

                
def early_stop(loss_history, early_stop_k):
    if len(loss_history) < early_stop_k:
        return False
    idx_min = loss_history.index(min(loss_history))
    return len(loss_history) - 1 - idx_min >= early_stop_k
        
def run_batch(phase, args, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, batch, device):

    assert phase in ("train", "val", "test"), "invalid phase"
    
    if phase == "train":
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
    
    loss = 0
    
    # TODO: it seems that currently batch size is always the same. Make sure to use the last batch
    target_sequence_length, batch_size = batch.trg[0].shape
    
    encoder.random_init_hidden(device, batch_size)
    
    # Move batch.src and batch.trg to cuda
    #batch.src[0] = batch.src[0].cuda()
    #batch.src[1] = batch.src[1].cuda()
    #batch.trg[0] = batch.trg[0].cuda()
    #batch.trg[1] = batch.trg[1].cuda()
    
    encoder(batch.src[0], batch.src[1])
    
    # This step is necessary to get the hidden state from encoder
    # TODO: is this mechanism of getting hidden layer correct?
    # Shouldn't it be last n layers, instead of first n layers?
    decoder.hidden = encoder.hidden[:decoder.n_layers]
    
    
    # TEACHER FORCING
    # Feed all target sentences at once instead of reusing output as input
    # nice to look, and should be fast
    number_of_loss_calculation = 0
    if phase == 'train' and np.random.random() < args.teacher_forcing:
        logits = decoder(batch.trg[0])
        # get prediction
        # log_softmax with NLLLoss == CrossEntropyLoss with logits
        output = F.log_softmax(logits, dim=1)

        # Now, loop through output and calculate loss
        # be careful not to compare SOS_token with first non_sos token
        # make sure to stop when encountered EOS_token
    
        # order of nested for loop is important!
        for j in range(batch_size):
            for i in range(target_sequence_length-1):
                if batch.trg[0][i,j] == config.EOS_TOKEN:
                    # we do not need to feed the last EOS token to the decoder. 
                    # we do not need to feed any padding token either. 
                    # move onto the next data in the batch.
                    break
                # first output is what the decoder produced after seeing SOS token
                # therefore, compare it with second target token
                loss += loss_function(output[i,j,:].view(1,-1), batch.trg[0][i+1,j].view(1))
                number_of_loss_calculation += 1
                
    else:
        # this is needed when we are not using teacher forcing
        # To feed output of the decoder (the word with highest prob) into itself, has to use for loop, will be slower (is there faster alternative?)
        translated_tokens_list = []
        decoder_input = torch.tensor([config.SOS_TOKEN]*batch_size, device=device, requires_grad=False).view(1,-1)
        translated_tokens_list.append(decoder_input)
        eos_encountered_list = [False]*batch_size
        i = 0
        
        # TODO: Even though the logic might be correct, the speed is extremely slow.
        while ((i < args.max_sentence_length) and (i+1 < target_sequence_length) and (sum(eos_encountered_list) < batch_size)): # fix off-by-1 error, if any
            
            logits = decoder(decoder_input)
            output = F.log_softmax(logits, dim=1)
            decoder_input = torch.tensor([config.PAD_TOKEN]*batch_size, device=device, requires_grad=False).view(1,-1)
            for j in range(batch_size):   
                
                if not eos_encountered_list[j]:
                    # get index of maximum probability word
                    max_index = output[0,j].max(0)[1]
                    decoder_input[0,j] = max_index
                    loss += loss_function(output[0,j,:].view(1,-1), batch.trg[0][i+1,j].view(1))
                    number_of_loss_calculation += 1
                
                    if max_index == config.EOS_TOKEN or batch.trg[0][i+1,j] == config.EOS_TOKEN: #?
                        # if EOS token, stop.
                        eos_encountered_list[j] = True
                    
            
            translated_tokens_list.append(decoder_input)
            i += 1
    
        
    if phase == "train":
        loss.backward() # this should calculate gradient for both encoder and decoder

        nn.utils.clip_grad_norm_(encoder.parameters(), args.clip)
        nn.utils.clip_grad_norm_(decoder.parameters(), args.clip)

        decoder_optimizer.step() # does it really matter which one takes step first?
        encoder_optimizer.step()

        # do not return translation output when training
        translation_output = None
    
    # if validation, test: output translated sentences as well
    else:
        translation_output = torch.cat(translated_tokens_list, dim=0)
        
    return loss.item() / number_of_loss_calculation, translation_output
    
    
def train_and_val(args, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, device, epoch_idx, train_data, val_data, trg):
    train_iter = data.BucketIterator(
        dataset=train_data, 
        batch_size=args.batch_size,
        repeat=False,
        sort_key=lambda x: len(x.src),
        sort_within_batch=True,
        device=device,
        train=True
    )
    
    val_iter = data.BucketIterator(
        dataset=val_data, 
        batch_size=args.batch_size,
        train=False,
        shuffle=False,
        #A key to use for sorting examples in order to batch together 
        # examples with similar lengths and minimize padding.
        sort=True,
        sort_key=lambda x: len(x.src),
        repeat=False,
        sort_within_batch=True,
        device=device
    )

    # turn on dropout
    encoder.train()
    decoder.train()
    
    train_loss_list = []
    for i, train_batch in enumerate(iter(train_iter)):
        loss, _ = run_batch(
            "train",
            args,
            encoder,
            decoder, 
            encoder_optimizer, 
            decoder_optimizer, 
            loss_function,
            train_batch,
            device
        )
        train_loss_list.append(loss)
        # TODO: use tensorboard to see train / val plot while training
        if i % args.print_every == 0:
            print("train, epoch: {}, step: {}, average loss for current epoch: {}, batch loss: {}".format(
                epoch_idx, i, np.mean(train_loss_list), loss))
        
    print("train done. epoch: {}, average loss for current epoch: {}, numbatch: {}, size of last batch: {}".format(
        epoch_idx, np.mean(train_loss_list), i+1, train_batch.src[0].shape[1]))

    # turn off dropout
    encoder.eval()
    decoder.eval()
    
    val_loss_list = []
    val_bleu_list = []
    for i, val_batch in enumerate(iter(val_iter)):
        #val_batch.trg: size N x B
        loss, translation_output = run_batch(
            "val",
            args,
            encoder,
            decoder, 
            encoder_optimizer, 
            decoder_optimizer, 
            loss_function,
            val_batch,
            device
        )
        #translation_output = indices, N x B
        #todo: 1. check if trg.vocab.itos is pass in this function. 2.!reference! 
        val_reference = []
        for each in val_batch.idx:
            val_reference.append(" ".join(train_iter.dataset[each].trg))
        val_bleu = bleu(trg.vocab.itos, translation_output, val_reference)
        val_bleu_list.append(val_bleu)
        val_loss_list.append(loss)
        if i % args.print_every == 0:
            print("val, epoch: {}, step: {}, average loss for current epoch: {}, batch loss: {}, average bleu for current epoch: {}, batch bleu: {}".format(
                epoch_idx, i, np.mean(val_loss_list), loss, np.mean(val_bleu_list), val_bleu))
        
    print("val done. epoch: {}, average loss for current epoch: {}, average bleu for current epoch: {}".format(
        epoch_idx, np.mean(val_loss_list), np.mean(val_bleu_list)))
    
    return np.mean(train_loss_list), np.mean(val_loss_list), np.mean(val_bleu_list)
        
def test():
    pass

def rnn_encoder_decoder_argparser():
    # TODO: set min max size for vocab
    # TODO: specify optimizer
    # TODO: set max length of sentences
    
    # parser = argparse.ArgumentParser(description='Run Tests')
    parser = HyperOptArgumentParser(description='Run Tests', strategy='grid_search')
    
    parser.add_argument('--test', help="Run test instead of training/validation", action="store_true")
    parser.add_argument("--njobs", help="Number of jobs to use when loading data", type=int, default=1)
    parser.add_argument("--epoch", help="Number of epoch to run", type=int, default=10000)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=32)
    parser.add_argument("--hidden_size", help="Hidden size", type=int, default=256)
    parser.add_argument("--embedding_size", help="Embedding size", type=int, default=256)
    parser.add_argument("--print_every", help="How frequently print result", type=int, default=100)
    parser.add_argument("--lr", help="Learning Rate", type=float, default=1e-03)
    parser.add_argument("--dropout", help="Dropout Rate", type=float, default=0)
    parser.add_argument('--bidirectional', help="Use bidirectional RNN in encoder", action="store_true") # don't set this true in this model
    parser.add_argument('--cpu', help="Use cpu instead of gpu", action="store_true")
    parser.add_argument("--data", help="Directory where data is stored", default='../data/iwslt-zh-en/')
    parser.add_argument("--rnn_type", help="Which rnn to use (rnn, lstm, gru)", default='gru')
    parser.add_argument("--num_encoder_layers", help="Number of rnn layers in encoder", type=int, default=1)    
    parser.add_argument("--num_decoder_layers", help="Number of rnn layers in encoder", type=int, default=1)
    parser.add_argument("--early_stopping", help="Stop if validation does not improve", type=int, default=10)
    parser.add_argument('--l2_penalty', help="L2 pelnalty coefficient in optimizer", type=float,  default=0) #1e-06
    parser.add_argument('--clip', help="clip coefficient in optimizer", type=float,  default=1)
    parser.add_argument('--teacher_forcing', help="probability of performing teacher forcing", type=float,  default=0.5)
    parser.add_argument("--max_sentence_length", help="maximum sentence length", type=int, default=80)
    parser.add_argument('--split_chinese_into_characters', help="Split chinese into characters", action="store_true")
    parser.add_argument("--min_freq", help="Vocabulary needs to be present at least this amount of time", type=int, default=3)
    parser.add_argument("--max_vocab_size", help="At most n vocaburaries are kept in the model", type=int, default=100000)
    parser.add_argument("--source_lang", help="Source language (vi, zh)", default="zh")
    parser.add_argument("--logs_path", help="Path to save training logs", type=str, default="../training_logs/")
    parser.add_argument("--model_weights_path", help="Path to save best model weights", type=str, default="../model_weights/")
    parser.add_argument('--save_all_epoch', help="save all epoch", action="store_true")
    return parser

if __name__ == '__main__':
    parser = rnn_encoder_decoder_argparser()
    args = parser.parse_args()
    run(args)
    

