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
from torch.autograd import Variable
import numpy as np
from collections import defaultdict
from bleu_score import bleu
from test_tube import Experiment, HyperOptArgumentParser, SlurmCluster
import os
import pathlib
from detok import detok

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
    
    encoder = models.CnnEncoder(args, src_padding_idx, src_vocab_size).to(device)
    if args.attention:
        assert args.bidirectional, "if using attention model, bidirectional must be true"
        decoder = models.LuongAttnDecoderRNN(args, trg_padding_idx, trg_vocab_size).to(device)
    else:
        assert not args.bidirectional, "if not using attention model, bidirectional must be false"
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
            
    if args.encoder_word_embedding is not None:
        encoder_embedding_dict = torch.load(args.encoder_word_embedding)
        encoder.word_embedding.load_state_dict({'weight': encoder_embedding_dict['weight']})
        if args.freeze_all_words:
            encoder.word_embedding.requires_grad=False
    else: #####
        encoder_embedding_dict = None #####
    if args.decoder_word_embedding is not None:
        decoder_embedding_dict = torch.load(args.decoder_word_embedding)
        decoder.embedding.load_state_dict({'weight': decoder_embedding_dict['weight']})
        if args.freeze_all_words:
            decoder.embedding.requires_grad=False
    else: #####
        decoder_embedding_dict = None #####
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
    if not args.test:
        exp = Experiment(
            name=args.name,
            save_dir=args.logs_path,
            autosave=True,
        )
        exp.argparse(args)

        model_path = os.path.join(args.model_weights_path, exp.name)
        model_path = os.path.join(model_path, str(exp.version))
        pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
        print(model_path)
    
    if args.test:
        encoder.load_state_dict(torch.load(os.path.join(args.model_weights_path, 'encoder_weights.pt')))
        decoder.load_state_dict(torch.load(os.path.join(args.model_weights_path, 'decoder_weights.pt')))
        return test(args, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, device, i, test_data, trg, encoder_embedding_dict, decoder_embedding_dict)
    else:
        for i in range(args.epoch):
            train_loss, val_loss, val_bleu = train_and_val(args, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, device, i, train_data, val_data, trg, encoder_embedding_dict, decoder_embedding_dict)
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
            exp.log({'train epoch loss': train_loss,
                    'val epoch loss': val_loss, 
                    'val epoch bleu': val_bleu})
            
            if early_stop(bleu_history["val"], args.early_stopping, max):
                print("Early stopped.")
                break
    

                
def early_stop(loss_history, early_stop_k, min_or_max=min):
    if len(loss_history) < early_stop_k:
        return False
    idx_min_or_max = loss_history.index(min_or_max(loss_history))
    return len(loss_history) - 1 - idx_min_or_max >= early_stop_k
        
    
def run_batch_with_attention(phase, args, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, batch, device, encoder_embedding_dict, decoder_embedding_dict):

    assert phase in ("train", "val", "test"), "invalid phase"
    
    if phase == "train":
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
    
    loss = 0
    
    # TODO: it seems that currently batch size is always the same. Make sure to use the last batch
    target_sequence_length, batch_size = batch.trg[0].shape
    #print("This should be batch size:", batch_size) #
    
    position_ids = Variable(torch.LongTensor(range(0, batch.src[0].shape[0])))
    position_ids = position_ids.unsqueeze(1).repeat(1,batch_size).to(device)
    
    encoder_outputs = encoder(batch.src[0], position_ids)
    
    # Initiate decoder hidden state
    hidden, cell_state = decoder.random_init_hidden(device, batch_size)
    
    number_of_loss_calculation = 0

    # TEACHER FORCING
    # Feed all target sentences at once instead of reusing output as input
    # nice to look, and should be fast
    if phase == 'train' and np.random.random() < args.teacher_forcing:
        # this is needed when we are not using teacher forcing
        # To feed output of the decoder (the word with highest prob) into itself, has to use for loop, will be slower (is there faster alternative?)
        translated_tokens_list = []
        decoder_input = batch.trg[0][0,:]
        translated_tokens_list.append(decoder_input.unsqueeze(0))
        eos_encountered_list = [False]*batch_size
        i = 0
        
        # TODO: Even though the logic might be correct, the speed is extremely slow.
        while ((i+1 < target_sequence_length)  and (sum(eos_encountered_list) < batch_size)): # fix off-by-1 error, if any
            
            #logits = decoder(decoder_input)
            
            logits, decoder_attn, hidden, cell_state = decoder(
                hidden, cell_state, decoder_input, encoder_outputs
            )
            #loss += loss_function(decoder_output, batch.trg[i+1])
            #number_of_loss_calculation += 1
            logits = logits.unsqueeze(0)
            output = F.log_softmax(logits, dim=2)
            decoder_input = batch.trg[0][i+1,:]#torch.tensor([config.PAD_TOKEN]*batch_size, device=device, requires_grad=False)#.view(1,-1) # take care of different input shape
            for j in range(batch_size):   
                
                if not eos_encountered_list[j]:
                    # get index of maximum probability word
                    loss += loss_function(output[0,j,:].view(1,-1), batch.trg[0][i+1,j].view(1))
                    number_of_loss_calculation += 1
                
                    if batch.trg[0][i+1,j] == config.EOS_TOKEN: #?
                        # if EOS token, stop.
                        eos_encountered_list[j] = True
                    
            
            translated_tokens_list.append(decoder_input.unsqueeze(0))
            i += 1
    else:
        # this is needed when we are not using teacher forcing
        # To feed output of the decoder (the word with highest prob) into itself, has to use for loop, will be slower (is there faster alternative?)
        translated_tokens_list = []
        decoder_attn_list = []
        decoder_input = torch.tensor([config.SOS_TOKEN]*batch_size, device=device, requires_grad=False)#.view(1,-1) # take care of different input shape
        translated_tokens_list.append(decoder_input.unsqueeze(0))
        eos_encountered_list = [False]*batch_size
        i = 0
        # TODO: Even though the logic might be correct, the speed is extremely slow.
        while ((i+1 < target_sequence_length) and (sum(eos_encountered_list) < batch_size)): # fix off-by-1 error, if any
            
            logits, decoder_attn, hidden, cell_state = decoder(
                hidden, cell_state, decoder_input, encoder_outputs
            )
            decoder_attn_list.append(decoder_attn.detach())
            logits = logits.unsqueeze(0)
            output = F.log_softmax(logits, dim=2)
            decoder_input = torch.tensor([config.PAD_TOKEN]*batch_size, device=device, requires_grad=False)
            for j in range(batch_size):   
                
                if not eos_encountered_list[j]:
                    # get index of maximum probability word
                    max_index = output[0,j].max(0)[1].detach()
                    decoder_input[j] = max_index
                    loss += loss_function(output[0,j,:].view(1,-1), batch.trg[0][i+1,j].view(1))
                    number_of_loss_calculation += 1
                
                    if max_index == config.EOS_TOKEN or batch.trg[0][i+1,j] == config.EOS_TOKEN: #?
                        # if EOS token, stop.
                        eos_encountered_list[j] = True
                    
            
            translated_tokens_list.append(decoder_input.unsqueeze(0))
            i += 1
    
        

    if phase == "train":
        loss.backward() # this should calculate gradient for both encoder and decoder

        nn.utils.clip_grad_norm_(encoder.parameters(), args.clip)
        nn.utils.clip_grad_norm_(decoder.parameters(), args.clip)

        if (args.encoder_word_embedding is not None) and (not args.freeze_all_words):
            encoder.word_embedding.weight.grad.data[encoder_embedding_dict['oov_indices'],:].fill_(0)
        if (args.decoder_word_embedding is not None) and (not args.freeze_all_words):
            decoder.embedding.weight.grad.data[decoder_embedding_dict['oov_indices'],:].fill_(0)

        decoder_optimizer.step() # does it really matter which one takes step first?
        encoder_optimizer.step()

        # do not return translation output when training
        translation_output = None
        decoder_attn_list = None
    
    # if validation, test: output translated sentences as well
    else:
        translation_output = torch.cat(translated_tokens_list, dim=0)
        
    return loss.item() / number_of_loss_calculation, translation_output, decoder_attn_list
    
    
def train_and_val(args, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, device, epoch_idx, train_data, val_data, trg, encoder_embedding_dict, decoder_embedding_dict):
    
    assert args.attention, "if using cnn encoder, attention must be true"
    run_batch_func = run_batch_with_attention
        
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
        loss, _, _ = run_batch_func(
            "train",
            args,
            encoder,
            decoder, 
            encoder_optimizer, 
            decoder_optimizer, 
            loss_function,
            train_batch,
            device, 
            encoder_embedding_dict, #####
            decoder_embedding_dict#####
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
        loss, translation_output, _ = run_batch_func(
            "val",
            args,
            encoder,
            decoder, 
            encoder_optimizer, 
            decoder_optimizer, 
            loss_function,
            val_batch,
            device, 
            encoder_embedding_dict, #####
            decoder_embedding_dict#####
        )
        #translation_output = indices, N x B
        #todo: 1. check if trg.vocab.itos is pass in this function. 2.!reference! 
        val_reference = []
        for each in val_batch.idx:
            val_reference.append(" ".join(val_iter.dataset[each].trg))
        val_bleu = bleu(trg.vocab.itos, translation_output, val_reference)
        val_bleu_list.append(val_bleu)
        val_loss_list.append(loss)
        if i % args.print_every == 0:
            print("val, epoch: {}, step: {}, average loss for current epoch: {}, batch loss: {}, average bleu for current epoch: {}, batch bleu: {}".format(
                epoch_idx, i, np.mean(val_loss_list), loss, np.mean(val_bleu_list), val_bleu))
        
    print("val done. epoch: {}, average loss for current epoch: {}, average bleu for current epoch: {}".format(
        epoch_idx, np.mean(val_loss_list), np.mean(val_bleu_list)))
    
    return np.mean(train_loss_list), np.mean(val_loss_list), np.mean(val_bleu_list)
        
def test(args, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, device, i, test_data, trg):
    if args.attention:
        run_batch_func = run_batch_with_attention
    else:
        run_batch_func = run_batch
    
    test_iter = data.BucketIterator(
        dataset=test_data, 
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

    # turn off dropout
    encoder.eval()
    decoder.eval()
    
    test_loss_list = []
    test_bleu_list = []
    test_reference_list, translation_output_list = [], []
    test_source_list = []
    attention_lists = []
    for i, test_batch in enumerate(iter(test_iter)):
        #val_batch.trg: size N x B
        loss, translation_output, attention_list = run_batch_func(
            "test",
            args,
            encoder,
            decoder, 
            encoder_optimizer, 
            decoder_optimizer, 
            loss_function,
            test_batch,
            device, 
            encoder_embedding_dict, #####
            decoder_embedding_dict#####
        )
        #translation_output = indices, N x B
        #todo: 1. check if trg.vocab.itos is pass in this function. 2.!reference! 
        test_reference = []
        test_source = []
        for each in test_batch.idx:
            test_reference.append(" ".join(test_iter.dataset[each].trg))
            test_source.append(" ".join(test_iter.dataset[each].src))
        test_bleu = bleu(trg.vocab.itos, translation_output, test_reference)
        test_reference_list.append(test_reference)
        translation_output_list.append(detok(translation_output, np.array(trg.vocab.itos)))
        test_source_list.append(test_source)
        test_bleu_list.append(test_bleu)
        test_loss_list.append(loss)
        attention_lists.append(attention_list)
        if i % args.print_every == 0:
            print("test, step: {}, average loss for current epoch: {}, batch loss: {}, average bleu for current epoch: {}, batch bleu: {}".format(
                i, np.mean(test_loss_list), loss, np.mean(test_bleu_list), test_bleu))
        
    print("test done. average loss for current epoch: {}, average bleu for current epoch: {}".format(
        np.mean(test_loss_list), np.mean(test_bleu_list)))
    
    return np.mean(test_loss_list), np.mean(test_bleu_list), test_source_list, test_reference_list, translation_output_list, attention_lists


def cnn_encoder_decoder_argparser():
    # TODO: set min max size for vocab
    # TODO: specify optimizer
    # TODO: set max length of sentences
    
    # parser = argparse.ArgumentParser(description='Run Tests')
    parser = HyperOptArgumentParser(description='Run Tests', strategy='grid_search')
    
    parser.add_argument('--name', help="experiment name", default="rnn_encoder_decoder")    
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
    parser.add_argument("--kernel_size", help="Kernel size of cnn layers in encoder", type=int, default=3)
    parser.add_argument("--num_encoder_layers", help="Number of cnn layers in encoder", type=int, default=1)    
    parser.add_argument("--num_decoder_layers", help="Number of rnn layers in encoder", type=int, default=1)
    parser.add_argument("--early_stopping", help="Stop if validation does not improve", type=int, default=10)
    parser.add_argument('--l2_penalty', help="L2 pelnalty coefficient in optimizer", type=float,  default=0) #1e-06
    parser.add_argument('--clip', help="clip coefficient in optimizer", type=float,  default=1)
    parser.add_argument('--teacher_forcing', help="probability of performing teacher forcing", type=float,  default=0.5)
    parser.add_argument("--max_sentence_length", help="maximum sentence length", type=int, default=50)
    parser.add_argument('--split_chinese_into_characters', help="Split chinese into characters", action="store_true")
    parser.add_argument("--min_freq", help="Vocabulary needs to be present at least this amount of time", type=int, default=3)
    parser.add_argument("--max_vocab_size", help="At most n vocaburaries are kept in the model", type=int, default=100000)
    parser.add_argument("--source_lang", help="Source language (vi, zh)", default="zh")
    parser.add_argument("--logs_path", help="Path to save training logs", type=str, default="../training_logs/")
    parser.add_argument("--model_weights_path", help="Path to save best model weights", type=str, default="../model_weights/")
    parser.add_argument('--save_all_epoch', help="save all epoch", action="store_true")
    parser.add_argument('--relu', help="use relu in decoder after embedding", action="store_true")
    parser.add_argument('--attention', help="use luong attention decoder", action="store_true")
    parser.add_argument("--attn_model", help="dot, general, or concat", default='general')
    parser.add_argument("--preprocess_version", help="Version of preprocessing", type=int, default=2)
    parser.add_argument('--freeze_all_words', help="freeze word embedding and use character embedding from ELMo", action="store_true")
    parser.add_argument("--encoder_word_embedding", help="Word embedding weights file", default=None)
    parser.add_argument("--decoder_word_embedding", help="Word embedding weights file", default=None)
    return parser

if __name__ == '__main__':
    parser = cnn_encoder_decoder_argparser()
    args = parser.parse_args()
    run(args)
