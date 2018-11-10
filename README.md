# neural_machine_trainslation_2018

## Sample script for training on vi-en
```bash
python3 rnn_encoder_decoder.py --source_lang=vi --data=../data/iwslt-vi-en --num_encoder_layers=1 --num_decoder_layers=1 --save_all_epoch --dropout=0.2
```
## Sample script for testing and seeing results
```python
parser = rnn_encoder_decoder.rnn_encoder_decoder_argparser()
args = parser.parse_args([])
args.source_lang = 'vi'
args.data = '../data/short-sentences-vi-en/'
args.num_encoder_layers = 2
args.num_decoder_layers = 2
args.test = True
args.model_weights_path = '../model_weights/short/3'
loss, bleu, test_source_list, test_reference_list, translation_output_list, attention_list = rnn_encoder_decoder.run(args)

for i in len(test_source_list):
  for triplet in zip(test_source_list[i], test_reference_list[i], translation_output_list[i]):
      print(triplet)
```

## Command line arguments
- --name
  - Name of the experiment, it will be used for save path of training log and trained model weights
- --test 
  - If this flag is present, do one test epoch. Otherwise, do train and validation
- --source_lang
  - Indicate source language (vi, zh) so that different preprocessing can be applied.
- --data 
  - Indicate data folder
- and other hyperparams of the model...
