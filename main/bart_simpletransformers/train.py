import logging

import jsonlines
import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
import torch
import os
import parsers

import sys
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler
 
default_collate_func = dataloader.default_collate

def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)
 
setattr(dataloader, 'default_collate', default_collate_override)
 
for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]

os.environ["TOKENIZERS_PARALLELISM"] = "false"

cuda_available = torch.cuda.is_available()

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

args=parsers.TrainParser().parse_args()
print(args)
train_data_path=args.train_data_path
eval_data_path=args.eval_data_path


def getData(path):
    res=[]
    try:
      with open(path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
              res.append([item['input'],item['output'][0]])
    except:
      try:
        with open(path.replace('seq','seq_1'), "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                res.append([item['input'],item['output'][0]])
        with open(path.replace('seq','seq_2'), "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                res.append([item['input'],item['output'][0]])
      except:
        with open(path.replace('.jsonl','_1.jsonl'), "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                res.append([item['input'],item['output'][0]])
        with open(path.replace('.jsonl','_2.jsonl'), "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                res.append([item['input'],item['output'][0]])
    return res



train_data = getData(train_data_path)
print(train_data[0])
train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

eval_data = getData(eval_data_path)
eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

# Configure the model
model_args = Seq2SeqArgs()
model_args.num_train_epochs = args.num_train_epochs
model_args.evaluate_generated_text = args.evaluate_generated_text
model_args.evaluate_during_training = args.evaluate_during_training
model_args.evaluate_during_training_verbose = args.evaluate_during_training_verbose
model_args.save_model_every_epoch= args.save_model_every_epoch
model_args.train_batch_size=args.train_batch_size
model_args.learning_rate=args.learning_rate
model_args.max_length=args.max_length
model_args.eval_batch_size=args.eval_batch_size
model_args.max_seq_length=args.max_seq_length
model_args.n_gpu=args.n_gpu
model_args.evaluate_during_training_steps=30000
model_args.save_eval_checkpoints=False
model_args.output_dir=f'./model/{args.output}epochs{model_args.num_train_epochs}lr{model_args.learning_rate}batch{model_args.train_batch_size}/'

model_init=args.model

model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name=model_init,
    args=model_args,
    use_cuda=cuda_available,
)

model.train_model(train_df, eval_data=eval_df)
result = model.eval_model(eval_df)
