import logging
import parsers
import jsonlines
import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

cuda_available = torch.cuda.is_available()

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


args=parsers.TestParser().parse_args()
test_data_path=args.test_data_path



def getData(path):
    res=[]
    truth=[]
    with open(path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            res.append(item['input'])
            truth.append([item['id'],item['output'][0]])
    return res,truth

test_data,truth=getData(test_data_path)
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
model_args.num_beams=args.num_beams

model_path=args.model_path
# Loading a saved model
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name=model_path,
    args=model_args,
)
output=args.output
# Use the model for prediction
with jsonlines.open(f'./result/{output}.jsonl','w') as f:
    ans=model.predict(
        test_data
    )
    for i in range(len(ans)):
        f.write({"truth":truth[i][1],"predict":ans[i],'id':truth[i][0]})