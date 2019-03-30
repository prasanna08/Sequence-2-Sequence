# NLU Assignment 2
This code implements Sequence to Sequence architecture with Various attention mechanism.

## Prerequisites

### Requirements
Following dependencies are required to run this code.
* pytorch
* torchtext
* nltk
* numpy

Alternatively, you can use following commands to install necessary libraries.
```
pip install -r requirements.txt
``` 

### Merge
Run following command to merge splitted models into a single file before
training or testing of models.

```
bash scripts/merge.sh
```

## Training model from scratch.
To train model from scratch use `Trainer.py`. Run following command:
``` 
python Trainer.py [-h] [--embedding-dim=300] [--hidden-size=256] [--num-layers=2] [--iters=10000] [--bidirectional=True] [--batch-size=32] [--summary-steps=10] [--checkpoint-steps=500] [--attention=DotProductAttn] [--language=de]
```

Arguments:
* `--attention`: Attention mechanism to use. It can be either DotProductAttn, AdditiveAttn or MultiplicativeAttn. The model will use self attention by default.
* `--language`: Can be either `de` or `hi`. Source language is always english. This specifies target language.


## Model evaluation.
To evaluate model use following command:
```
python Inference.py  [--embedding-dim] [--hidden-size] [--num-layers] [--bidirectional] [--attention] [--model] [--infile] [--tgtfile] [--outfile] [--language]
```
Arguments:
*  `--model`: Specify checkpoint file which will be used to restore model. In addition to model file, the architecture of
checkpoint model needs to be specified using `hidden-size`, `embedding-dim`, `num-layers`, `bidirectional`, `attention` arguments.
*  `--infile`: Specify input sentences file.
*  `--tgtfile`: Sepcify reference translations file.
*  `--outfile`: The translated output by model will be stored in outfile.

To Evaluate model for English-Germal translation use following command:
```
python Inference.py  --embedding-dim=300 --hidden-size=256 --num-layers=2 --bidirectional=True --attention=MultiplicativeAttn --model=./models/ENDE.pt [--infile] [--tgtfile] [--outfile] --language=de
```

To Evaluate model for English-Hindi translation use following command:
```
python Inference.py  --embedding-dim=300 --hidden-size=256 --num-layers=2 --bidirectional=True --attention=MultiplicativeAttn --model=./models/ENHI.pt [--infile] [--tgtfile] [--outfile] --language=hi
```
