# **Predicting the coding potential of** **sORFs in plant lncRNAs with pretrained BERT model** 
The codes and data here are used to predict the coding potential of lncRNA-sORFs. It will give researchers useful guidelines to discover peptides.

### Data folder:
contains pretraining samples.  
### LSCPP_BERT.bin:
is a model file.

### LSCPP_BERT.py:
You can run this file to test.

In line 88, you can change the path of the test file for testing your own data.

In line 92, this is the path of model file.

## Library dependency:
Based on python3.6  
Python modules:  

```
numpy
torch
multiprocessing
pandas
os
random
math
```
will be used. 