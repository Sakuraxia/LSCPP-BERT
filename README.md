# **A multi-granularity information-enhanced pre-training method for predicting the coding potential of sORFs in plant lncRNAs** 
The codes and data here are used to predict the coding potential of lncRNA-sORFs. It will give researchers useful guidelines to discover peptides.

### Data folder:
contains pretraining samples.  
### LSCPP_BERT.bin:
is a model file.  You need to download "LSCPP_BERT.bin" from (https://drive.google.com/file/d/1o7KZwG5fbGZd3K1LMYiD6qCOyOHEXU4m/view?usp=sharing) or (https://pan.baidu.com/s/18P3w7MQUBI49IEjCyf6C8Q?pwd=18p1). Then, you should move the file "LSCPP_BERT.bin" to the "model" folder

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
