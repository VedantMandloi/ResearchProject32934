# Research Project - 32934

* This repository contains the source code for experiments and prototype for the Research Project.

* The experimentation details can be found in **src** folder and prototype in **api** folder. The folder names inside should be self explanatory.

* All the data files need to be placed in the **Data** folder.

## Important Note:
**There is a tensorflow version conflict between the prototype and experiments. Install in different virtual environments.**

## Important details about BERT Prototype:

* The library Bert-as-service was used to obtain embeddings for text.

* It can work on just a CPU however it's not been tested in the prototype environment and as a result the code has been commented out. A GPU with > 8 GB VRAM is required for fast encoding.

* Bert-as-service loads a saved BERT model and starts an encoding server which can be sent text to encode. 

* The command used to run the bert server is 
`bert-serving-start -pooling_layer -4 -3 -2 -model_dir="PATH_TO_SAVED_BERT_MODEL" -max_seq_len=256`

* The model can be downloaded from https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip and placed in the appropriate folder in **api**

* The saved models have not been included but they can be obtained from the respective notebooks in **src** folder.

