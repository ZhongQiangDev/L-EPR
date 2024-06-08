# P-NPR: Practical Neural Program Repair
This is the artifact of our paper.
## Dependency
* Python 3.6.9
* Pytorch >=1.11.0
* Transformers >= 4.18.0
* Defects4J
## Content
* P-EPR: some script files to execute P-EPR
* P-NPR:
  * dataset: some csv files of datasets used in our experiments
  * bert.py: implementation of BERT model
  * codebert.py: implementation of CodeBERT model
  * graphcodebert.py: implementation of GraphCodeBERT model
  * run.py: a script to execute P-NPR
  * train_eval.py: a script to define the training and testing of P-NPR
  * unixcoder.py: implementation of Unixcoder model
  * utils.py: a script including some functions used in run.py and train_eval.py
* Results: a folder containing experimental results on Defects4J and Test-5000
* multi-label_dataset_meta.xlsx: an excel sheet containing the meta data of the multi-label dataset we built
