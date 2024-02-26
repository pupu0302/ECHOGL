# ECHO-GL: Earnings Calls-driven Heterogeneous Graph Learning for Stock Movement Prediction

<!-- Abstract        -->

## Abstract

Stock movement prediction serves an important role in quantitative trading. Despite advances in existing models that enhance stock movement prediction by incorporating stock relations, these prediction models face two limitations, i.e., constructing either insufficient or static stock relations, which fail to effectively capture the complex dynamic stock relations because such complex dynamic stock relations are influenced by various factors in the ever-changing financial market. To tackle the above limitations, we propose a novel stock movement prediction model, ECHO-GL, based on stock relations derived from earnings calls. ECHO-GL not only constructs comprehensive stock relations by exploiting the rich semantic information in the earnings calls but also captures the movement signals between related stocks based on multimodal and heterogeneous graph learning. Moreover, ECHO-GL customizes learnable stock stochastic processes based on the post earnings announcement drift (PEAD) phenomenon to generate the temporal stock price trajectory, which can be easily plugged into any investment strategy with different time horizons to meet investment demands. Extensive experiments on two financial datasets demonstrate the effectiveness of ECHO-GL on stock price movement prediction tasks together with high prediction accuracy and trading profitability.

<!-- About this Repo  -->

## About This Repo

<!-- TODO:  xxx: data folder
            paper link -->

This repository includes:

1. code for the Earnings Calls-driven HeterOgeneous Graph Learning (ECHO-GL) model in our paper "ECHO-GL: Earnings Calls-driven Heterogeneous Graph Learning for Stock Movement Prediction."`<!--, [paper](paper link).-->`
2. constructed earnings call-driven heterogeneous graphs (E-Graph in our paper), which model the complex stock relations derived from earnings calls.

<!--
ECHO-GL to deeply model the complex stock relations in an earnings call-driven heterogeneous dynamic graph (termed as E-Graph) for better predicting stock movements.
-->

<!-- Environment        -->

## Environment

<!-- TODO: 环境要求 -->

Python version and packages required to install for executing the code.

```
Python >=3.8
PyTorch >=2.0.1
torchsde >= 0.2.5
```

<!-- Data Introduction -->

## Data

All data, including stock price data and constructed E-Graph, are under the [data](https://github.com/pupu0302/ECHOGL/tree/main/data) folder.

Note that, for pre-processed earnings call data, we adopted two widely studied earnings call datasets Qin's [[1]](https://aclanthology.org/P19-1038.pdf) and MAEC [[2]](https://dl.acm.org/doi/pdf/10.1145/3340531.3412879), both of which have provided pre-processed data.

### Stock price data

We collect dividend-adjusted closing prices from [Yahoo Finance](https://aclanthology.org/P19-1038.pdf)). Collected price data is under the [historical_price](https://github.com/pupu0302/ECHOGL/tree/main/data/historical_price) folder.

<!-- All data, including Sequential Data, Industry Relation, and Wiki Relation, are under the data folder. 
We use Qin's and MAEC as raw data

Processed data: xxx is the dataset used to conducted experiments in our paper. -->

<!-- ### Relation Data
The original industry relation and Wiki relation data used for E-Graph construction are under the [relation]() folder. -->

### E-Graph

In our paper, we introduce an earnings call-driven heterogeneous dynamic graph (termed as E-Graph) that portrays comprehensive stock relations in the current market.

E-Graph encompasses four types of nodes(stock price node(P), earnings call text sentence node(S), topic node(O), and entity node(E)) and four types of edges(P-S, S-O, S-E, and E-E).

The specific E-Graph construction algorithm has been shown in Section 4.1 in our paper.
The constructed E-Graph data is under the [E-Graph]() folder.

<!-- To get the relation data, run the following command:
```python
# TODO
# 参考写法：tar zxvf relation.tar.gz
``` -->

<!-- Code Introduction  -->

## Code

|     Script     |               Function               |
| :------------: | :----------------------------------: |
|   ECHO_GL.py   |            ECHO-GL model            |
|  container.py  | ECHO-GL model container for training |
| run_ECHO_GL.py |       Train a model of ECHO_GL       |

Note that, since ECHO-GL is implemented in an integrated quantitative system under development, the quantitative system cannot be open-sourced for the time being. Therefore, There is only the code of ECHO-GL's model, which can not run at present. I hope the code can help you better understand ECHO-GL's paper.

After the quantitative system development is completed, we will provide the complete code as soon as possible. I believe this code will come out in the near future.

<!--### Training
<!-- ### Pre-processing
### Training -->

<!-- Pre-processing 和 training 的写法↓  https://github.com/fulifeng/Temporal_Relational_Stock_Ranking/blob/master/README.md?plain=1 -->

<!-- | Script | Function |
| :-----------: | :-----------: |
| rank_lstm.py | Train a model of Rank_LSTM |
| relation_rank_lstm.py | Train a model of Relational Stock Ranking | -->

<!--### Training

| run_egraph.py | Train a model of ECHO-GL |
-->

<!-- Run Command   
## Run
To repeat the experiment, download the Qin's and MAEC earnings call dataset, and extract the file into the data folder.
### Qin's
```python
python experiments/run_egraph.py --data_name qin
```
### MAEC
```python
python experiments/run_egraph.py --data_name maec
```
     -->

<!--
参考：https://github.com/fulifeng/Temporal_Relational_Stock_Ranking/blob/master/README.md?plain=1
### NASDAQ
```
python relation_rank_lstm.py -rn wikidata -l 16 -u 64 -a 0.1
```

### NYSE
```
python relation_rank_lstm.py -m NYSE -l 8 -u 32 -a 10 -e NYSE_rank_lstm_seq-8_unit-32_0.csv.npy
```

to enable gpu acceleration, add the flag of:
```
-g 1
```
!>




<!-- Citation    
## Citation
If you use the code, please kindly cite the following paper: 
```
# TODO: publish 之后
```   -->

<!-- References           -->

## Citation

[1]: [Qin, Y.; and Yang, Y. What You Say and How You Say It Matters: Predicting Stock Volatility Using Verbal and Vocal Cues. ACL 2019](https://aclanthology.org/P19-1038.pdf)

[2] [Li, J.; Yang, L.; Smyth, B.; and Dong, R. Maec: A
Multimodal Aligned Earnings Conference Call Dataset for
Financial Risk Prediction. CIKM 2020](https://dl.acm.org/doi/pdf/10.1145/3340531.3412879)

<!-- Contact   
## Contact
<mengpuliu@zju.edu.cn>
 -->
