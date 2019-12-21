## Measuring and Improving the Use of Graph Information in Graph Neural Networks

#### Authors: [Yifan Hou](https://yifan-h.github.io/)(yfhou@cse.cuhk.edu.hk), Jian Zhang (jzhang@cse.cuhk.edu.hk), [James Cheng](https://www.cse.cuhk.edu.hk/~jcheng/) (jcheng@cse.cuhk.edu.hk), Kaili Ma (klma@cse.cuhk.edu.hk), Richard T. B. Ma (tbma@comp.nus.edu.sg), Hongzhi Chen (hzchen@cse.cuhk.edu.hk), Ming-Chang Yang (mcyang@cse.cuhk.edu.hk)

### Overview

Graph neural networks (GNNs) have been widely used for representation learning on graph data. However, there is limited understanding on how much performance GNNs actually gain from graph data. 

This paper introduces a context-surrounding GNN framework and proposes two smoothness metrics to measure the quantity and quality of information obtained from graph data. A new, improved GNN model, called CS-GNN, is then devised to improve the use of graph information based on the smoothness values of a graph. CS-GNN is shown to achieve better performance than existing methods in different types of real graphs. 

Please see the [paper](https://openreview.net/forum?id=rkeIIkHKvS) for more details.

*Note:* If you make use of this code or the CS-GNN model in your work, please cite the following paper:

    @inproceedings{
    hou2020measuring,
    title={Measuring and Improving the Use of Graph Information in Graph Neural Networks},
    author={Yifan Hou and Jian Zhang and James Cheng and Kaili Ma and Richard T. B. Ma and Hongzhi Chen and Ming-Chang Yang},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=rkeIIkHKvS}
    }

### Requirements

Recent versions of pytorch, numpy, sklearn, tqdm, networkx, pygsp, and pandas are required. You can install those required packages using the following command:

	$ pip install -r requirements.txt

Deep Graph Library (DGL) is also required. You can install it with the tutorial in [dgl](https://docs.dgl.ai/install/index.html).

### How to run

You can run the .py file directly, using the following command to run the model:

	$ python ./src/train.py

#### Input format

* <train_prefix>-G.json -- A networkx-specified json file describing the input graph. Nodes have 'val' and 'test' attributes specifying if they are a part of the validation and test sets. When node's 'val' and 'test' are all 'True', it means this node is unlabeled.
* <train_prefix>-id_map.json -- A json-stored dictionary mapping the graph node ids to consecutive integers.
* <train_prefix>-class_map.json -- A json-stored dictionary mapping the graph node ids to classes.
* <train_prefix>-feats.npy --- A numpy-stored array of node features; ordering given by id_map.json.
* <train_prefix>-feats_t.npy [optional] --- A numpy-stored array of node topology features; ordering given by id_map.json. Can be re-caculated by removing the file.

### Academic Paper

[**ICLR 2020**] **Measuring and Improving the Use of Graph Information in Graph Neural Networks**, Yifan Hou, Jian Zhang, James Cheng, Kaili Ma, Richard T. B. Ma, Hongzhi Chen, Ming-Chang Yang.

### Acknowledgement
We owe many thanks to Ng Hong Wei for cleaning the BGP data.
