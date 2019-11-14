# Bidirectional Context Aware Hierarchical Attention Network for Document Understanding

Code for the paper [Bidirectional Context-Aware Hierarchical Attention Network for Document Understanding](https://arxiv.org/abs/1908.06006)

<img src="https://github.com/JbRemy/Cahan/blob/master/figures/CAHAN.png" width="100"> 

## Abstract

The Hierarchical Attention Network (HAN) has made great strides, but it suffers a major limitation: at level 1, each sentence is encoded in complete isolation. In this work, we propose and compare several modifications of HAN in which the sentence encoder is able to make context-aware attentional decisions (CAHAN). Furthermore, we propose a bidirectional document encoder that processes the document forwards and backwards, using the preceding and following sentences as context. Experiments on three large-scale sentiment and topic classification datasets show that the bidirectional version of CAHAN outperforms HAN everywhere, with only a modest increase in computation time. While results are promising, we expect the superiority of CAHAN to be even more evident on tasks requiring a deeper understanding of the input documents, such as abstractive summarization.

![alt text](https://github.com/JbRemy/Cahan/blob/master/figures/yelp_tricky_baseline_cropped.png | width = 100) ![alt text](https://github.com/JbRemy/Cahan/blob/master/figures/yelp_tricky_sum_bidir_cropped.png | width = 100)
<p style="text-align: center;">HAN (left) vs CAHAN (right) on an example extracted from the Yelp dataset.</p>

![alt text](https://github.com/JbRemy/Cahan/blob/master/figures/amazon_baseline_cropped.png | width = 100) ![alt text](https://github.com/JbRemy/Cahan/blob/master/figures/amazon_mean_bidir_cropped.png | width = 100)
<p style="text-align: center;">HAN (left) vs CAHAN (right) on a motivational example.</p>

## Description :
 
* `V1`: Weights and records of the experiments.
** `weights` - Initial weights
** `baseline` - Trained weights and results on the baseline
** `agg=sum_bidir=True_discount=1_cutgradient=False` - Trained weights and results for the experiment with summed attention, bidirectional contextual attention and discount factor = 1.

* `code`: All the scripts needed to run the experiments. To run the experiments you can run the `main_*` scripts.

## Requirements:

This repository was developped using `python 3.6` and `Cuda 9.0`. 
Requirements are contained in the `requirements.txt` file.

## Citing this work:

If you use this code or build up on the idea proposed in the paper, please cite it as:

`
@article{remy2019bidirectional,
  title={Bidirectional Context-Aware Hierarchical Attention Network for Document Understanding},
  author={Remy, Jean-Baptiste and Tixier, Antoine Jean-Pierre and Vazirgiannis, Michalis},
  journal={arXiv preprint arXiv:1908.06006},
  year={2019}
}
`

## Authors:

If you liked this work you can follow the authors:

* Jean-Baptiste Remy [![github](https://github.com/JbRemy/Cahan/blob/master/figures/logos/GitHub-Mark-64px.png)](https://github.com/JbRemy) [![scholar](https://github.com/JbRemy/Cahan/blob/master/figures/logos/Google_Scholar_logo_2015.png)](https://scholar.google.com/citations?user=kZNC1yAAAAAJ&hl=fr)
* Antoine Jean-Pierre Tixier [![github](https://github.com/JbRemy/Cahan/blob/master/figures/logos/GitHub-Mark-64px.png)](https://github.com/Tixierae) [![scholar](https://github.com/JbRemy/Cahan/blob/master/figures/logos/Google_Scholar_logo_2015.png)](https://scholar.google.fr/citations?user=mGLmAh0AAAAJ&hl=fr)



