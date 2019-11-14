# Bidirectional Context Aware Hierarchical Attention Network for Document Understanding

Code for the paper [Bidirectional Context-Aware Hierarchical Attention Network for Document Understanding](https://arxiv.org/abs/1908.06006)

## Abstract

The Hierarchical Attention Network (HAN) has made great strides, but it suffers a major limitation: at level 1, each sentence is encoded in complete isolation. In this work, we propose and compare several modifications of HAN in which the sentence encoder is able to make context-aware attentional decisions (CAHAN). Furthermore, we propose a bidirectional document encoder that processes the document forwards and backwards, using the preceding and following sentences as context. Experiments on three large-scale sentiment and topic classification datasets show that the bidirectional version of CAHAN outperforms HAN everywhere, with only a modest increase in computation time. While results are promising, we expect the superiority of CAHAN to be even more evident on tasks requiring a deeper understanding of the input documents, such as abstractive summarization.

![alt text](https://github.com/JbRemy/cahan/figures/yelp_tricky_baseline_cropped.pdf)
![alt text](https://github.com/JbRemy/cahan/figures/yelp_tricky_sum_bidir_cropped.pdf)




