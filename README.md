# NTRD
This repository is the Pytorch implementation of our paper "**Learning Neural Templates for Recommender Dialogue System**" in **EMNLP 2021**.



In this paper, we introduce NTRD, a novel recommender dialogue system (i.e., conversational recommendation system) framework that decouples the dialogue generation from the item recommendation via a two-stage strategy. Our approach makes the recommender dialogue system more flexible and controllable. Extensive experiments show our approach significantly outperforms the previous state-of-the-art methods.

# Dependencies
pytorch==1.3.0, torch_geometric==1.3.2



the required data **word2vec_redial.npy** can be produced by the function ```dataset.prepare_word2vec()``` or directly download from the [Google Drive](https://drive.google.com/file/d/1BzwGgbUBilaEZXAu7e1SlvxSwcAfVe2w/view?usp=sharing) shared by [KGSF](https://github.com/RUCAIBox/KGSF).

# Run
Run the script below to pre-train the recommender module. It would converge after 3 epochs pre-training and 3 epochs fine-tuning.

```python
python run.py
```

Then, run the following script to train the seq2seq dialogue task. Transformer model is difficult to coverge, so the model need many of epochs to covergence. Please be patient to train this model.

```python
python run.py --is_finetune True
```

The model will report the result on test data automatically after covergence.

# Citation

If you find this paper helps your research, please kindly consider citing our paper in your publications.

```bibtex
@inproceedings{liang2021learning,
  title={Learning Neural Templates for Recommender Dialogue System},
  author={Liang, Zujie and Hu, Huang and Xu, Can and Miao, Jian and He, Yingying and Chen, Yining and Geng, Xiubo and Liang, Fan and Jiang, Daxin},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2021}
}
```

# Acknowledgment

This codebase is implemented based on [KGSF](https://github.com/RUCAIBox/KGSF). Many thanks to the authors for their efforts.
