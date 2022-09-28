#!/bin/bash
# common
# kadai_tutorial.ipynb
# katakana_model.pickle
# submit_katakana.ipynb
# train_4_7.ipynb
# util.py
# 1_submitではなく1_compress, 2_submitとしたい
mkdir 1_compress
cp -r common 1_compress 
cp -r kadai_tutorial.ipynb 1_compress 
cp -r katakana_model.pickle 1_compress 
cp -r submit_katakana.ipynb 1_compress 
cp -r train_4_7.ipynb 1_compress 
cp -r util.py 1_compress 

mv ./1_compress/train*.ipynb ./1_compress/train.ipynb
zip -r dl_tokyo_m_2022_submit_katakana_MATSUO_NAOYA_2022yymm_n.zip 1_compress 

mv dl_tokyo_m_2022_submit_katakana_MATSUO_NAOYA_2022yymm_n.zip ~/Downloads/
rm -rf 1_compress