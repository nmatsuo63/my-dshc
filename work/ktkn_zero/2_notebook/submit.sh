#!/bin/bash
mkdir 1_compress
cp -r common 1_compress 
cp -r katakana_model.pickle 1_compress 
cp -r submit_katakana.ipynb 1_compress 
cp -r util.py 1_compress 

zip -r dl_tokyo_m_2022_submit_katakana_MATSUO_NAOYA_2022yymm_n.zip 1_compress 

mv dl_tokyo_m_2022_submit_katakana_MATSUO_NAOYA_2022yymm_n.zip ~/Downloads/
rm -rf 1_compress