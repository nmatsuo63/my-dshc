## 2022/9/10追記
下記を参考にmlflowを追加

https://data-campus.ai/announcements/va8y3ip34jj4ncar?type=text

# 使い方

- パスワード設定
  - `docker-compose.yml`のJUPYTER_PASSを変更(default: password)

- 起動

```
docker-compose up -d
```

- その他

hostからコンテナに渡したいデータがある場合は./dataに格納。  
host側の`./ipynb/`は`/home/jovyan/work`にマウントされている。
`./data`と`./ipynb`は適切にpermissionをつけてください。


**`./data`と`./ipynb`は`gitignore`に含まれているので注意**

# 中身

jupyter/datascience-notebookベース
追加で必要なものがあれば`./jupyter/requirements.txt`に追記

# 参考：jupyter-notebook環境立ち上げセット

https://zenn.dev/k8shiro/articles/my-docker-jupyterlab

- port: 8888
- 初期パスワード：password
