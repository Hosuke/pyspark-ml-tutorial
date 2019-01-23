# pyspark-ml-tutorial
An example to use pyspark to train a logistic regression classifier

关于提交pyspark到spark on yarn cluster上的环境指南
-
创建环境，包含所需依赖库
```bash
conda create --name py3spark_env --quiet --copy --yes python=3 numpy scipy pandas pyspark
```
然后进anaconda3/envs目录下，打包环境

```bash
tar -zvcf ./py3spark_env.tar.gz ./py3spark_env/
```

之后将py3spark_env.tar.gz放到提交脚本目录下，跑提交脚本