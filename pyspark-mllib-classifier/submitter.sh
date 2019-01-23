#!/usr/bin/env bash

export JAVA_HOME=/usr/local/jdk1.8.0_112/
export HADOOP_HOME=/home/work/hadoop-client-m7-model-inf01/
export SPARK_HOME=/home/huanggeyang/local/spark/spark
export HADOOP_CONF_DIR=${HADOOP_HOME}/etc/hadoop

##local

spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --executor-cores 32 \
    --num-executors 2 \
    --executor-memory 260G \
    --queue default \
    --files ./logistic-regression-SGD.py \
    --archives ./py3spark_env.tar.gz \
    --conf spark.pyspark.python=./py3spark_env.tar.gz/py3spark_env/bin/python3.7 \
    ./logistic-regression-SGD.py
