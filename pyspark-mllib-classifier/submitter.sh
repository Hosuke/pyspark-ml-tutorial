#!/usr/bin/env bash

##local

spark-submit --master yarn \
    --conf spark.yarn.naxAppAttempts=1 \
    --num-executors 5 \
    --queue pico \
    --conf spark.driver.maxResultSize=50G \
    --driver-memory 50G \
    --executor-memory 50G \
    --executor-cores 10 \
    --deploy-mode cluster \
    pySparkHDFS.py