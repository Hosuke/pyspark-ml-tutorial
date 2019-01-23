# -*- coding: UTF-8 -*-
# Huang Geyang(Hosuke) @ 2019.01.23

# import os
# import sys
# # The following is for specifying a Python version for PySpark. Here we
# # use the currently calling Python version.
# # This is handy for when we are using a virtualenv, for example, because
# # otherwise Spark would choose the default system Python version.
# os.environ['PYSPARK_PYTHON'] = sys.executable

import time

iterno = 10
# dataset file must be provided in libsvm format with [0,1] as label
higgs_file_path = "hdfs://higgs/libsvmformat/train/train.libsvm"
higgs_test_path = "hdfs://higgs/libsvmformat/test/test.libsvm"

if __name__ == '__main__':
    from pyspark import SparkContext, SparkConf, SQLContext
    from pyspark.mllib.util import MLUtils
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.classification import LogisticRegressionWithSGD
    from pyspark.mllib.evaluation import BinaryClassificationMetrics

    conf = SparkConf().setAppName("lr-SGD")  # .setMaster('yarn-client') #.setMaster("local[*]")
    sc = SparkContext(conf=conf)

    sqlContext = SQLContext(sc)

    higgs_data = MLUtils.loadLibSVMFile(sc, higgs_file_path)

    begin = time.time()

    lr_model = LogisticRegressionWithSGD.train(higgs_data, iterations=iterno)
    end = time.time()

    print("Training time: %.3fs" % (end - begin))
    print("Training pass: " + str(iterno))
    print("Time per pass: " + str(float(end-begin)/iterno))

    higgs_test_data = MLUtils.loadLibSVMFile(sc, higgs_test_path)

    prediction = lr_model.predict(higgs_test_data.map(lambda x: x.features))
    prediction = prediction.map(lambda x: float(x))
    predictionAndLabel = prediction.zip(higgs_test_data.map(lambda x: float(x.label)))

    metrics = BinaryClassificationMetrics(predictionAndLabel)

    print("Test areaUnderPR = " + str(metrics.areaUnderPR))
    print("Test areaUnderROC = " + str(metrics.areaUnderROC))

    sc.stop()

