import time

iterno = 10

if __name__ == '__main__':
    from pyspark import SparkContext, SparkConf, SQLContext
    from pyspark.mllib.util import MLUtils
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.classification import LogisticRegressionWithSGD
    from pyspark.mllib.evaluation import BinaryClassificationMetrics

    conf = SparkConf().setAppName("lr-SGD")  # .setMaster('yarn-client') #.setMaster("local[*]")
    sc = SparkContext(conf=conf)

    sqlContext = SQLContext(sc)

    higgs_file_path = "hdfs://m7-model-hdp01:8020/user/root/4thModelData/higgs/libsvmformat/train/train.libsvm"
    higgs_data = MLUtils.loadLibSVMFile(sc, higgs_file_path)

    begin = time.time()
    global iterno
    lr_model = LogisticRegressionWithSGD.train(higgs_data, iterations=iterno)
    end = time.time()

    print("训练时间: %.3f秒" % (end - begin))
    print("训练轮数： " + str(iterno))
    print("训练集AUC: " + str(lr_model.summary.areaUnderROC))

    higgs_test_path = "hdfs://m7-model-hdp01:8020/user/root/4thModelData/higgs/libsvmformat/test/test.libsvm"
    higgs_test_data = MLUtils.loadLibSVMFile(sc, higgs_test_path)

    prediction = lr_model.predict(higgs_test_data.map(lambda x: x.features))
    predictionAndLabel = prediction.zip(higgs_test_data.map(lambda x: x.label))

    metrics = BinaryClassificationMetrics(predictionAndLabel)

    print("Test areaUnderPR = " + str(metrics.areaUnderPR))
    print("Test areaUnderROC = " + str(metrics.areaUnderROC))

