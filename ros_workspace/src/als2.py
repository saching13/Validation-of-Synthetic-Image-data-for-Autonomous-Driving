import os
import numpy as np
from pyspark.sql import SparkSession
from pyspark import SparkContext

from pyspark.ml.recommendation import ALS

import math
import pyspark.sql
from pyspark.sql.functions import udf
from pyspark.sql.types import *
import pyspark.sql.functions as func
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as sf


spark = SparkSession \
    .builder \
    .appName("Recom") \
    .config("spark.recom.demo", "4") \
    .getOrCreate()
# lambda word: (word, 1)
sc = spark.sparkContext


seed = 5 #Random seed for initial matrix factorization model. A value of None will use system time as the seed.
iterations = 30
regularization_parameter = 0.05 #run for different lambdas - e.g. 0.01
rank = 40 #number of features



# Load ratings
schema = StructType([
    StructField("asin", IntegerType()),
    #StructField("helpful", ArrayType(IntegerType())),
    StructField("helpful", StringType()),
    StructField("overall", FloatType()),
    StructField("reviewText", StringType()),
    StructField("reviewTime", StringType()),
    StructField("reviewerID", IntegerType()),
    StructField("reviewerName", StringType()),
    StructField("summary", StringType()),
    StructField("unixReviewTime", LongType())
                ])


ratings_df = spark.read \
                  .format("csv") \
                  .option("header", "true") \
                  .option("inferSchema", "true") \
                  .option("escape","\"") \
                  .load("training1.csv")
rating_No_Null = ratings_df

input_df = rating_No_Null.select('asin','reviewerID','overall')

input_df= input_df.withColumnRenamed('reviewerID','userId')\
                  .withColumnRenamed('asin','BookID')\
                  .withColumnRenamed('overall','rating')


avg_rating = input_df.groupBy('userId').agg({'rating':'avg'})
            

avg_df = input_df.join(avg_rating, input_df["userId"] == avg_rating["userId"],how='left').drop(input_df.userId)

meaned_input_data = avg_df.withColumn('mean_rating',avg_df['rating']-avg_df['avg(rating)']).select('userId','BookID','mean_rating')




als = ALS(rank=rank, maxIter=iterations, regParam=regularization_parameter, userCol="userId", itemCol="BookID", ratingCol="mean_rating")


model = als.fit(meaned_input_data)


#predictions = model.transform(validationData)

#evaluator = RegressionEvaluator(metricName="rmse", labelCol="mean_rating",predictionCol="prediction")
#rmse = evaluator.evaluate(predictions)



#for rank in ranks:
# model = ALS.train(trainingData, rank, seed=seed, iterations=iterations,
#                     lambda_=regularization_parameter)

# predictions = model.predictAll(validation_for_predict.rdd).map(lambda r: ((r[0], r[1]), r[2]))
# rates_and_preds = validationData.rdd.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
# error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean()) # RMSE Error

# #print ('For rank',rank, "the RMSE is ", error)
# if error < min_error:
#     min_error = error
#     best_rank = rank

#print ("The best model was trained with rank", best_rank)

#output = "\n error on validation: "+ str(rmse) 



#predictions_test = model.transform(testData)
#rmse_test = evaluator.evaluate(predictions_test)






spark_testing_df = spark.read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("test_with_asin_reviewerID.csv")

spark_testing_df= spark_testing_df.withColumnRenamed('reviewerID','userId')\
                  .withColumnRenamed('asin','BookID')

output_predictions = model.transform(spark_testing_df)
output_predictions = output_predictions.join(avg_rating, output_predictions["userId"] ==\
                                        avg_rating["userId"],how='left').drop(output_predictions.userId)


output_predictions = output_predictions.withColumn('prediction',output_predictions['prediction']+output_predictions['avg(rating)']).select('userId','BookID','prediction')


op = output_predictions.withColumn('key',sf.concat(sf.col('userId'),sf.lit('-'), sf.col('BookId')))\
.select('key','prediction').withColumnRenamed('prediction','overall')


op.coalesce(1).write.option("header", "true")\
  .option("delimiter",',').csv('output6.csv')
