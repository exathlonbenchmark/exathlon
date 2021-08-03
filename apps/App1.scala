package com.spark.benchmark.example.userclicks

import scala.collection.mutable.HashMap

/**
  * Application 1.
  *
  * Counts the number of clicks performed by each user within the last received batch.
  *
  * SQL-like equivalent:
  *
  * SELECT userId, COUNT(userId)
  * FROM UserClicks
  * GROUP BY userId;
  */
object App1 extends App {

  def main(args: Array[String]): Unit = {

    // set application configuration
    val conf = new HashMap[String, String]()
    conf("appName") = "benchmark_userclicks_exp1"
    custom(args, conf)

    // get streaming context
    val ssc = getSsc(conf)
    val dstream = getDstream(ssc, conf)

    // flag variable used for saving the results for the first batch only
    var flag = true

    // count the number of
    dstream
      .map(line => (line.split(" - - ")(0), 1)).reduceByKey(_ + _)
      // save the results for the first batch only
      .filter(x => if (flag) { flag = false; true} else false)
      .saveAsTextFiles(conf("savePath"))
    ssc.start()
    ssc.awaitTermination()
  }
}
