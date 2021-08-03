package com.spark.benchmark.example.userclicks

import org.apache.spark.streaming.Seconds
import scala.collection.mutable.HashMap

/**
  * Application 8.
  *
  * Selects users that have performed more than 300 clicks within a 10-second jumping window.
  *
  * All results are saved to HDFS.
  *
  * SELECT userId, COUNT(userId) as count
  * FROM UserClicks [RANGE 10 SLIDE 10]
  * GROUP BY userId
  * HAVING count > 300;
  */
object App8 extends App {

  def main(args: Array[String]): Unit = {

    // set application configuration and parameters
    val conf = new HashMap[String, String]()
    conf("appName") = "benchmark_userclicks_exp11"
    conf("count") = "300"
    custom(args, conf)

    // setup spark streaming context
    val ssc = getSsc(conf)
    val dstream = getDstream(ssc, conf)

    dstream
      .map(line => (line.split(" - - ")(0), 1))
      .reduceByKeyAndWindow((a: Int, b: Int) => (a + b), Seconds(10), Seconds(10))
      .filter(_._2 > Integer.parseInt(conf("count")))
      // save all results to HDFS
      .saveAsTextFiles(conf("savePath"))

    ssc.start()
    ssc.awaitTermination()
  }
}
