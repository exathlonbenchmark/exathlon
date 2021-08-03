package com.spark.benchmark.example.userclicks

import org.apache.spark.streaming.Seconds
import scala.collection.mutable.HashMap

/**
  * Application 10.
  *
  * Selects URLs visited more than 500 times within a 10-second jumping window.
  *
  * All results are saved to HDFS.
  *
  * SELECT url, COUNT(url) as count
  * FROM UserClicks [RANGE 10 SLIDE 10]
  * GROUP BY url
  * HAVING count > 500;
  */
object App10 extends App {

  def main(args: Array[String]): Unit = {

    // set application configuration and parameters
    val conf = new HashMap[String, String]()
    conf("appName") = "benchmark_userclicks_exp14"
    conf("count") = "500"
    custom(args, conf)

    // setup spark streaming context
    val ssc = getSsc(conf)
    val dstream = getDstream(ssc, conf)

    dstream
      .map(line => (line.split(" - - ")(2), 1))
      .reduceByKeyAndWindow((x: Int, y: Int) => x + y, Seconds(Integer.parseInt(conf("windowSec"))), Seconds(10))
      .filter(_._2 > Integer.parseInt(conf("count")))
      // save all results to HDFS
      .saveAsTextFiles(conf("savePath"))

    ssc.start()
    ssc.awaitTermination()
  }
}
