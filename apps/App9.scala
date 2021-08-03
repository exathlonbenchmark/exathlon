package com.spark.benchmark.example.userclicks

import org.apache.spark.streaming.Seconds
import scala.collection.mutable.HashMap

/**
  * Application 9.
  *
  * Groups information by user within a 10-second jumping window.
  *
  * Apart from the windowing parameters, this application differs from 4 and 7 in saving all
  * the results to HDFS, and from 7 in not including the additional CPU pressure.
  *
  * SELECT userId
  * FROM UserClicks [RANGE 10 SLIDE 10]
  * GROUP BY userId;
  */
object App9 extends App {

  def main(args: Array[String]): Unit = {

    // set application configuration and parameters
    val conf = new HashMap[String, String]()
    conf("appName") = "benchmark_userclicks_exp12"
    conf("windowSec") = "30"
    custom(args, conf)

    // setup spark streaming context
    val ssc = getSsc(conf)
    val dstream = getDstream(ssc, conf)

    dstream
      .map(line => (line.split(" - - ")(0), 1))
      .reduceByKeyAndWindow((x: Int, y: Int) => x, Seconds(Integer.parseInt(conf("windowSec"))), Seconds(10))
      .map(_._1)
      // save all results to HDFS
      .saveAsTextFiles(conf("savePath"))

    ssc.start()
    ssc.awaitTermination()
  }
}
