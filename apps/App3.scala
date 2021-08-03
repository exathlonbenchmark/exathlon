package com.spark.benchmark.example.userclicks

import org.apache.spark.streaming.Seconds
import scala.collection.mutable.HashMap

/**
  * Application 3.
  *
  * Selects URLs visited more than 1,000 times within a 30-second sliding window.
  *
  * SQL-like equivalent:
  *
  * SELECT url, COUNT(url) as count
  * FROM UserClicks [RANGE 30s SLIDE 20s]
  * GROUP BY URL
  * HAVING count > 1000;
  */
object App3 extends App {

  def main(args: Array[String]): Unit = {

    // set application configuration and parameters
    val conf = new HashMap[String, String]()
    conf("appName") = "benchmark_userclicks_exp3"
    conf("count") = "1000"
    conf("windowSec") = "30"
    conf("slideSec") = "20"
    custom(args, conf)

    // setup spark streaming context
    val ssc = getSsc(conf)
    val dstream = getDstream(ssc, conf)

    // flag variable used for saving the results for the first batch only
    var flag = true

    dstream
      .map(line => (line.split(" - - ")(2), 1))
      .reduceByKeyAndWindow(_ + _, _ - _, Seconds(Integer.parseInt(conf("windowSec"))), Seconds(Integer.parseInt(conf("slideSec"))))
      .filter(_._2 > Integer.parseInt(conf("count")))
      // save the results for the first batch only
      .filter(x => if (flag) { flag = false; true} else false)
      .saveAsTextFiles(conf("savePath"))

    ssc.start()
    ssc.awaitTermination()
  }
}
