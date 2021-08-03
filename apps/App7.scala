package com.spark.benchmark.example.userclicks

import org.apache.spark.streaming.Seconds
import scala.collection.mutable.HashMap

/**
  * Application 7.
  *
  * Groups information by user within a 30-second sliding window.
  *
  * The operation is the same as application 4, but with some added CPU pressure to simulate
  * a user-defined function (UDF).
  *
  * SELECT userId
  * FROM UserClicks [RANGE 30s]
  * GROUP BY userId;
  */
object App7 extends App {

  def main(args: Array[String]): Unit = {

    // set application configuration and parameters
    val conf = new HashMap[String, String]()
    conf("appName") = "benchmark_userclicks_exp9"
    conf("windowSec") = "30"
    conf("pressure") = "70"
    custom(args, conf)

    // setup spark streaming context
    val ssc = getSsc(conf)
    val dstream = getDstream(ssc, conf)

    // flag variable used for saving the results for the first batch only
    var flag = true

    dstream
      .map(line => {
        // added cpu pressure to simulate a UDF before the actual operation
        cpuPressure(conf("pressure").toInt)
        (line.split(" - - ")(0), null)
      })
      .reduceByKeyAndWindow((x, y) => x, Seconds(Integer.parseInt(conf("windowSec"))))
      .map(_._1)
      // save the results for the first batch only
      .filter(x => if (flag) { flag = false; true} else false)
      .saveAsTextFiles(conf("savePath"))

    ssc.start()
    ssc.awaitTermination()
  }
}
