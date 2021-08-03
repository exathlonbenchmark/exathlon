package com.spark.benchmark.example.userclicks

import scala.collection.mutable.HashMap

/**
  * Application 2.
  *
  * Selects URLs visited more than 1,000 times from the start of the application.
  *
  * SQL-like equivalent:
  *
  * SELECT url, COUNT(url) as count
  * FROM UserClicks
  * GROUP BY url
  * HAVING count > 1000;
  */
object App2 extends App {

  def main(args: Array[String]): Unit = {

    // set application configuration and parameters
    val conf = new HashMap[String, String]()
    conf("appName") = "benchmark_userclicks_exp2"
    conf("count") = "1000"
    custom(args, conf)

    // setup spark streaming context
    val ssc = getSsc(conf)
    val dstream = getDstream(ssc, conf)

    // flag variable used for saving the results for the first batch only
    var flag = true

    // accumulator function to maintain the total number of visits
    val updateFunction = (currValues: Seq[Int], prevValueState: Option[Int]) => {
      val currentCount = currValues.sum
      val previousCount = prevValueState.getOrElse(0)
      Some(currentCount + previousCount)
    }

    dstream
      .map(line => (line.split(" - - ")(2), 1))
      .updateStateByKey[Int](updateFunction)
      .filter(_._2 > Integer.parseInt(conf("count")))
      // save the results for the first batch only
      .filter(x => if (flag) { flag = false; true} else false)
      .saveAsTextFiles(conf("savePath"))

    ssc.start()
    ssc.awaitTermination()
  }
}
