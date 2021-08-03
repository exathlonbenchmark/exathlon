package com.spark.benchmark.example.userclicks

import org.apache.spark.streaming.Seconds
import scala.collection.mutable.HashMap

/**
  * Application 5.
  *
  * Computes the sum of ranks for the URLs visited by each user within a 30-second sliding window.
  *
  * SQL-like equivalent:
  *
  * SELECT userId, SUM(pageRank)
  * FROM PageRanks, UserClicks [RANGE 30s]
  * WHERE PageRanks.url = UserClicks.url
  * GROUP BY userId;
  */
object App5 extends App {

  def main(args: Array[String]): Unit = {

    // set application configuration and parameters
    val conf = new HashMap[String, String]()
    conf("appName") = "benchmark_userclicks_exp7"
    conf("windowSec") = "30"
    conf("datasetPath") = "hdfs://path/to/PageRanks/dataset"
    custom(args, conf)

    // setup spark streaming context
    val ssc = getSsc(conf)
    val dstream = getDstream(ssc, conf)

    // flag variable used for saving the results for the first batch only
    var flag = true

    // load and broadcast the PageRanks dataset (to join to the streaming records)
    val dataset = ssc.sparkContext.textFile(conf("datasetPath")).map(
      line => {
        val splits = line.split(" - - ")
        (splits(1), splits(0).toInt)
      }
    )
    val dataMap = Map[String, Int]() ++ dataset.collect()
    val broadcastMap = ssc.sparkContext.broadcast(dataMap)

    dstream
      .map(line => {
        val splits = line.split(" - - ")
        (splits(2), splits(0))
      })
      .filter(line => broadcastMap.value.contains(line._1))
      .map(line => (line._2, broadcastMap.value.get(line._1).get))
      .reduceByKeyAndWindow(_ + _, Seconds(Integer.parseInt(conf("windowSec"))))
      // save the results for the first batch only
      .filter(x => if (flag) { flag = false; true} else false)
      .saveAsTextFiles(conf("savePath"))

    ssc.start()
    ssc.awaitTermination()
  }
}
