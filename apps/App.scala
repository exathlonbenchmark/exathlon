package com.spark.benchmark.example.userclicks

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.streaming.{Seconds, StreamingContext}
import scala.collection.mutable.HashMap

/**
  * Application base class.
  */
class App {

  // default application parameters
  private def defaultParameters(conf: HashMap[String, String]): Unit = {
    conf("batchInterval") = "5"
    conf("blockInterval") = "600"
    conf("parallelism") = "40"
    conf("senderNum") = "1"
    conf("senderHostname1") = "host"
    conf("senderPort1") = "10000"
    conf("checkpointDir") = "hdfs:///path/to/checkpointing/dir"
    conf("metricsPath") = "/path/to/metrics"
    conf("savePath") = "hdfs:///path/to/saved/data"
  }

  // simple cpu pressure simulating a UDF
  protected def cpuPressure(pressure: Int): Unit = {
    var pi: Double = 0;
    (1 to pressure).foreach(i => pi = pi + Math.pow(-1, (i + 1)) * 4 / (2 * i - 1))
  }

  // manually configured parameters (in the form "key=value")
  protected def custom(args: Array[String], conf: HashMap[String, String]) = {
    defaultParameters(conf)
    for (line <- args)
      try {
        conf(line.split("=")(0)) = line.split("=")(1)
      } catch {
        case ex: Exception =>
          ex.printStackTrace()
          System.exit(1)
      }
  }

  // create StreamingContext
  protected def getSsc(conf: HashMap[String, String]) = {

    val sparkConf = new SparkConf()

    sparkConf.setAppName(conf("appName"))
    sparkConf.set("spark.streaming.backpressure.enabled", "true")
    sparkConf.set("spark.streaming.blockInterval", conf("blockInterval"))
    sparkConf.set("spark.default.parallelism", conf("parallelism"))
    sparkConf.set("spark.metrics.namespace", conf("appName"))

    val sc = new SparkContext(sparkConf)

    // configure batch interval
    val ssc = new StreamingContext(sc, Seconds(Integer.parseInt(conf("batchInterval"))))

    // configure checkpointing for the window operator
    ssc.checkpoint(conf("checkpointDir"))
    ssc
  }

  // get Dstream
  protected def getDstream(ssc: StreamingContext, conf: HashMap[String, String]) = {
    getSocketDstream(ssc, conf)
  }

  // every receivers need a sender coming from senderHostname* and senderPort*
  protected def getSocketDstream(ssc: StreamingContext, conf: HashMap[String, String]) = {

    // make receivers distribute on all cluster nodes
    ssc.sparkContext.makeRDD(1 to 5000, 100).map(x => (x, 1)).reduceByKey(_ + _, 100).collect()

    val num = Integer.parseInt(conf("senderNum"))
    if (num <= 0) {
      println("senderNum must > 0")
      System.exit(1)
    }
    // union all receivers to one
    ssc.union((1 to num).map(n => ssc.socketTextStream(conf("senderHostname" + n), Integer.parseInt(conf("senderPort" + n)))))
  }
}
