name := "Homework2"

version := "1.0"

scalaVersion := "2.11.8"
val sparkVersion = "2.4.4"


libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-streaming" % sparkVersion