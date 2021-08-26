Exathlon
==============================

Access to high-quality data repositories and benchmarks have been instrumental in advancing the state of the art in many experimental research domains. 

Exathlon is a benchmark for explainable anomaly detection over high-dimensional time series data, constructed based on real data traces from repeated executions of large-scale stream processing jobs on an Apache Spark cluster. For some of these executions, we introduced instances of six different types of anomalous events, for which we provide ground truth labels to evaluate a wide range of anomaly detection (AD) and explanation discovery (ED) methods.

This repository contains the labeled dataset and source code for comparing various AD and ED methods under our evaluation framework.

[Description and documentation](https://github.com/exathlonbenchmark/exathlon/wiki). 

## Project Configuration

The data traces and ground truth table were uploaded as zip files under the `data/raw` directory. To extract them on Linux, macOS, or using Git Bash on Windows, execute the `extract_data` script from the project root folder:

```bash
./extract_data.sh
```

This will extract all data files inside `data/raw`, preserving its directory structure. The content of `data/raw` can then either be left there or moved to any other location. In all cases, the full path to the extracted raw data must be provided to `DATA_ROOT` entry of the `.env` file described below.

Please refer to the [dataset documentation](https://github.com/exathlonbenchmark/exathlon/wiki/Dataset) for additional details regarding the dataset's content and format.

Using `conda`, from the project root folder, execute the following commands: 

```bash
conda create -n exathlon python=3.7
conda activate exathlon
conda install -c conda-forge --yes --file requirements.txt
```

At the root of the project folder, create a `.env` file containing the lines:

```bash
USED_DATA=SPARK
DATA_ROOT=path/to/extracted/data/raw
OUTPUTS_ROOT=path/to/pipeline/outputs
```

The pipeline outputs refer to all the outputs that will be produced by Exathlon's pipeline, including intermediate and fully processed data, models, model information and final results.

### Note: Running this Project on Windows

Some results and logging paths might exceed the Windows historical path length limitation of 260 characters, leading to some errors when running the pipeline. To counter this, we advise to disable this limitation following the procedure described in the [official Python documentation](https://docs.python.org/3/using/windows.html):

>Windows historically has limited path lengths to 260 characters. This meant that paths longer than this would not resolve and errors would result.
>
>In the latest versions of Windows, this limitation can be expanded to approximately 32,000 characters. Your administrator will need to activate the “Enable Win32 long paths” group policy, or set `LongPathsEnabled` to `1` in the registry key `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`.
>
>This allows the [`open()`](https://docs.python.org/3/library/functions.html#open) function, the [`os`](https://docs.python.org/3/library/os.html#module-os) module and most other path functionality to accept and return paths longer than 260 characters.
>
>After changing the above option, no further configuration is required.
>
>Changed in version 3.6: Support for long paths was enabled in Python. 

## Data License

The provided dataset is licensed under a [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Publications

If you are using this benchmark as part of your research or work, please cite the following paper:

[Exathlon: A Benchmark for Explainable Anomaly Detection over Time Series](http://vldb.org/pvldb/vol14/p2613-tatbul.pdf). Vincent Jacob, Fei Song, Arnaud Stiegler, Bijan Rad, Yanlei Diao, and Nesime Tatbul. Proceedings of the VLDB Endowment (PVLDB), 14(11): 2613 - 2626, 2021.