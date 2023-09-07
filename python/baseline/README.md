* DeepBlocker:
	* In configurations.py, set the path of fasttext model, i.e. the .bin file.
	* To execute DeepBlocker, run:
      ```sh
      python run_DeepBlocker.py <raw_data_dir> <log_dir>
      ```
	* More information can be found [here](https://github.com/qcri/DeepBlocker).

* JedAI:
	* Install JedAI from PyPI:
	 ```
	 pip install pyjedai
	 ```
	* To execute JedAI, run:
      ```sh
      python run_jedai.py <raw_data_dir> <log_dir>
      ```
	* More information can be found [here](https://github.com/AI-team-UoA/pyJedAI).
	
* Sparkly:
	* Download PyLucene from [here](https://lucene.apache.org/pylucene/) and make the appropriate changes in Makefile. We have used version 9.7.0.
	* Build the Docker image by 
	 ```
	 docker build -t sparkly .
	 ```
	* To run the Sparkly container, run:
      ```sh
      docker run -v <log_dir>:/usr/src/sparkly/logs -v <raw_data_dir>:/usr/src/sparkly/data --name sparkly sparkly
      ```
	* More information can be found [here](https://github.com/anhaidgroup/sparkly).	
	
* TokenJoin:
	* Install TokenJoin from PyPI:
	 ```
	 pip install pytokenjoin
	 ```
	* To execute TokenJoin, run:
      ```sh
      python run_tjk.py <raw_data_dir> <log_dir>
      ```
	* More information can be found [here](https://github.com/alexZeakis/pyTokenJoin).

* ZeroER:
    * ZeroER needs to make intermediate files for each case, thus we copy the original data directory into the local directory under the name datasets.
	* Create a conda environment by:
	```
	conda env create -f environment.yml
	conda activate ZeroER
	```
	* To run ZeroER, run:
	```
	./run.sh <log_dir>
    ```
