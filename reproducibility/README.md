# Guidelines to reproduce experiments

All experiments were executed on a server with Ubuntu 20.04, AMD Ryzen Threadripper 3960X 24-Core processor, 256 GB RAM and an RTX 4090 GPU.

## Download Docker image
```
docker image pull alzeakis/embeddings4er:v2
```

## Create container
```
docker run -v path-to-logs:/app/logs/ -v path-to-plots:/app/plots/ -d --gpus=all alzeakis/embeddings4er:v2
```
The two -v options guarantee that you can find the logs and plots created inside the container at your host machine. Simply replace the <path-to-logs> and <path-to-plots> with your local directories. Since these affects logs in your local directories, avoid passing existing directories. The -d --gpus=all option guarantees that the container will have access to the gpus of the host machine. If you want CPUs only, please remove this option.

## Enter container
```
docker exec -it <container-id> bash
```

## Reproduce experiments
If you want to execute all experiments and visualizations at once, you can run the above command:
```
source run_all.sh
```

The default behaviour is to skip ZeroER and experiments on Synthetic data. The first one is time-consuming and the latter needs high disk usage (>300GB). They can be added in the execution via special arguments, i.e.:
```
source run_all.sh --zeroer --synthetic
```

If not, run the commands in Partial Execution. Finally, check the Visualizations section to produce the corresponding plots.

## Partial Execution
### Data
There are 3 folders of data:
- Clean-Clean ER Datasets: 10 real-world datasets
- Dirty ER Datasets: 7 synthetic datasets
- Supervised Matching Datasets: 5 mostly synthetic datasets

You can download all data [here](https://zenodo.org/record/8433873/files/data_ea.tar.gz).

### Static Models
For static models, please download the versions of the models we have used from [here](https://zenodo.org/records/11243756/files/models.tar.gz). After decompressing it, it will create a directory called models/.
 
This outer directory will be used in various executions. More instructions on each page.

### Executions
#### Core executions

* For Vectorization:
    * For Vectorization on Real Data (*Exec 1a*), run:
    ```sh
    python vectorize_real.py <raw_data_dir> <emb_data_dir> <log_dir> <static_model_dir>
    ```

    * For Vectorization on Synthetic Data (*Exec 1b*), run:
    ```sh
    python vectorize_synthetic.py <raw_data_dir> <emb_data_dir> <log_dir>  <static_model_dir>
    ```
    
* For Blocking: 
    * For Blocking on Real Data (*Exec 2a*), run:
    ```sh
    python blocking_real.py <raw_data_dir> <emb_data_dir> <log_dir>
    ```

    * For Blocking on Synthetic Data (*Exec 2b*), run:
    ```sh
    python blocking_synthetic.py <raw_data_dir> <emb_data_dir> <log_dir>
    ```
    
* For Matching:
    * For Unsupervised Matching:
        * For Unsupervised Matching without blocking (*Exec 3a*), run:
        ```sh
        python matching_unsupervised.py <raw_data_dir> <emb_data_dir> <log_dir>
        ```

        * For Unsupervised Matching with blocking (*Exec 3b*), run:
        ```sh
        python matching_unsupervised_block.py <raw_data_dir> <emb_data_dir> <log_dir>
        ```

    * For Supervised Matching:
        * For Supervised Matching on static models (*Exec 4a*):
            * To create the environment:
            ```
                python -m venv deepmatcher_venv
                source deepmatcher_venv/bin/activate
                pip install deepmatcher torch==1.9.0 torchtext==0.10
            ```
            * To transform labeled data, run:
            ```
            python transform_labeled.py <data_dir_input> <data_dir_output>
            ```
            
            * To run DeepMatcher, run:
            ```
            ./run_dm.sh <data_dir> <log_dir> <static_model_dir>
            ```
            
        * For Supervised Matching on dynamic models (*Exec 4b*), run:
        ```sh
        ./matching_supervised.sh <data_dir> <log_dir> <exp_dir>
        ```

#### Baseline executions
* DeepBlocker:
	* To execute DeepBlocker (*Exec 5a*), run:
      ```sh
      python run_DeepBlocker.py <raw_data_dir> <log_dir>
      ```
	* More information can be found [here](https://github.com/qcri/DeepBlocker).

* ZeroER:
    * ZeroER needs to make intermediate files for each case, thus we copy the original data directory into the local directory, by running:
	```
	./create_data.sh
	```
	* Create a dedicated conda environment by:
	```
	conda env create -f environment.yml
	conda activate ZeroER
	```
	* To run ZeroER (*Exec 5b*), run:
	```
	./run.sh <log_dir>
    ```
    
### Visualizations

To produce all tables and figure found in the original paper, please run the `visualizations/Schema-Agnostic-Core.ipynb`. The correspondence can be found below.

|        Execution Type | Execution ID | Tables |     Figures |
|----------------------:|-------------:|-------:|------------:|
|         Vectorization |           1a |      4 |          10 |
|                       |           1b |        |           9 |
|              Blocking |           2a |      5 | 1, 2, 3, 10 |
|                       |           2b |        |        4, 9 |
| Unsupervised Matching |           3a |        | 5, 6, 7, 10 |
|                       |           3b |      5 |           5 |
|   Supervised Matching |           4a |      6 |       8, 10 |
|                       |           4b |      6 |       8, 10 |
|              Baseline |           5a |      5 |           1 |
|                       |           5b |      5 |           5 |


