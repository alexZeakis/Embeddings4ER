## Environment
Most executions are placed within the same conda environment, called `vldb2023_ea_basic`. To create and activate it, run:
```
conda env create -f conda/vldb_ea_basic.yml
conda activate vldb_ea_basic
```

* For Vectorization:
    * For Vectorization on Real Data, run:
    ```sh
    python vectorize_real.py <raw_data_dir> <emb_data_dir> <log_dir> <static_dir>
    ```
    * For Vectorization on Synthetic Data, run:
    ```sh
    python vectorize_synthetic.py <raw_data_dir> <emb_data_dir> <log_dir>  <static_dir>
    ```

* For Blocking: 
    * For Blocking on Real Data, run:
    ```sh
    python blocking_real.py <raw_data_dir> <emb_data_dir> <log_dir>
    ```
    * For Blocking on Synthetic Data, run:
    ```sh
    python blocking_synthetic.py <raw_data_dir> <emb_data_dir> <log_dir>
    ```

* For Matching:
    * For Unsupervised Matching:
        * For Unsupervised Matching without blocking, run:
        ```sh
        python matching_unsupervised.py <raw_data_dir> <emb_data_dir> <log_dir>
        ```
        * For Unsupervised Matching with blocking, run:
        ```sh
        python matching_unsupervised_block.py <raw_data_dir> <emb_data_dir> <log_dir>
        ```

    * For Supervised Matching:
        * For Supervised Matching on static models:
            * To transform labeled data, run:
            ```
            python transform_labeled.py <data_dir_input> <data_dir_output>
            ```
            * To run DeepMatcher, run:
            ```
            ./run_dm.sh <data_dir> <log_dir> <model_dir>
            ```
            
        * For Supervised Matching on dynamic models, run:
        ```sh
        ./matching_supervised.sh <data_dir> <log_dir> <exp_dir>
        ```
