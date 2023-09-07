* For finding the range of k in Blocking, run:
    ```sh
    python blocking_k_range.py  <raw_data_dir> <emb_data_dir> <log_dir>
    ```

* For finding the impact of Indexing in Blocking, run:
    ```sh
    python blocking_indexing.py  <raw_data_dir> <emb_data_dir> <log_dir>
    ```

* For finding the complementarity of models in Blocking, run:
    ```sh
    python blocking_complementarity.py  <raw_data_dir> <emb_data_dir> <log_dir>
    ```

* For finding the impact of Supervision:
    * First, vectorize the 5 datasets with:
        ```sh
        python supervision_impact/vectorize.py <raw_data_dir> <emb_data_dir> <log_dir>  <static_dir>
        ```
    * Then, run the matching method:
        ```sh
        python supervision_impact/matching.py  <raw_data_dir> <emb_data_dir> <log_dir>
        ```