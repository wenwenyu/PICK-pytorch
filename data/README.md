## Data format example
This implementation requires the input data in the following format:
### Training/Validation data
1. `train_samples_list.csv` file: multi-line with format: `index,document_type,file_name`.
2. `boxes_and_transcripts` folder: `file_name.tsv` files.
    * every `file_name.tsv` file has multi-line with format: `index,box_coordinates (clockwise 8 values),
    transcripts,box_entity_types` .
3. `images` folder:  `file_name.jpg` files.
4. `entities` folder (optional) : `file_name.txt` files.
    * every `file_name.txt` file contains a json format string, providing the exactly label value of
    every entity.
    * if `iob_tagging_type` is set to `box_level`, this folder will not be used, then `box_entity_types` in
     `file_name.tsv` file of `boxes_and_transcripts` folder will be used as label of entity.
      otherwise, it must be provided.
### Testing data
1. `boxes_and_transcripts` folder: `file_name.tsv` files
    * every `file_name.tsv` file has multi-line with format: `index,box_coordinates (clockwise 8 values),
    transcripts`.
2. `images` folder:  `file_name.jpg` files.