This package provides an example to train PICK on [DocBank](https://doc-analysis.github.io/docbank-page/index.html) dataset.

## Dataset details
### Statistics
| Split | Abstract | Author | Caption |  Date | Equation | Figure | Footer |  List  | Paragraph | Reference | Section | Table | Title |  Total  |
|:-----:|:--------:|:------:|:-------:|:-----:|:--------:|:------:|:------:|:------:|:---------:|:---------:|:-------:|:-----:|:-----:|:-------:|
| Train |   25,387  |  25,909 |  106,723 |  6,391 |  161,140  |  90,429 |  38,482 |  44,927 |   398,086  |   44,813   |  180,774 | 19,638 | 21,688 |  400,000 |
|       |   6.35%  |  6.48% |  26.68% | 1.60% |  40.29%  | 22.61% |  9.62% | 11.23% |   99.52%  |   11.20%  |  45.19% | 4.91% | 5.42% | 100.00% |
|  Dev  |   3,164   |  3,286  |  13,443  |  797  |   20,154  |  11,463 |  4,804  |  5,609  |   49,759   |    5,549   |  22,666  |  2,374 |  2,708 |  50,000  |
|       |   6.33%  |  6.57% |  26.89% | 1.59% |  40.31%  | 22.93% |  9.61% | 11.22% |   99.52%  |   11.10%  |  45.33% | 4.75% | 5.42% | 100.00% |
|  Test |   3,176   |  3,277  |  13,476  |  832  |   20,244  |  11,378 |  4,876  |  5,553  |   49,762   |    5,641   |  22,384  |  2,505 |  2,729 |  50,000  |
|       |   6.35%  |  6.55% |  26.95% | 1.66% |  40.49%  | 22.76% |  9.75% | 11.11% |   99.52%  |   11.28%  |  44.77% | 5.01% | 5.46% | 100.00% |
| Total |   31,727  |  32,472 |  133,642 |  8,020 |  201,538  | 113,270 |  48,162 |  56,089 |   497,607  |   56,003   |  225,824 | 24,517 | 27,125 |  500,000 |
|       |   6.35%  |  6.49% |  26.73% | 1.60% |  40.31%  | 22.65% |  9.63% | 11.22% |   99.52%  |   11.20%  |  45.16% | 4.90% | 5.43% | 100.00% |

### Annotation
There are 11 labels: abstract, author, caption, equation, figure, footer, list, paragraph, reference, section, table, title. 
In the process, we will ignore the blank pages (they are probably wrong annotations).


## Usage
1. Download annotation zip files (~47GB) and image zip file (~3GB) from [DocBank](https://doc-analysis.github.io/docbank-page/index.html). 
We assume you have saved those files to the folder `${BASE_DATA_DIR}`.
You will end up with 10 image zip files (`DocBank_500K_ori_img.zip.0*`) and one annotation zip file (`DocBank_500K_txt.zip`) in `${BASE_DATA_DIR}`;
2. Merge the image zip files to one file, named `DocBank_500K_ori_img.zip`;
3. Run the scripts. The script will convert annotation files and split the dataset. You will end up with 
`train`, `dev` and `test` folders under `${BASE_DATA_DIR}`.