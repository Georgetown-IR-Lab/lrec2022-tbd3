# lrec2022-tbd3

This repository contains the resource code for the paper "TBD3: A Thresholding-Based Dynamic Depression Detection from Social Media for Low-Resource Users" accepted at Language Resources and Evaluation Conference (LREC) 2022.

## Citation

If you use the resource code please cite the following paper:

```
@inproceedings{kulkarni:lrec2022-TBD3,
  title={TBD3: A Thresholding-Based Dynamic Depression Detection from Social Media for Low-Resource Users},
  author={Kulkarni, Hrishikesh and MacAvaney, Sean and Goharian, Nazli and Frieder, Ophir},
  booktitle={Proceedings of the Thirteenth International Conference on Language Resources and Evaluation (LREC 2022)}
  year={2022}
}
```

## Run

Change the corresponding options to set parameters of the filter:

```
    args.add_argument('-p', '--percentage', default=1.0, type=float, help='From 0 to 1')
    args.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size')
    args.add_argument('-nob', '--no_of_batches', default=24, type=int, help='No of Batches')
    args.add_argument('-mp', '--max_posts', default=2000, type=int, help='Max Posts')
    args.add_argument('-pt', '--part', default='first', type=str, help='first or last or random')
    args.add_argument('-m', '--model', default='cnn', type=str, help='ML algorithm')
    args.add_argument('-md','--median', default='False', type=str, help='True or False')
```

## Use

Returns a generator with given data specifications.

### Filters

- Percentage: For low-resource users with dynamic number of posts
- Part: First, Last or Random selection of posts
- Max_Posts: Maximum number of posts considered per user 
- Median: Only the users with more than median number of posts considered

## Data

Reddit Self-reported Depression Diagnosis (RSDD) dataset used. Details can be found [here](https://ir.cs.georgetown.edu/resources/rsdd.html).

## Requirements

Python 3
