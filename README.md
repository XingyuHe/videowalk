ø<h1 style='font-size: 1.6em'>Space-Time Correspondence as a Contrastive Random Walk</h1>

<!-- ![](https://github.com/ajabri/videowalk/raw/master/figs/teaser_animation.gif) -->
<p align="center">
<img src="https://github.com/ajabri/videowalk/raw/master/figs/teaser_animation.gif" width="600">
</p>

This is the repository for *Space-Time Correspondence as a Contrastive Random Walk*, published at NeurIPS 2020.  

[[Paper](https://arxiv.org/abs/2006.14613)]
[[Project Page](http://ajabri.github.io/videowalk)]
[[Slides](https://www.dropbox.com/s/qrqb0ssjlh1tph1/jabri.nips.12min.public.key)]
[[Poster](https://www.dropbox.com/s/snpj68cssu3b4to/jabri.neurips2020.poster.pdf)]
[[Talk](https://youtu.be/UaOcjxrPaho)]

```
@inproceedings{jabri2020walk,
    Author = {Allan Jabri and Andrew Owens and Alexei A. Efros},
    Title = {Space-Time Correspondence as a Contrastive Random Walk},
    Booktitle = {Advances in Neural Information Processing Systems},
    Year = {2020},
}
```
Consider citing our work or acknowledging this repository if you found this code to be helpful :)

##  Requirements
- pytorch (>1.3)
- torchvision (0.6.0)
- cv2
- matplotlib
- skimage
- imageio

For visualization (`--visualize`):
- wandb
- visdom
- sklearn



## Train
An example training command is:
```
python -W ignore train.py --data-path /path/to/kinetics/ \
--frame-aug grid --dropout 0.1 --clip-len 4 --temp 0.05 \
--model-type scratch --workers 16 --batch-size 20  \
--cache-dataset --data-parallel --visualize --lr 0.0001
```

This yields a model with performance on DAVIS as follows (see below for evaluation instructions), provided as `pretrained.pth`:
```
 J&F-Mean    J-Mean  J-Recall  J-Decay    F-Mean  F-Recall   F-Decay
  0.67606  0.645902  0.758043   0.2031  0.706219   0.83221  0.246789
```

Arguments of interest:

* `--dropout`: The rate of edge dropout (default `0.1`).
* `--clip-len`: Length of video sequence.
* `--temp`: Softmax temperature.
* `--model-type`: Type of encoder. Use `scratch` or `scratch_zeropad` if training from scratch. Use `imagenet18` to load an Imagenet-pretrained network. Use `scratch` with `--resume` if reloading a checkpoint.
* `--batch-size`: I've managed to train models with batch sizes between 6 and 24. If you have can afford a larger batch size, consider increasing the `--lr` from 0.0001 to 0.0003.
* `--frame-aug`: `grid` samples a grid of patches to get nodes; `none` will just use a single image and use embeddings in the feature map as nodes.
* `--visualize`: Log diagonistics to `wandb` and data visualizations to `visdom`.

### Data

We use the official `torchvision.datasets.Kinetics400` class for training. You can find directions for downloading Kinetics [here](https://github.com/pytorch/vision/tree/master/references/video_classification). In particular, the code expects the path given for kinetics to contain a `train_256` subdirectory.

You can also provide `--data-path` with a file with a list of directories of images, or a path to a directory of directory of images. In this case, clips are randomly subsampled from the directory.


### Visualization
By default, the training script will log diagnostics to `wandb` and data visualizations to `visdom`.


### Pretrained Model
You can find the model resulting from the training command above at `pretrained.pth`.
We are still training updated ablation models and will post them when ready.

---

## Evaluation: Label Propagation
The label propagation algorithm is described in `test.py`.  The output of `test.py` (predicted label maps) must be post-processed for evaluation.

### DAVIS
To evaluate a trained model on the DAVIS task, clone the [davis2017-evaluation](https://github.com/davisvideochallenge/davis2017-evaluation) repository, and prepare the data by downloading the [2017 dataset](https://davischallenge.org/davis2017/code.html) and modifying the paths provided in `eval/davis_vallist.txt`. Then, run:


**Label Propagation:**
```
python test.py --filelist /path/to/davis/vallist.txt \
--model-type scratch --resume ../pretrained.pth --save-path /save/path \
--topk 10 --videoLen 20 --radius 12  --temperature 0.05  --cropSize -1
```
Though `test.py` expects a model file created with `train.py`, it can easily be modified to be used with other networks. Note that we simply use the same temperature used at training time.

You can also run the ImageNet baseline with the command below.
```
python test.py --filelist /path/to/davis/vallist.txt \
--model-type imagenet18 --save-path /save/path \
--topk 10 --videoLen 20 --radius 12  --temperature 0.05  --cropSize -1
```

comment on test.py arguments
--filelist 
   [row[0] is a directory path that contains jpgfile path], [row[1] is a directory path that contains lblfile path]

   each row represents one sequence of frames. The filename of these frames should be named according to its alphebetical order

   these elements will be called os.listdir() on
   each files under 1 directory path will be added to img_paths and lbl_paths in VOSDatatset.__getitem__

comment on VOSDatatset
VOSDatatset[i] gets the resized image and the labels (see __getitem__)
   lbls_map [uH x nW]: nH number of unique rows; each row is of length nW; all rows below to the first image of the frame
   lbls_resize [N x C [nH x uH]]: the one-hot vector for each row in N x C. The jth entry in the vector indicates whether it matchs the jth row in lbls_map



**Post-Process:**  
```
# Convert
python eval/convert_davis.py --in_folder /save/path/ --out_folder /converted/path --dataset /davis/path/

# Compute metrics
python /path/to/davis2017-evaluation/evaluation_method.py \
--task semi-supervised   --results_path /converted/path --set val \
--davis_path /path/to/davis/
```

You can generate the above commands with the script below, where removing `--dryrun` will actually run them in sequence.
```
python eval/run_test.py --model-path /path/to/model --L 20 --K 10  --T 0.05 --cropSize -1 --dryrun
```


## Test-time Adaptation
To do.


### Data format
data_path: a file where in each row, row[0]=jpgfile, row[1]=fnum
jpgfile: path to a folder that contains a list of images (sorted by file name to form a video)
frame_gap: the number of frames between each selected frames
clip_len: the number of frames selected


#### How is data loaded? 
##### dataset: VideoList
__getitem__ return 3 tensors
1. imgs: T x H x W x 3
   1. img: H x W x 3 (RGB)
   2. 1 imgs represents 
2. 0
3. 0

###### data_loader: DataLoader
collate_fn inputs a list of return values from __getitem__ and returns a list of imgs. The model eventually only receives the imgs object from __getitem__

#### Model
##### model: CRW: nn.Module
self.encoder: resnet18 or otherwise 
self.selfsim_fc: nn.Sequential that contains the fully connected layer [self.enc_hid_dim, ..., 128]

forward():
    input: x [B, T, _N*C, H, W]
    B: batch_size
    T: frame_counts
    _N: patch_counts
    C: channel_counts
    H: height
    W: width

    1. add a new dimension for patch _N
       1. x [B, _N, C, T, H, W]
    2. pixels_to_nodes
       1. q [B x C x T x N] node embeddings
       2. mm [B x N x C x T x H x W]

pixels_to_nodes()
    convert batch of images to nodes where each pixel is a node
    output: 
    1. maps [B x N x C x T x H x W]
    2. feats [B x C x T x N]  node embeddings
       1. sum across H and W dimensions
       2. normalize across C dimensino
       3. reshape