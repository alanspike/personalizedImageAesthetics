# Personalized Image Aesthetics


### Datasets

The FLICKR-AES and REAL-CUR are available at [google drive](https://drive.google.com/drive/folders/1LR6trJhN4XbgTtqZo1zfe272cAkXqA7e?usp=share_link)


`FLICKR-AES_image_labeled_by_each_worker.csv` saves the images labeled by each worker. 

`FLICKR-AES_image_score.txt` saves the weighted average score for each image.

All the images are under Creative Commons license. The datasets are for research purpose only.

### Model

The structure of the generic model is provided. 

### RUN Generic Aesthetics Model

```
python test.py --filename /path/to/imageList.txt
```

If you find our work useful, please cite our paper:

	@InProceedings{Ren_2017_ICCV,
	  author = {Ren, Jian and Shen, Xiaohui and Lin, Zhe and Mech, Radomir and Foran, David J.},
	  title = {Personalized Image Aesthetics},
	  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
	  month = {Oct},
	  year = {2017}
	}
