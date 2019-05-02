**********Test**********
You can find the trained model (*.caffemodel) and the test protocol (./model/test.prototxt) under each folder.

**********Training**********
./PWC-Net_plus_kitti/run.sh or ./PWC-Net_plus_sintel/run.sh
Please modify the local paths to the Caffe bin in train*.py and data LMDB files in train*.prototxt

**********Reference**********
For more details, please see (https://arxiv.org/abs/1809.05571)
@article{Sun2018:Model:Training:Flow,
  author={Sun, Deqing and Yang, Xiaodong and Liu, Ming-Yu and Kautz, Jan},
  title={Models Matter, So Does Training: An Empirical Study of CNNs for Optical Flow Estimation},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2019}
}
