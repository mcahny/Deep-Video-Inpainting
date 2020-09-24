pip install scipy
pip install opencv-python
pip install pillow

cd ./lib/resample2d_package
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py install --user

cd ../../models/correlation_package
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py install --user

