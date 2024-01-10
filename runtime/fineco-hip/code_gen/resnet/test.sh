
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=90
echo "set MPS 90"
python3 resnet_test.py conv1  100
python3 resnet_test.py conv2_1  100
python3 resnet_test.py conv3_1  100
python3 resnet_test.py conv4_1  100
python3 resnet_test.py conv5_1  100

export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80
echo "set MPS 80"
python3 resnet_test.py conv1  100
python3 resnet_test.py conv2_1  100
python3 resnet_test.py conv3_1  100
python3 resnet_test.py conv4_1  100
python3 resnet_test.py conv5_1  100

export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=70
echo "set MPS 70"
python3 resnet_test.py conv1  100
python3 resnet_test.py conv2_1  100
python3 resnet_test.py conv3_1  100
python3 resnet_test.py conv4_1  100
python3 resnet_test.py conv5_1  100

export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=60
echo "set MPS 60"
python3 resnet_test.py conv1  100
python3 resnet_test.py conv2_1  100
python3 resnet_test.py conv3_1  100
python3 resnet_test.py conv4_1  100
python3 resnet_test.py conv5_1  100

export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50
echo "set MPS 50"
python3 resnet_test.py conv1  100
python3 resnet_test.py conv2_1  100
python3 resnet_test.py conv3_1  100
python3 resnet_test.py conv4_1  100
python3 resnet_test.py conv5_1  100

export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40
echo "set MPS 40"
python3 resnet_test.py conv1  100
python3 resnet_test.py conv2_1  100
python3 resnet_test.py conv3_1  100
python3 resnet_test.py conv4_1  100
python3 resnet_test.py conv5_1  100

export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=30
echo "set MPS 30"
python3 resnet_test.py conv1  100
python3 resnet_test.py conv2_1  100
python3 resnet_test.py conv3_1  100
python3 resnet_test.py conv4_1  100
python3 resnet_test.py conv5_1  100

export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20
echo "set MPS 20"
python3 resnet_test.py conv1  100
python3 resnet_test.py conv2_1  100
python3 resnet_test.py conv3_1  100
python3 resnet_test.py conv4_1  100
python3 resnet_test.py conv5_1  100

export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=10
echo "set MPS 10"
python3 resnet_test.py conv1  100
python3 resnet_test.py conv2_1  100
python3 resnet_test.py conv3_1  100
python3 resnet_test.py conv4_1  100
python3 resnet_test.py conv5_1  100
