import wget
import os
import scipy.io
import numpy as np
import pandas as pd
import base64


def my_raw_denoiser(x):
    """TODO: Implement your own raw-RGB denoiser here."""
    return x.copy()


def array_to_base64string(x):
    array_bytes = x.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string


def base64string_to_array(base64string, array_dtype, array_shape):
    decoded_bytes = base64.b64decode(base64string)
    decoded_array = np.frombuffer(decoded_bytes, dtype=array_dtype)
    decoded_array = decoded_array.reshape(array_shape)
    return decoded_array


# Download input file, if needed.
url = 'https://competitions.codalab.org/my/datasets/download/7a9badd9-fffc-4e56-ba41-0c5e2a76e58a'
input_file = 'BenchmarkNoisyBlocksRaw.mat'
if os.path.exists(input_file):
    print(f'{input_file} exists. No need to download it.')
else:
    print('Downloading input file BenchmarkNoisyBlocksRaw.mat...')
    wget.download(url, input_file)
    print('Downloaded successfully.')

# Read inputs.
key = 'BenchmarkNoisyBlocksRaw'
inputs = scipy.io.loadmat(input_file)
inputs = inputs[key]
print(f'inputs.shape = {inputs.shape}')
print(f'inputs.dtype = {inputs.dtype}')

# Denoising.
output_blocks_base64string = []
for i in range(inputs.shape[0]):
    for j in range(inputs.shape[1]):
        in_block = inputs[i, j, :, :]
        out_block = my_raw_denoiser(in_block)
        assert in_block.shape == out_block.shape
        assert in_block.dtype == out_block.dtype
        out_block_base64string = array_to_base64string(out_block)
        output_blocks_base64string.append(out_block_base64string)

# Save outputs to .csv file.
output_file = 'SubmitRaw.csv'
print(f'Saving outputs to {output_file}')
output_df = pd.DataFrame()
n_blocks = len(output_blocks_base64string)
print(f'Number of blocks = {n_blocks}')
output_df['ID'] = np.arange(n_blocks)
output_df['BLOCK'] = output_blocks_base64string

output_df.to_csv(output_file, index=False)

# TODO: Submit the output file SubmitRaw.csv at 
# kaggle.com/competitions/sidd-benchmark-raw-psnr
print('TODO: Submit the output file SubmitRaw.csv at')
print('kaggle.com/competitions/sidd-benchmark-raw-psnr')

print('Done.')