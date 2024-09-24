import torch
import torch.fft as fft

def fourierTransformST(featuresTuple,cutoff=30):
    for i in range(len(featuresTuple)):
        features_freq=featuresTuple[i].squeeze()
        features_freq=fourierBatchTransform(features_freq,cutoff)
        featuresTuple[i]=features_freq.unsqueeze(0)
    return featuresTuple


def fourierBatchTransform(features,cutoff=30):
    features=batchFourierFilterLowPass(features,cutoff)
    features=batchFourierFilterLowPass(features,cutoff)
    return features

def batchFourierFilterLowPass(batch, cutoff=30):
    f = fft.fft2(batch, dim=(-2, -1))
    fshift = fft.fftshift(f, dim=(-2, -1))

    rows, cols = batch.shape[-2:]
    crow, ccol = int(rows / 2), int(cols / 2)
    d = cutoff  
    n = 2  
    Y, X = torch.meshgrid(torch.arange(rows), torch.arange(cols),indexing='ij')
    dist = torch.sqrt((X - crow) ** 2 + (Y - ccol) ** 2)
    maska = 1 / (1 + (dist / d) ** (2 * n))
    maska = maska.unsqueeze(0).to('cuda')  
    fshift_filtered = fshift * maska
    f_ishift = fft.ifftshift(fshift_filtered, dim=(-2, -1))
    batch_filtered = fft.ifft2(f_ishift, dim=(-2, -1))
    batch_filtered = torch.abs(batch_filtered).float()
    return batch_filtered










