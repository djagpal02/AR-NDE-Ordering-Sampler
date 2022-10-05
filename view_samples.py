import numpy as np
import matplotlib.pyplot as plt
import os

def save_samples(samples, save_path = "samples.pdf", ncols=10, nrows=10):
    n_samples = ncols*nrows
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows)
    ax = axes.ravel()
    for i in range(n_samples):
        if samples.shape[2] == 32:
            # For CIFAR-10
            ax[i].imshow(samples[i])
        else:
            # For MNIST and Binary MNIST
            ax[i].imshow(np.transpose(samples[i], (0, 1)), cmap="gray")
            
        ax[i].axis("off")
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].set_frame_on(False)


    save_path = "results/" + save_path

    fig.subplots_adjust(wspace=-0.35, hspace=0.065)
    plt.gca().set_axis_off()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0,)
    plt.close()

    print('Samples saved to: ' + save_path)


def CIFAR10_sample_to_image(samples):
    # Remove invalid values: 
    # Set values above 1 to 1 and below 0 to 0
    less_than_zero_filter = (samples > 0).astype(int)
    samples = (((samples - 1 < 0).astype(int) * (samples-1)) + 1) * less_than_zero_filter
    
    # Params
    n_samples = samples.shape[0]
    D = int(samples.shape[1] / 3)

    # Reshape 
    r = samples[:, :D]                             # red component
    g = samples[:, D:2*D]                          # green component
    b = samples[:, 2*D:]  
    samples = np.stack([r, g, b], axis=2)

    return samples.reshape(n_samples, 32, 32, 3)

def MNIST_sample_to_image(samples):
    n_samples = samples.shape[0]
    return samples.reshape(n_samples, 28, 28)