import numpy as np
import cv2

INTERNAL_DTYP = np.float32


def generating_kernel(a):
    kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    return np.outer(kernel, kernel)


def convolve(image, kernel=generating_kernel(0.4)):
    return cv2.filter2D(image, -1, kernel, None, borderType=cv2.BORDER_REFLECT_101)


def gaussian_pyramid(images, levels):
    pyramid = [images.astype(INTERNAL_DTYP)]
    num_images = images.shape[0]

    for _ in range(levels):
        level = []
        for layer in range(images.shape[0]):
            level.append(cv2.pyrDown(pyramid[-1][layer]))
        pyramid.append(np.asarray(level))

    return pyramid


def laplacian_pyramid(images, levels):
    gaussian = gaussian_pyramid(images, levels)

    pyramid = [gaussian[-1]]
    for level in range(len(gaussian) - 1, 0, -1):
        gauss = gaussian[level - 1]
        level_array = []
        for layer in range(images.shape[0]):
            gauss_layer = gauss[layer]
            expanded = cv2.pyrUp(gaussian[level][layer])
            if expanded.shape != gauss_layer.shape:
                expanded = expanded[: gauss_layer.shape[0], : gauss_layer.shape[1]]
            level_array.append(gauss_layer - expanded)
        pyramid.append(np.asarray(level_array))

    return pyramid[::-1]


def collapse(pyramid):
    image = pyramid[-1]
    for layer in pyramid[-2::-1]:
        expanded = cv2.pyrUp(image)
        if expanded.shape != layer.shape:
            expanded = expanded[: layer.shape[0], : layer.shape[1]]
        image = expanded + layer

    return image


def get_probabilities(gray_image):
    levels, counts = np.unique(gray_image, return_counts=True)
    probabilities = np.zeros((256,), dtype=INTERNAL_DTYP)
    probabilities[levels] = counts.astype(INTERNAL_DTYP) / counts.sum()
    return probabilities


def entropy_deviation(image, kernel_size):
    def _area_entropy(area, probabilities):
        levels = area.flatten()
        return -1.0 * (levels * np.log(probabilities[levels])).sum()

    def _area_deviation(area):
        average = np.average(area)  # .astype(INTERNAL_DTYP)
        return np.square(area - average).sum() / area.size

    probabilities = get_probabilities(image)
    pad_amount = int((kernel_size - 1) / 2)
    padded_image = cv2.copyMakeBorder(
        image, pad_amount, pad_amount, pad_amount, pad_amount, cv2.BORDER_REFLECT101
    )
    entropies = np.zeros(image.shape[:2], dtype=INTERNAL_DTYP)
    deviations = np.zeros(image.shape[:2], dtype=INTERNAL_DTYP)
    offset = np.arange(-pad_amount, pad_amount + 1)
    for row in range(deviations.shape[0]):
        for column in range(deviations.shape[1]):
            area = padded_image[
                row + pad_amount + offset[:, np.newaxis], column + pad_amount + offset
            ]
            entropies[row, column] = _area_entropy(area, probabilities)
            deviations[row, column] = _area_deviation(area)

    return entropies, deviations


def get_fused_base_channel(images, kernel_size, channel):
    entropies = np.zeros(images.shape[:3], dtype=INTERNAL_DTYP)
    deviations = np.zeros(images.shape[:3], dtype=INTERNAL_DTYP)
    for layer in range(images.shape[0]):
        gray_image = images[layer][:, :, channel].astype(np.uint8)
        probabilities = get_probabilities(gray_image)
        entropy, deviation = entropy_deviation(gray_image, kernel_size)
        entropies[layer] = entropy
        deviations[layer] = deviation

    best_e = np.argmax(entropies, axis=0)
    best_d = np.argmax(deviations, axis=0)
    fused = np.zeros(images.shape[1:-1], dtype=INTERNAL_DTYP)

    for layer in range(images.shape[0]):
        fused += np.where(best_e[:, :] == layer, images[layer][:, :, channel], 0)
        fused += np.where(best_d[:, :] == layer, images[layer][:, :, channel], 0)

    return (fused / 2).astype(images.dtype)


def get_fused_base(images, kernel_size):
    fused = np.zeros(images.shape[1:], dtype=INTERNAL_DTYP)
    for channel in range(images.shape[-1]):
        fused[:, :, channel] = get_fused_base_channel(images, kernel_size, channel)
    return fused


def fuse_pyramids(pyramids, kernel_size):
    fused = [get_fused_base(pyramids[-1], kernel_size)]
    for layer in range(len(pyramids) - 2, -1, -1):
        fused.append(get_fused_laplacian(pyramids[layer]))

    return fused[::-1]


def get_fused_laplacian_channel(laplacians, channel):
    layers = laplacians.shape[0]
    region_energies = np.zeros(laplacians.shape[:3], dtype=INTERNAL_DTYP)

    for layer in range(layers):
        gray_lap = laplacians[layer][:, :, channel]
        region_energies[layer] = region_energy(gray_lap)

    best_re = np.argmax(region_energies, axis=0)
    fused = np.zeros(laplacians.shape[1:-1], dtype=laplacians.dtype)

    for layer in range(layers):
        fused += np.where(best_re[:, :] == layer, laplacians[layer][:, :, channel], 0)

    return fused


def get_fused_laplacian(laplacians):
    fused = np.zeros(laplacians.shape[1:], dtype=laplacians.dtype)
    for channel in range(laplacians.shape[-1]):
        fused[:, :, channel] = get_fused_laplacian_channel(laplacians, channel)
    return fused


def region_energy(laplacian):
    return convolve(np.square(laplacian))


def get_pyramid_fusion(images, min_size=32):
    smallest_side = min(images[0].shape[:2])
    depth = int(np.log2(smallest_side / min_size))
    kernel_size = 5

    pyramids = laplacian_pyramid(images, depth)
    fusion = fuse_pyramids(pyramids, kernel_size)
    return collapse(fusion)
