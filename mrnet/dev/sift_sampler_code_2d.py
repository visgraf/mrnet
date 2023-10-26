import numpy as np
import matplotlib.pyplot as plt
import cv2


def sample_points_disk(radius, num_points):

    r = np.random.uniform(low=0, high=radius, size=num_points)
    theta = np.random.uniform(low=0, high=2*np.pi, size=num_points)  # angle

    x = np.sqrt(r) * np.cos(theta)
    y = np.sqrt(r) * np.sin(theta)

    np.stack([x, y], axis=1)

    return np.stack([x, y], axis=1)


def sample_nearby_points_disk(sift_points, radius, num_points_in_disk):

    points_in_disk = sample_points_disk(radius, num_points_in_disk)

    sift_points = sift_points[:, None, :]
    points_in_disk = points_in_disk[None, :, :]

    points_in_disk = points_in_disk + sift_points

    points_in_disk = points_in_disk.reshape(-1, 2)

    total_points = np.concatenate(
        [points_in_disk, sift_points.reshape(-1, 2)], axis=0)

    return total_points


def get_samples_sift(image, threshold=1e-20, radius=100., num_points_in_disk=10):

    image = np.transpose(image, (1, 2, 0))*255
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = image.astype(np.uint8)

    sift = cv2.SIFT_create()
    sift.setContrastThreshold(threshold)

    list_kp = sift.detect(image, None)

    list_samples = [np.array(kp.pt) for kp in list_kp]

    samples = np.stack(list_samples, axis=0, dtype=np.float32)

    visualize_samples(image, samples)

    samples = sample_nearby_points_disk(
        samples, radius, num_points_in_disk)

    visualize_samples(image, samples)

    samples[:, 0] = samples[:, 0] / image.shape[0]
    samples[:, 1] = samples[:, 1] / image.shape[1]

    samples = 2*samples - 1

    print(samples.shape)

    return samples


def visualize_samples(image, samples):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.imshow(image)

    # Extract x and y coordinates from the samples list
    x_coords, y_coords = zip(*samples)

    print(samples.shape)
    ax.scatter(x_coords, y_coords, c='r', s=0.5)

    plt.show()
