import torch 
import numpy as np
# Adapted from https://github.com/makeyourownalgorithmicart/makeyourownalgorithmicart/blob/master/blog/perlin_gradient_noise/1d_perlin.ipynb

def blend(x):
  return 6*x**5 - 15*x**4 + 10*x**3

def noise(scale, samples):
    # create a list of 2d vectors
    angles = torch.rand(scale) * 2*np.pi
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)

    x = torch.linspace(0, scale-1, samples)
    noise_values = []
    for value in x[:-1]:
        i = int(np.floor(value))
        lower, upper = gradients[i], gradients[i+1]

        dot1 = torch.dot(lower, torch.tensor([value - i, 0]))
        dot2 = torch.dot(upper, torch.tensor([value - i - 1, 0]))
        # TODO: review interpolation
        k1 = blend(value-i)
        k2 = blend(i+1 - value)
        interpolated = k1*dot2 + k2*dot1 
        noise_values.append(interpolated.item())
    
    noise_values.append(0.0)
    return torch.tensor(noise_values)

def perlin_noise(samples, scale=10, octaves=1, p=1):
    pnoise = 0
    for i in range(octaves):
        partial = noise(2**i * scale, samples)/(p**i)
        pnoise = partial + pnoise
    return pnoise



# def checker(res, scale=10):
#     tensors = tuple(3 * [torch.linspace(-1, 1, steps=res)])
#     grid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
#     grid = grid.reshape(-1, 3)

#     volume = torch.sin(scale * grid)
#     volume = grid[:, 0] * grid[:, 1] * grid[:, 2]
#     volume[volume < 0] = 0.0
#     volume[volume > 0] = 1.0
#     return volume.view(res, res, res).numpy()

def checker(texsize, cubesize):
    # Create a 3D grid of coordinates
    x, y, z = np.mgrid[0:texsize, 0:texsize, 0:texsize]

    # Generate a sine wave pattern in each dimension
    sine_x = np.sin(2 * np.pi * x / cubesize)
    sine_y = np.sin(2 * np.pi * y / cubesize)
    sine_z = np.sin(2 * np.pi * z / cubesize)

    # Combine the sine waves and threshold to create checkerboard pattern
    return ((sine_x + sine_y + sine_z) > 0).astype(np.float32)


def solid_texture(N, noise_scale=0.1):
    x, y, z = np.ogrid[-1:1:N*1j, -1:1:N*1j, -1:1:N*1j]
    r = np.sqrt(x**2 + y**2 + z**2)
    solid = np.zeros_like(r)
    solid[r < 0.5] = 1
    noise = noise_scale * np.random.randn(N, N, N)
    solid += noise
    solid[solid < 0] = 0
    solid[solid > 1] = 1
    return solid.astype(np.float32)


if __name__ == '__main__':
    from PIL import Image

    # Define texture size and cube size
    texture_size = 128
    cube_size = 16

    

    # Combine the sine waves and threshold to create checkerboard pattern
    # board = ((sine_x + sine_y + sine_z) > 0).astype(np.uint8) * 255
    # print(board.shape)
    board = (solid_texture(texture_size, 0.4) * 255).astype(np.uint8)
    Image.fromarray(board[0, :, :]).save('x.png')
    Image.fromarray(board[:, 0, :]).save('y.png')
    Image.fromarray(board[:, :, 0]).save('z.png')
    # # Convert NumPy array to PIL image
    # image = Image.fromarray(checkerboard)

    # # Save the generated texture
    # image.save('checkerboard_sine_3d.png')

    